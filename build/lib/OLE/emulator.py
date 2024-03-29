"""
Emulator instance. This is the main class that interacts with the sampling codes and provides their interfaces.

It is aware of the quanities which are to be emulated and their dimensionalities. It stores all hyperparameters.

It creates and accesses the data cache and creates the individual emulators for the different quantities.

"""

import numpy as np
import os
import pickle
import time
import warnings
import logging
import jax.numpy as jnp

from copy import deepcopy

from .utils.base import BaseClass
from .utils.mpi import *
from .data_cache import DataCache
from .gp_predicter import GP_predictor

class Emulator(BaseClass): 

    # The emulators are a dictionary of the individual emulators for the different quantities. The keys are the names of the quantities.
    emulators: dict

    # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
    hyperparameters: dict

    # Initialization state
    ini_state: dict

    # The data cache is a dictionary of the individual states. The keys are the names of the states.
    data_cache: DataCache

    # Trained flag states whether the emulator has been trained.
    trained: bool

    # list of input parameters
    input_parameters: list

    def __init__(self, data_cache = None, **kwargs):
        super().__init__("Emulator", **kwargs)

        if data_cache is None:
            self.data_cache = DataCache(**kwargs)
        else:
            self.data_cache = data_cache

        self.trained = False

        self.added_data_points = 0



        self.info("Emulator initialized")
        self.debug("Emulator initialized")

        pass

    def initialize(self, ini_state, **kwargs):
        # default hyperparameters
        defaulthyperparameters = {
            # kernel
            'kernel': 'RBF',

            # kernel fitting frequency. Every n-th state the kernel parameters are fitted.
            'kernel_fitting_frequency': 4,

            # the number of data points in cache before the emulator is to be trained
            'min_data_points': 80,

            # numer of test samples to determine the quality of the emulator
            'N_quality_samples': 5,

            # here we define the quality threshold for the emulator. If the emulator is below this threshold, it is retrained. We destinguish between a constant, a linear and a quadratic threshold
            'quality_threshold_constant': 0.1,
            'quality_threshold_linear': 0.0,
            'quality_threshold_quadratic': 0.0,

            # the radius around the checked points for which we do not need to check the quality criterium
            'quality_points_radius': 0.3,

            # veto list of parameters which are not to be emulated.
            'veto_list': None,

            # logfile for emulator
            'logfile': None,

        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        # Add initial state to the data cache
        self.data_cache.initialize(ini_state)

        # Initialize the emulator with an initial state. In this state the parameters and the quantitites which are to be emulated are defined. They also come with an example value to determine the dimensionality of the quantities.
        self.ini_state = ini_state

        # if there is a input_parameters list in the kwargs, we use this list to determine the order of the input parameters
        if 'input_parameters' in kwargs:
            self.input_parameters = kwargs['input_parameters']
        else:
            self.input_parameters = list(ini_state['parameters'].keys())


        # A state dictionary is a nested dictionary with the following structure:
        # state = {
        #     "parameters": {
        #         "parameter1": [123],
        #         "parameter2": [456],
        #         ...
        #     },
        #     "quantities": {
        #         "quantity1": [element1, element2, ...],
        #         "quantity2": [element1, element2, ...],
        #         ...
        #     },
        #     "loglike": 123, (or None if not available)
        # }
            
        # remove the parameters which are not in the input_parameters list
        for key in list(ini_state['parameters'].keys()):
            if key not in self.input_parameters:
                del ini_state['parameters'][key]

        # Here: Create the emulators for the different quantities
        self.emulators = {}

        for quantity_name, quantity in ini_state['quantities'].items():

            # check whether the quantity is in the veto list
            if self.hyperparameters['veto_list'] is not None:
                if quantity_name in self.hyperparameters['veto_list']:
                    continue
            
            # initialize the emulator for the quantity
            self.emulators[quantity_name] = GP_predictor(quantity_name, debug=self.debug_mode)
            self.emulators[quantity_name].initialize(ini_state, **kwargs)

        # initialize the points which were already tested with the quality criterium
        self.quality_points = []
        
        pass

    def add_state(self, state):
        # new state
        new_state = deepcopy(state)

        # remove parameters which are not in the input_parameters list:
        for key in list(state['parameters'].keys()):
            if key not in self.input_parameters:
                del new_state['parameters'][key]

        # Add a state to the emulator. This means that the state is added to the data cache and the emulator is retrained.
        state_added = self.data_cache.add_state(new_state)

        if state_added:
            self.added_data_points += 1
        
        # if the emulator is already trained, we can add the new state to the GP without fitting the Kernel parameters
        if self.trained and state_added:
            if self.added_data_points%self.hyperparameters['kernel_fitting_frequency'] == 0:
                self.train()
            else:
                self.update()

        # here we need to interact with the cache
        pass

    def update(self):
        # Update the emulator. This means that the emulator is retrained.
        # Load the data from the cache.
        self.debug("Loading data from cache")

        # Train the emulator.
        for quantity, emulator in self.emulators.items():
            self.debug("Updating emulator for quantity %s", quantity)
            input_data_raw = self.data_cache.get_parameters()
            output_data_raw = self.data_cache.get_quantities(quantity)

            input_data_raw = np.array(input_data_raw)
            output_data_raw = np.array(output_data_raw)

            # load data into emulators
            emulator.load_data(input_data_raw, output_data_raw)

            # normalize and compress data
            emulator.data_processor.normalize_training_data()
            emulator.data_processor.compress_training_data()
            
        self.trained = True
        pass

    def train(self):
        # Load the data from the cache.
        self.debug("Loading data from cache")

        # Train the emulator.
        for quantity, emulator in self.emulators.items():
            self.debug("Start training emulator for quantity %s", quantity)
            input_data_raw = self.data_cache.get_parameters()
            output_data_raw = self.data_cache.get_quantities(quantity)

            input_data_raw = np.array(input_data_raw)
            output_data_raw = np.array(output_data_raw)

            # load data into emulators
            emulator.load_data(input_data_raw, output_data_raw)

            # compute normalization and compression and apply it to the data
            self.debug("Compute normalization and compression for quantity %s", quantity)
            emulator.data_processor.compute_normalization()
            emulator.data_processor.normalize_training_data()
            self.debug("Normalization done for quantity %s", quantity)

            emulator.data_processor.compute_compression()
            emulator.data_processor.compress_training_data()

            self.debug("Train GP for quantity %s", quantity)
            emulator.train()
            
        self.trained = True
        pass

    def emulate(self, parameters):
        # Prepare output state
        output_state = self.ini_state.copy()
        output_state['parameters'] = parameters

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])
        
        for quantity, emulator in self.emulators.items():
            emulator_output = emulator.predict(input_data)
            output_state['quantities'][quantity] = emulator_output

        return output_state
    
    # function to get N samples from the same input parameters
    def emulate_samples(self, parameters, N):
        # Prepare list of N output states
        output_states = [deepcopy(self.ini_state) for i in range(N)]
        for state in output_states:
            state['parameters'] = parameters

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])

        for quantity, emulator in self.emulators.items():
            emulator_output = emulator.sample_prediction(input_data, N)
            
            for i, output_state in enumerate(output_states):
                output_states[i]['quantities'][quantity] = emulator_output[i,:]

        return output_states
    
    def check_quality_criterium(self, loglikes):
        # check whether the emulator is good enough to be used
        # if the emulator is not yet trained, we return False
        if not self.trained:
            return False

        # if the emulator is trained, we check the quality criterium
        # we check whether the loglikes are within the quality criterium
        mean_loglike = jnp.mean(loglikes)
        std_loglike = jnp.std(loglikes)

        max_loglike = self.data_cache.max_loglike

        delta_loglike = jnp.abs(mean_loglike - max_loglike)

        # if the mean loglike is above the maximum found loglike we only check the constant term
        if mean_loglike > max_loglike:
            if std_loglike > self.hyperparameters['quality_threshold_constant']:
                self.debug("Emulator quality criterium NOT fulfilled")
                return False
        else:
            # calculate the absolute difference between the mean loglike and the maximum loglike
            delta_loglike = jnp.abs(mean_loglike - max_loglike)
            
            # the full criterium 
            if std_loglike > self.hyperparameters['quality_threshold_constant'] + self.hyperparameters['quality_threshold_linear']*delta_loglike + self.hyperparameters['quality_threshold_quadratic']*delta_loglike**2:
                self.debug("Emulator quality criterium NOT fulfilled")
                return False

        self.debug("Emulator quality criterium fulfilled")
        return True
    

    def require_quality_check(self, parameters):
        # check whether the emulator is expected to perform well for the given parameters.
        # if we return False, the emulator is expected to perform well and we do not need to check the quality criterium
        # if we return True, the emulator is expected to perform poorly and we need to check the quality criterium

        # The idea is that we collect all points which were checked by the quality criterium and then check whether the new point is within the convex hull of the checked points.
        input_data = jnp.array([[value[0] for key, value in parameters.items()]])

        # normalize the input data
        quantity = list(self.emulators.keys())[0]
        normalized_input_data = self.emulators[quantity].data_processor.normalize_input_data(input_data)

        # check whether the normalized input data is within a certain radius of the checked points
        if len(self.quality_points) == 0:
            self.debug("No quality points yet. Quality check required.")
            return True
        else:
            # calculate the distance between the normalized input data and the checked points
            distances = jnp.linalg.norm(normalized_input_data - jnp.array(self.quality_points), axis=-1)

            # if the distance is smaller than a threshold, we return True
            if jnp.any(distances < self.hyperparameters['quality_points_radius']):
                self.debug("Quality check not required. Point is close to already checked points")
                return False
            else:
                self.debug("Quality check required. Point is far from already checked points")
                return True
        
    def add_quality_point(self, parameters):
        input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])

        # normalize the input data
        quantity = list(self.emulators.keys())[0]
        normalized_input_data = self.emulators[quantity].data_processor.normalize_input_data(input_data)

        # add the normalized input data to the quality points
        self.quality_points.append(normalized_input_data)
        self.debug("Added quality point to list of quality points")

    
    def get_gradients(self, parameters):
        # Prepare output state
        output_state = self.ini_state.copy()
        output_state['parameters'] = parameters

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])
        
        for quantity, emulator in self.emulators.items():
            emulator_output = emulator.predict_gradients(input_data)
            output_state['quantities'][quantity] = emulator_output

        return output_state
    

    def write_to_log(self, message):
        # write the message to the logfile
        file_name = self.hyperparameters['logfile']+'_'+str(get_mpi_rank())+'.log'
        with open(file_name, 'a') as logfile:
            logfile.write(message)

        pass






