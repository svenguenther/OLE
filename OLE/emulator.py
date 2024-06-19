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
import jax

from copy import deepcopy
from copy import copy

from functools import partial

from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.data_cache import DataCache
from OLE.gp_predicter import GP_predictor
from OLE.likelihood import Likelihood

from OLE.plotting import plot_loglikes

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

    # likelihood calculation
    likelihood: Likelihood

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

    def initialize(self, likelihood=None, ini_state=None, **kwargs):
        # default hyperparameters
        defaulthyperparameters = {
            # kernel
            'kernel': 'RBF',

            # kernel fitting frequency. Every n-th state the kernel parameters are fitted.
            'kernel_fitting_frequency': 4,
            'test_noise_levels': 100,
            # the number of data points in cache before the emulator is to be trained
            'min_data_points': 80,


            # veto list of parameters which are not to be emulated.
            'veto_list': None,

            # logfile for emulator
            'logfile': None,

            # Load initial state from cache. Usually, before we can use the emulator we need to have a state in the cache to determine the dimensionality of the quantities.
            # However, if we have a state in the cache, we can load the initial state from the cache. And not a single theory call has to be made.
            'load_initial_state': False,

            ## TESTING:

            # should we test the performance of the emulator once it is trained? If Flase, the following parameters are not needed
            'test_emulator': True,

            # numer of test samples to determine the quality of the emulator
            'N_quality_samples': 5,

            # path to a directory with data covmats to do better normalization. If given, the emulator will search for data covmats in this directoy and if one is found, it will be used for normalization
            'data_covmat_directory': None,

            # here we define the quality threshold for the emulator. If the emulator is below this threshold, it is retrained. We destinguish between a constant, a linear and a quadratic threshold
            'quality_threshold_constant': 0.1,
            'quality_threshold_linear': 0.01,
            'quality_threshold_quadratic': 0.0001,

            'error_tolerance' : 1.,
            'sparse_GP_points' : 0.,
            'test_noise_levels_counter' : 50,
            # the radius around the checked points for which we do not need to check the quality criterium
            'quality_points_radius': 0.0,

            # plotting directory
            'plotting_directory': None,

            # only relevant for cobaya
            'cobaya_state_file': None, # TODO: put this somewhere else. This is only used in the cobaya wrapper
            'jit_threshold': 10, # number of samples to be emulated before we jit the emulator to accelerate it TODO: put this somewhere else

            # learn about the actual emulation task to estimate 'quality_threshold_quadratic'.
            'N_sigma': 6,
            'dimensionality': None, # if we give the dimensionality, the code estimates where we need to be accruate in the quality criterium (inside of N_sigma). Thus, we can estimate the quality_threshold_quadratic, in a way, that it becomes dominant over the linear at this point!


        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        # If load_initial_state is True, we load the initial state from the cache
        if self.hyperparameters['load_initial_state']:
            # check that cache file exists, otherwise raise error
            if not os.path.exists(self.hyperparameters['cache_file']):
                raise FileNotFoundError("Cache file does not exist. Cannot load initial state from cache. Set 'load_initial_state' to False until the cache file is created.")

            self.data_cache.load_cache()
            self.ini_state = self.data_cache.states[0].copy()
        else:
            self.ini_state = ini_state

        # Add initial state to the data cache
        self.data_cache.initialize(ini_state)

        # Initialize the emulator with an initial state. In this state the parameters and the quantitites which are to be emulated are defined. They also come with an example value to determine the dimensionality of the quantities.
        ini_state = self.ini_state

        # if there is a input_parameters list in the kwargs, we use this list to determine the order of the input parameters
        if 'input_parameters' in kwargs:
            self.input_parameters = kwargs['input_parameters']
        else:
            self.input_parameters = list(ini_state['parameters'].keys())

        self.likelihood = likelihood
        if self.likelihood is not None:
            self.likelihood.initialize(**kwargs)

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

        # if 'dimensionality' is given we need to estimate the quality_threshold_quadratic
        if self.hyperparameters['dimensionality'] is not None:
            from scipy.stats import chi2
            # we need to estimate the quality_threshold_quadratic
            # we use the N_sigma to estimate the quality_threshold_quadratic
            # we estimate the quality_threshold_quadratic such that it becomes dominant over the linear term at the N_sigma point

            # up to which p value do we want to be accurate?
            p_val = chi2.cdf(self.hyperparameters['N_sigma']**2, 1)

            if p_val == 1.0:
                self.warning("N_sigma is too large. The p value is 1.0 due to double precision. The estimated delta_loglike is not accurate. Thus, we set N_sigma = 8!")
                p_val = chi2.cdf(8**2, 1)

            # the corresponding loglike
            accuracy_loglike = chi2.ppf(p_val, self.hyperparameters['dimensionality'])/2

            # at this point the (constant + linear) term is equal to the quadratic term
            self.hyperparameters['quality_threshold_quadratic'] = (self.hyperparameters['quality_threshold_constant'] + self.hyperparameters['quality_threshold_linear']*accuracy_loglike)/accuracy_loglike**2

            self.debug("Quality threshold quadratic: ", self.hyperparameters['quality_threshold_quadratic'])

        # if we have a data_covmat_directory we search for the data covmat in this directory
        self.data_covmats = {quantity_name: None for quantity_name in ini_state['quantities'].keys()}
        if self.hyperparameters['data_covmat_directory'] is not None:
            # check whether the directory exists
            if not os.path.exists(self.hyperparameters['data_covmat_directory']):
                self.warning("Data covmat directory does not exist. Cannot use data covmat for normalization.")
            else:
                # search for the data covmat in the directory. They should be named <quantity_name>.covmat
                for quantity_name, quantity in ini_state['quantities'].items():
                    covmat_file = self.hyperparameters['data_covmat_directory'] + '/' + quantity_name + '.covmat'
                    if os.path.exists(covmat_file):
                        self.debug("Data covmat found for quantity %s. Using it for normalization.", quantity_name)
                        self.data_covmats[quantity_name] = np.loadtxt(covmat_file)
                    else:
                        self.debug("Data covmat not found for quantity %s. Not using it for normalization.", quantity_name)


        # Here: Create the emulators for the different quantities
        self.emulators = {}

        for quantity_name, quantity in ini_state['quantities'].items():

            # check whether the quantity is in the veto list
            if self.hyperparameters['veto_list'] is not None:
                if quantity_name in self.hyperparameters['veto_list']:
                    continue
            
            # initialize the emulator for the quantity
            self.emulators[quantity_name] = GP_predictor(quantity_name, debug=self.debug_mode)
            self.emulators[quantity_name].initialize(ini_state,data_covmat=self.data_covmats[quantity_name], **kwargs)

        # initialize the points which were already tested with the quality criterium
        self.quality_points = []
        

        # if we fulfill the minimum number of data points, we train the emulator
        if len(self.data_cache.states) >= self.hyperparameters['min_data_points']:
            self.data_cache.load_cache()
            self.train()

        # this counter is used to determine the number of continously successful emulator calls
        self.continuous_successful_calls = 0

        # max loglike encountered in the run
        self.max_loglike_encountered = -np.inf
        
        pass

    def add_state(self, state):
        # returns 2 if the state was added to the data cache and the emulator was trained
        # returns 1 if the state was added to the data cache and the emulator was updated
        # returns 0 if the state was not added to the data cache and the emulator was not updated

        # new state
        new_state = deepcopy(state)

        # remove parameters which are not in the input_parameters list:
        for key in list(state['parameters'].keys()):
            if key not in self.input_parameters:
                del new_state['parameters'][key]

        # Add a state to the emulator. This means that the state is added to the data cache and the emulator is retrained.
        state_added = self.data_cache.add_state(new_state)

        if state_added:
            # write to log that the state was added
            _ = "State added to emulator: " + "; ".join([key+ ': ' +str(value) for key, value in new_state['parameters'].items()]) + " at loglike: " + str(new_state['loglike']) + " max. loglike: " + str(self.data_cache.max_loglike) + "\n"
            self.write_to_log(_)
            # write to log the current size of the data cache
            _ = "Current data cache size: %d\n" % len(self.data_cache.states)
            self.write_to_log(_)
        else:
            # write to log that the state was not added
            _ = "State not added to emulator: " + "; ".join([key+ ': ' +str(value) for key, value in new_state['parameters'].items()]) + " at loglike: " + str(new_state['loglike']) + " max. loglike: " + str(self.data_cache.max_loglike) + "\n"
            self.write_to_log(_)
            # write to log the current size of the data cache
            _ = "Current data cache size: %d\n" % len(self.data_cache.states)
            self.write_to_log(_)
        
        # if the emulator is already trained, we can add the new state to the GP without fitting the Kernel parameters
        emulator_updated = False
        if self.trained and state_added:
            self.added_data_points += 1
            if self.added_data_points%self.hyperparameters['kernel_fitting_frequency'] == 0:
                self.train()
                emulator_updated = True
                self.hyperparameters['test_noise_levels_counter'] = 100
            else:
                if self.hyperparameters['sparse_GP_points'] == 0:
                    self.update()
                    emulator_updated = True
                #else: do nothing as updating does not really improve the sparese GP anyways
        
            
        if state_added:
            self.continuous_successful_calls = 0
            return True , emulator_updated
        else:
            return False , emulator_updated

    def update(self):
        # Update the emulator. This means that the emulator is retrained.
        # Load the data from the cache.
        self.debug("Loading data from cache")

        self.write_to_log("Update emulator\n")

        # Train the emulator.
        for quantity, emulator in self.emulators.items():
            self.debug("Updating emulator for quantity %s", quantity)
            input_data_raw = self.data_cache.get_parameters()
            output_data_raw = self.data_cache.get_quantities(quantity)

            input_data_raw_jax = jnp.array(input_data_raw)
            output_data_raw_jax = jnp.array(output_data_raw)

            # load data into emulators
            emulator.load_data(input_data_raw_jax, output_data_raw_jax)

            # normalize and compress data
            emulator.data_processor.normalize_training_data()
            emulator.data_processor.compress_training_data()

            del input_data_raw
            del output_data_raw

            del input_data_raw_jax
            del output_data_raw_jax
            
            
        self.trained = True
        pass

    def train(self):
        # Load the data from the cache.
        self.debug("Loading data from cache")

        self.write_to_log("Training emulator\n")

        
        # Train the emulator.
        input_data_raw = self.data_cache.get_parameters()
        input_data_raw_jax = jnp.array(input_data_raw)

        for quantity, emulator in self.emulators.items():
            self.info("Start training emulator for quantity %s", quantity)
            output_data_raw = self.data_cache.get_quantities(quantity)

            output_data_raw_jax = jnp.array(output_data_raw)

            # load data into emulators
            emulator.load_data(input_data_raw_jax, output_data_raw_jax)

            del output_data_raw_jax
            del output_data_raw

            # compute normalization and compression and apply it to the data
            self.debug("Compute normalization and compression for quantity %s", quantity)
            emulator.data_processor.compute_normalization()
            emulator.data_processor.normalize_training_data()
            self.debug("Normalization done for quantity %s", quantity)

            emulator.data_processor.compute_compression()
            emulator.data_processor.compress_training_data()

            emulator.initialize_training() # fill basic structures so we can set errors accurately
            
        self.set_error()

        for quantity, emulator in self.emulators.items():
            self.debug("Train GP for quantity %s", quantity)

            emulator.finalize_training()

            

        del input_data_raw_jax
        del input_data_raw

        self.trained = True

        jax.clear_backends()




        # if we have a plotting directory we plot the loglikelihood
        if self.hyperparameters['plotting_directory'] is not None:
            loglikes = jnp.array(self.data_cache.get_loglikes())
            parameters = jnp.array(self.data_cache.get_parameters())

            for i in range(len(parameters[0])):
                plot_loglikes(loglikes[:,0], parameters[:,i], self.input_parameters[i], self.hyperparameters['plotting_directory']+'/loglike_'+str(i)+'.png')
        pass

    def set_data_covmats(self, data_covmats):
        # set the data covmats for the different quantities
        for quantity_name, data_covmat in data_covmats.items():
            self.data_covmats[quantity_name] = data_covmat
            self.emulators[quantity_name].data_processor.data_covmat = data_covmat

    # @partial(jax.jit, static_argnums=0)
    def emulate(self, parameters):
        # Prepare output state
        output_state = {'quantities':{}} #self.ini_state.copy()
        output_state['parameters'] = parameters

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])
        
        for quantity, emulator in self.emulators.items():
            emulator_output = emulator.predict(input_data)
            output_state['quantities'][quantity] = emulator_output

        # write to log
        # self.write_parameter_dict_to_log(parameters)

        return output_state
    

    # function to get N samples from the same input parameters
    # @partial(jax.jit, static_argnums=0)
    def emulate_samples(self, parameters, RNGkey):
        # Prepare list of N output states
        state = {'parameters': {}, 'quantities': {}}
        state['parameters'] = parameters
        output_states = [deepcopy(state) for i in range(self.hyperparameters['N_quality_samples'])]

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])

        for quantity, emulator in self.emulators.items():
            emulator_output, RNGkey = emulator.sample_prediction(input_data, N=self.hyperparameters['N_quality_samples'], RNGkey=RNGkey)
            
            
            for i in range(self.hyperparameters['N_quality_samples']):
                output_states[i]['quantities'][quantity] = emulator_output[i,:]

        return output_states, RNGkey

    def emulate_samples_noiseFree(self, parameters, RNGkey):
        # Prepare list of N output states
        state = {'parameters': {}, 'quantities': {}}
        state['parameters'] = parameters
        output_states = [deepcopy(state) for i in range(self.hyperparameters['N_quality_samples'])]

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])




        for quantity, emulator in self.emulators.items():
            emulator_output, RNGkey = emulator.sample_prediction_noiseFree(input_data, N=self.hyperparameters['N_quality_samples'], RNGkey=RNGkey)
            
            
            for i in range(self.hyperparameters['N_quality_samples']):
                output_states[i]['quantities'][quantity] = emulator_output[i,:]

        return output_states, RNGkey
    

    # OUTDATED
    # function to get 1 sample from the same input parameters
    # @partial(jax.jit, static_argnums=0)
    # def emulate_sample(self, parameters, RNGkey=jax.random.PRNGKey(time.time_ns())):
    #     # Prepare list of N output states

    #     state = {'parameters': {}, 'quantities': {}}
    #     state['parameters'] = parameters

    #     # Emulate the quantities for the given parameters.
    #     input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])

    #     for quantity, emulator in self.emulators.items():
    #         emulator_output, RNGkey = emulator.sample_prediction(input_data, RNGkey=RNGkey)
            
    #         state['quantities'][quantity] = emulator_output[0,:]

    #     return state, RNGkey

    # function simply restes all errors for the GPs

    #def reset_error(self):
  
    #    for quantity_name, quantity in self.ini_state['quantities'].items():
    #        self.emulators[quantity_name].reset_error() 

    def set_error(self):

        if self.hyperparameters['sparse_GP_points'] > 0.:
            # require errors for sparse GP's
            self.hyperparameters['error_tolerance'] = 1.


        if self.hyperparameters['error_tolerance'] == 0.:
            for quantity_name, quantity in self.ini_state['quantities'].items():
                self.emulators[quantity_name].disable_error() 

        else:
            #num_GPs = 0
            #for quantity_name, quantity in self.ini_state['quantities'].items():
            #    self.emulators[quantity_name].reset_error() 
            #    num_GPs += self.emulators[quantity_name].num_GPs
           
                
        
            #for index in range(len(self.data_cache.states)):
            #    state = self.data_cache.states[index]
                
            #    delta_loglike = max(self.data_cache.max_loglike) - state['loglike']
            #    acceptable_error = self.hyperparameters['quality_threshold_constant'] + delta_loglike * self.hyperparameters['quality_threshold_linear'] + delta_loglike **2 * self.hyperparameters['quality_threshold_quadratic']
                            
                # we distribute this error equally among all GPs. Other options could be considered
            #    quantity_derivs = {}
        
            #    for name,val in state['quantities'].items():
            #        quantity_derivs[name] = jnp.array( self.likelihood.loglike_gradient(state, name))
            
            #    acceptable_error /= jnp.sqrt(num_GPs) * jnp.sqrt(100./self.hyperparameters['noise_percentage']) 
                                    
        
            #    if delta_loglike > 0.:
            #        for quantity_name, quantity in self.ini_state['quantities'].items():
            #            self.emulators[quantity_name].set_error(index,quantity_derivs[quantity_name],acceptable_error) 
                        
                
            # this could be in gp_predictor
            print('set the noise levels to ')
            for quantity_name, quantity in self.ini_state['quantities'].items():

                # check whether the quantity is in the veto list
                if self.hyperparameters['veto_list'] is not None:
                    if quantity_name in self.hyperparameters['veto_list']:
                        continue

                # Ali memory bug <3

                var = self.emulators[quantity_name].data_processor.output_pca_stds**2 * len(self.emulators[quantity_name].data_processor.output_data_emulator)
                # this is analytical for the variance of the components. since we compare to the total we might drop the len of data
            
                total_var = jnp.sum(var)
                variance_tolerance = 1. - self.emulators[quantity_name].data_processor.hyperparameters['explained_variance_cutoff']

                for i in range(self.emulators[quantity_name].num_GPs):
                    relative_variance = var[i]/total_var
                    #error = variance_tolerance / relative_variance * self.hyperparameters['noise_percentage']
                    error = ((variance_tolerance / relative_variance) ** 2. ) / len(var)
                    
                    self.emulators[quantity_name].GPs[i].hyperparameters['error_tolerance'] = error
                    print(self.emulators[quantity_name].GPs[i].hyperparameters['error_tolerance'] )

    

    def check_quality_criterium(self, loglikes, parameters, reference_loglike = None, write_log = True):
        # check whether the emulator is good enough to be used
        # if the emulator is not yet trained, we return False
        if not self.trained:
            self.continuous_successful_calls = 0
            return False

        # if the emulator is trained, we check the quality criterium
        # we check whether the loglikes are within the quality criterium
        if reference_loglike is None:
            mean_loglike = jnp.mean(loglikes)
        else:
            mean_loglike = reference_loglike
        std_loglike = jnp.std(loglikes)

        max_loglike = max(self.data_cache.max_loglike, self.max_loglike_encountered)


        delta_loglike = jnp.abs(mean_loglike - max_loglike)

        # write testing to log

        # if the mean loglike is above the maximum found loglike we only check the constant term
        if mean_loglike > max_loglike:
            if std_loglike > self.hyperparameters['quality_threshold_constant']:
                self.debug("Emulator quality criterium NOT fulfilled")
                _ = "Quality criterium NOT fulfilled; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, delta loglikes: " % (max_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
                if write_log: 
                    self.write_to_log(_)
                return False
        else:
            # calculate the absolute difference between the mean loglike and the maximum loglike
            delta_loglike = jnp.abs(mean_loglike - max_loglike)
            
            # the full criterium 
            if std_loglike > self.hyperparameters['quality_threshold_constant'] + self.hyperparameters['quality_threshold_linear']*delta_loglike + self.hyperparameters['quality_threshold_quadratic']*delta_loglike**2:
                self.debug("Emulator quality criterium NOT fulfilled")
                _ = "Quality criterium NOT fulfilled; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, delta loglikes: " % (max_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
                if write_log: 
                    self.write_to_log(_)
                return False

        self.debug("Emulator quality criterium fulfilled")
        _ = "Quality criterium fulfilled; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, delta loglikes: " % (max_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
        if write_log: 
            self.write_to_log(_)
        self.continuous_successful_calls += 1
        
        # if any of the loglikes is above the maximum loglike, we need to update the maximum loglike
        if jnp.any(mean_loglike > max_loglike):
            self.max_loglike_encountered = mean_loglike

        return True
    

    def require_quality_check(self, parameters):
        # check whether the emulator is expected to perform well for the given parameters.
        # if we return False, the emulator is expected to perform well and we do not need to check the quality criterium
        # if we return True, the emulator is expected to perform poorly and we need to check the quality criterium

        # if we do not require a quality check, we return False
        if not self.hyperparameters['test_emulator']:
            self.debug("Quality check not required. Test emulator is False")
            self.write_to_log("Quality check not required. Test emulator is False")
            self.continuous_successful_calls += 1
            return False

        # The idea is that we collect all points which were checked by the quality criterium and then check whether the new point is within the convex hull of the checked points.
        input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])

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
                self.write_to_log("Quality check not required. Point is close to already checked points: " + " ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + "\n")
                return False
            else:
                self.debug("Quality check required. Point is far from already checked points")
                self.write_to_log("Quality check required. Point is far from already checked points: " + " ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + "\n")
                return True
        
    def add_quality_point(self, parameters):
        # if there is no distance to check, there is no point in storing these points
        if self.hyperparameters['quality_points_radius'] == 0.0:
            return

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
        if self.hyperparameters['logfile'] is None:
            return
        
        file_name = self.hyperparameters['logfile']+'_'+str(get_mpi_rank())+'.log'
        with open(file_name, 'a') as logfile:
            # add timestamp to message
            message = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + message
            logfile.write(message)

        pass

    def write_parameter_dict_to_log(self, parameters):
        # write the parameters to the logfile
        _ = "Emulated state: " + " ".join([str(key) + ": " + str(value) for key, value in parameters.items()]) + "\n"
        self.write_to_log(_)
        pass




