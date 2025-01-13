"""
Emulator instance. This is the main class that interacts with the sampling codes and provides their interfaces.

It is aware of the quanities which are to be emulated and their dimensionalities. It stores all hyperparameters.

It creates and accesses the data cache and creates the individual emulators for the different quantities.

"""
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pickle
import time
import warnings
import logging
import jax.numpy as jnp
import jax
import fasteners
import psutil
import gc

# set jax_enable_compilation_cache to False to avoid memory issues
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['NPROC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '1'




from copy import deepcopy
from copy import copy

import datetime

from functools import partial

from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.data_cache import DataCache
from OLE.gp_predicter import GP_predictor, GP
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

    # likelihood collection calculation
    likelihood_collection: dict

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

    def initialize(self, likelihood_collection=None, ini_state=None, **kwargs):
        super().initialize(**kwargs)
        # default hyperparameters
        defaulthyperparameters = {
            # kernel
            'kernel': 'RBF',

            # kernel fitting frequency. Every n-th state the kernel parameters are fitted.
            'kernel_fitting_frequency': 40,
            'test_noise_levels': 100,
            # the number of data points in cache before the emulator is to be trained
            'min_data_points': 80,


            # veto list of parameters which are not to be emulated.
            'skip_emulation_quantities': None,

            # logfile for emulator
            'logfile': None,

            # Load initial state from cache. Usually, before we can use the emulator we need to have a state in the cache to determine the dimensionality of the quantities.
            # However, if we have a state in the cache, we can load the initial state from the cache. And not a single theory call has to be made.
            'load_initial_state': False,

            ## TESTING:

            # numer of test samples to determine the quality of the emulator
            'N_quality_samples': 5,

            # path to a directory with data covmats to do better normalization. If given, the emulator will search for data covmats in this directoy and if one is found, it will be used for normalization
            'data_covmat_directory': None,
            'white_noise_ratio' : 1., 
            'sparse_GP_points' : 0.,
            'test_noise_levels_counter' : 50,
            # the radius around the checked points for which we do not need to check the quality criterium
            'quality_points_radius': 0.0,

            # plotting directory
            'plotting_directory': None,
            'testset_fraction': 0.1,

            # only relevant for cobaya
            'cobaya_state_file': None, # TODO: put this somewhere else. This is only used in the cobaya wrapper
            'jit_threshold': 20, # number of samples to be emulated before we jit the emulator to accelerate it TODO: put this somewhere else

            # learn about the actual emulation task to estimate 'quality_threshold_quadratic'.
            'N_sigma': 4,
            'dimensionality': None, # if we give the dimensionality, the code estimates where we need to be accruate in the quality criterium (inside of N_sigma). Thus, we can estimate the quality_threshold_quadratic, in a way, that it becomes dominant over the linear at this point!

            # max sigma for the quality criterium. If this is set, we will use the emulator only if we are 'reasonably' close to the best fit posterior. 
            # Otherwise it can happen for chains very far from the best fit, that they get unreasonable results
            'max_sigma': 20,

            # a dictionary for the likelihood settings
            'likelihood_collection_settings': {},

            # this setting is a flag whether to jit the emulator
            'jit': True,

            # print frequency for the emulator
            'status_print_frequency': 200,


            # Settings related with the quality criterium

            # here we define the quality threshold for the emulator. If the emulator is below this threshold, it is retrained. We destinguish between a constant, a linear and a quadratic threshold
            'quality_threshold_constant': 0.1,
            'quality_threshold_linear': 0.05,
            'quality_threshold_quadratic': 0.0001,

            # testing_strategy: 
            # 'test_all', (default) test always all points
            # 'test_early', test all points until the first 'test_early_points' points were consecutively successful. Stop testing afterwards.
            # NOT IMPLEMENTED YET: 'test_GP_criterium', use GP to determine which points to test
            # 'test_none', do not test the emulator
            # 'test_stochastic', do not test the emulator

            'testing_strategy': 'test_stochastic',

            # it can be useful to check every now and then the cache for new points in the late stage of the run. In particular, if we want a very very high R-1, we need to sychronize the emulators.
            # Thus, every 'check_cache_for_new_points' consecutive successful emulator calls, we check the cache for new points.
            'check_cache_for_new_points': 1000,

            # number of points to test early
            'test_early_points': 1000,

            # stocaistic testing parameters
            'test_stochastic_scale': 40, # sclae of the exponential testing ratio
            'test_stochastic_rate': None, # minimal rate of testing, even after the exponential testing rate is reached. if not set, it will be estimated by meassureing time of testing and the emulation time.
            'test_stochastic_testing_time_fraction': 0.15, # fraction of the time which is used for testing

            'working_directory': './',
            'emulator_state_file': 'emulator_state.pkl',
            'normalized_cache_file': 'normalized_cache.pkl',
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

        # Initialize the emulator with an initial state. In this state the parameters and the quantitites which are to be emulated are defined. They also come with an example value to determine the dimensionality of the quantities.
        ini_state = self.ini_state

        # if there is a input_parameters list in the kwargs, we use this list to determine the order of the input parameters
        if 'input_parameters' in kwargs:
            self.input_parameters = kwargs['input_parameters']
        else:
            self.input_parameters = list(ini_state['parameters'].keys())

        self.likelihood_collection_differentiable = False

        self.likelihood_collection = likelihood_collection
        if self.likelihood_collection is not None:
            for likelihood_name, likelihood in self.likelihood_collection.items():
                if not likelihood.initialized:
                    likelihood.initialize(**self.hyperparameters['likelihood_collection_settings'][likelihood_name])
        
            # create likelihood flags if they are differentiable and jittable
            # doesnt work for MP/Cobaya
            # self.likelihood_collection_differentiable = all([likelihood.differentiable for likelihood in self.likelihood_collection.values()])
            # self.likelihood_collection_jitable = all([likelihood.jitable for likelihood in self.likelihood_collection.values()])

        self.maximal_delta_loglike = 300 # if we are insanely far away from where the emulator was trained (like > 20 sigma), we should not use the emulator

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
        #     "loglike": {'name_of_experiment': 123, 'name_of_experiment2': 456},
        #     "total_loglike": 123456,
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

            self.debug("Quality threshold quadratic: %f", self.hyperparameters['quality_threshold_quadratic'])


            # now check for max_sigma
            if self.hyperparameters['max_sigma'] is not None:
                max_sigma = self.hyperparameters['max_sigma']
                dimensionality = self.hyperparameters['dimensionality']
                self.maximal_delta_loglike = -1.53901996e-03 * dimensionality**2 + 3.46998485e-01  * max_sigma**2 + 5.55189162e-02 * dimensionality * max_sigma + 6.39086834e-01 * dimensionality + 2.36251372e+00 * max_sigma + -5.14787690e+00


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
                        self.warning("Data covmat not found for quantity %s. Not using it for normalization.", quantity_name)

        # Here: Create the emulators for the different quantities
        self.emulators = {}

        for quantity_name, quantity in ini_state['quantities'].items():

            # check whether the quantity is in the veto list
            if self.hyperparameters['skip_emulation_quantities'] is not None:
                if quantity_name in self.hyperparameters['skip_emulation_quantities']:
                    continue

            # write that we create an emulator for this quantity
            self.info("Create emulator for %s", quantity_name)
            
            # initialize the emulator for the quantity
            self.emulators[quantity_name] = GP_predictor(quantity_name, debug=self.debug_mode)
            self.emulators[quantity_name].initialize(ini_state,data_covmat=self.data_covmats[quantity_name], **kwargs)

        # initialize the points which were already tested with the quality criterium
        self.quality_points = []

        # set groundlevel_testing_prob:
        self.groundlevel_testing_prob = None
        if self.hyperparameters['test_stochastic_rate'] != None:
            self.groundlevel_testing_prob = self.hyperparameters['test_stochastic_rate']

        # rejit_flag. If the emulator is retrained/updated, we set this flag to True. This flag is used to determine whether the emulator is to be rejited.
        self.rejit_flag_emulator = True
        self.rejit_flag_sampling = True
        self.rejit_flag_likelihood_error = True

        # following function is the jitted version of the error computation for differentiable likelihoods
        self.jitted_likelihood_error_function = None

        # this counter is used to determine the number of continously successful emulator calls
        self.continuous_successful_calls = 0

        # this counter counts the number of times that self.emulate was called
        self.emulate_counter = 0

        # this counter counts the number of times that self.check_quality_criterium was called with a successful result
        self.quality_check_successful_counter = 0
        self.quality_check_unsuccessful_counter = 0

        # max loglike encountered in the run
        self.max_loglike_encountered = -np.inf

        # if we fulfill the minimum number of data points, we train the emulator
        if len(self.data_cache.states) >= self.hyperparameters['min_data_points']:
            self.data_cache.load_cache()
            self.train()

        self.broadcast_training_flag = False # flag for rank 0 if it demands a training
        
        pass

    # @profile
    def add_state(self, state):
        # start timer
        self.start("add_state")

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
            _ = "State added to emulator: " + "; ".join([key+ ': ' +str(value) for key, value in new_state['parameters'].items()]) + " at loglike: " + str(new_state['total_loglike']) + " max. loglike: " + str(self.data_cache.max_loglike) + "\n"
            self.write_to_log(_)
            # write to log the current size of the data cache
            _ = "Current data cache size: %d\n" % len(self.data_cache.states)
            self.write_to_log(_)
        else:
            # write to log that the state was not added
            _ = "State not added to emulator: " + "; ".join([key+ ': ' +str(value) for key, value in new_state['parameters'].items()]) + " at loglike: " + str(new_state['total_loglike']) + " max. loglike: " + str(self.data_cache.max_loglike) + "\n"
            self.write_to_log(_)
            # write to log the current size of the data cache
            _ = "Current data cache size: %d\n" % len(self.data_cache.states)
            self.write_to_log(_)
        
        # stop timer
        self.increment("add_state")
        
        # if the emulator is already trained, we can add the new state to the GP without fitting the Kernel parameters
        emulator_updated = False
        if self.trained and state_added:
            self.added_data_points = self.data_cache.recently_added
            if self.added_data_points%self.hyperparameters['kernel_fitting_frequency'] == 0:
                # if you are doing MPI, tell all others that we require training
                if not is_main_process():
                    get_mpi_comm().send("require_training", dest=0, tag=24)
                else:
                    self.broadcast_training_flag = True
                self.update() # we still update here, because we will do a joint training soon :)
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
        
    # @profile
    def update(self):
        # start timer
        self.start("update")

        # Update the emulator. This means that the emulator is retrained.
        # Load the data from the cache.
        self.debug("Loading data from cache")
        # del self.jitted_emulation_function
        # del self.jitted_sampling_function
        self.data_cache.load_cache()

        self.write_to_log("Update emulator\n")

        input_data_raw, _ = self.data_cache.get_parameters()
        input_data_raw_jax = jnp.array(input_data_raw)

        # Train the emulator.
        for quantity, emulator in self.emulators.items():
            self.debug("Updating emulator for quantity %s", quantity)
            output_data_raw, _ = self.data_cache.get_quantities(quantity)

            output_data_raw_jax = jnp.array(output_data_raw)

            # load data into emulators
            emulator.load_data(input_data_raw_jax, output_data_raw_jax)

            # normalize and compress data
            emulator.data_processor.normalize_training_data()
            emulator.data_processor.compress_training_data()

            emulator.update()  

            del output_data_raw_jax

        # a = time.time()
        del input_data_raw_jax
        jax.clear_caches()
        # print("Time to clear caches: ", time.time()-a)
          
            
        self.trained = True
        self.rejit_flag_emulator = True
        self.rejit_flag_sampling = True
        self.rejit_flag_likelihood_error = True

        # stop timer
        self.increment("update")

        pass

    def create_emulator_state_file(self):
        # we store the relevant information of the emulator (at this very point) in a file.
        # This includes a nested dictionary with the following structure:
        # emulator_state = {
        #     "parameter_mean": ,
        #     "parameter_std": ,
        #     "quantites": {
            #     "quantity1": {
            #         "GPs": [{
            #             "kernel": ,       # empty if not trained 
            #             "white_noise_level": ,  # empty if not trained
            #             "is_sparse": ,     # to be implemented by pdf
            #             "inducing_points": ,     # to be implemented by pdf
            #             "status": ,},     # 0 for not trained, 1 for currently training, 2 for trained
            #           {PCA: , mean_PCA: , std_PCA: , kernel: , status: },
            #           ...
            #         ],
            #         "quantity_mean": ,
            #         "quantity_std": ,
            #         "PCAs": ,
            #         "mean_PCA": ,
            #         "std_PCA": ,},
            #     "quantity2": {...},
        #    
        # 

        emulator_state = {}
        emulator_state['quantities'] = {}
        for quantity, emulator in self.emulators.items():
            emulator_state['quantities'][quantity] = {}
            GPs = []
            for GP in emulator.GPs:
                _ = {}
                _['kernel'] = None
                _['white_noise_level'] = GP.hyperparameters['white_noise_level']
                _['status'] = 0
                
                GPs.append(_)
            emulator_state['quantities'][quantity]["GPs"] = list(GPs)

            emulator_state['quantities'][quantity]["quantity_mean"] = emulator.data_processor.output_means
            emulator_state['quantities'][quantity]["quantity_std"] = emulator.data_processor.output_stds
            emulator_state['quantities'][quantity]["PCA"] = emulator.data_processor.projection_matrix
            emulator_state['quantities'][quantity]["mean_PCA"] = emulator.data_processor.output_pca_means
            emulator_state['quantities'][quantity]["std_PCA"] = emulator.data_processor.output_pca_stds

            emulator_state["parameter_mean"] = emulator.data_processor.input_means
            emulator_state["parameter_std"] = emulator.data_processor.input_stds

        # pickle the emulator state with a lock. Overwrite if it already exists. Create it if it does not exist.
        with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] , 'wb') as f:
            pickle.dump(emulator_state, f)


        return None     


    def store_normalized_training_cache(self):
        # here we want to store the compressed and normalized data in a file. This is smaller than the raw data.
        data = {}
        for quantity, emulator in self.emulators.items():
            data['input_data_normalized'] = emulator.data_processor.input_data_normalized
            data[quantity] = emulator.data_processor.output_data_emulator

        with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters["normalized_cache_file"] + ".lock"):
            with open(self.hyperparameters['working_directory'] + self.hyperparameters["normalized_cache_file"], "wb") as fp:
                pickle.dump(data, fp)


    # @profile
    def train(self):
        self.start("train")


        # if your rank is not 0, you should tell rank 0 that you need help
        self.write_to_log("Training emulator\n")




        # here we do different things depending on the MPI rank.

        # Rank 0 has to prepare the data, do the compression and create the emulator state file.

        # Meanwhile all other ranks have to wait for the signal from rank 0 that the data is ready. 
        # Then they can load the data and train the emulator for the different quantities.

        # If all quantities are trained, all ranks wait for the signal from rank 0 that training is done.
        # Then they can load the compressed data and the emulator state file and update the emulators with the compressed data.



        if is_main_process():
            # Load the data from the cache.
            self.debug("Loading data from cache")
            self.data_cache.load_cache()
            
            # Train the emulator.
            input_data_raw, _ = self.data_cache.get_parameters()
            input_data_raw_jax = jnp.array(input_data_raw)

            for quantity, emulator in self.emulators.items():
                self.debug("Start training emulator for quantity %s", quantity)
                output_data_raw, _ = self.data_cache.get_quantities(quantity)

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

            # create the emulator state file
            self.create_emulator_state_file()

            # store the normalized training cache
            self.store_normalized_training_cache()

            # here do comminication with other ranks that the data is ready
            for i in range(1, get_mpi_size()):
                get_mpi_comm().send("Data is ready", dest=i, tag=24)
            pass

        else:
            # wait for signal from rank 0
            message = get_mpi_comm().recv(source=0, tag=24)
            pass


        # Uki, now all the data is ready and we can train the emulator for the different quantities
        untrained_emulators_flag = True
        while untrained_emulators_flag:
            untrained_emulators_flag = self.parallel_train()


        # here we wait for the signal from rank 0 that training is done
        if is_main_process():
            training_complete = False
            while not training_complete:
                # open emulators state file and check if all quantities are trained
                with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] + ".lock"):
                    with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'], 'rb') as f:
                        emulator_state = pickle.load(f)
                        training_complete = True
                        for quantity, emulator_instance in emulator_state['quantities'].items():
                            for i in range(len(emulator_instance['GPs'])):
                                if emulator_instance['GPs'][i]['status'] != 2:
                                    training_complete = False
                                    break
                
                time.sleep(0.1) # wait for 0.1 seconds before checking again

            for i in range(1, get_mpi_size()):
                get_mpi_comm().send("Training done", dest=i, tag=24)
            pass
        else:
            message = get_mpi_comm().recv(source=0, tag=24)
            pass

        # now we can load the weights from the emulator state file and update the emulators with the compressed data
        # 1. load the emulator state file
        with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] + ".lock"):
            with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'], 'rb') as f:
                emulator_state = pickle.load(f)

        # 2. load the compressed data
        with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters["normalized_cache_file"] + ".lock"):
            with open(self.hyperparameters['working_directory'] + self.hyperparameters["normalized_cache_file"], "rb") as fp:
                data = pickle.load(fp)

        # 3. update the emulators with the compressed data
        for quantity, emulator in self.emulators.items():
            emulator.data_processor.input_data_normalized = data['input_data_normalized']
            emulator.data_processor.output_data_emulator = data[quantity]
            emulator.data_processor.input_means = emulator_state["parameter_mean"]
            emulator.data_processor.input_stds = emulator_state["parameter_std"]
            emulator.data_processor.output_means = emulator_state['quantities'][quantity]["quantity_mean"]
            emulator.data_processor.output_stds = emulator_state['quantities'][quantity]["quantity_std"]
            emulator.data_processor.projection_matrix = emulator_state['quantities'][quantity]["PCA"]
            emulator.data_processor.output_pca_means = emulator_state['quantities'][quantity]["mean_PCA"]
            emulator.data_processor.output_pca_stds = emulator_state['quantities'][quantity]["std_PCA"]
            
            
            emulator.initialize_training() # fill basic structures so we can set errors accurately
            for i, GP in enumerate(emulator.GPs):
                GP.hyperparameters['white_noise_level'] = emulator_state['quantities'][quantity]['GPs'][i]['white_noise_level']
                GP.kernel = emulator_state['quantities'][quantity]['GPs'][i]['kernel']
                
            emulator.finalize_training(new_kernel=False)
        













        # import sys 
        # sys.exit(0)





        # # Load the data from the cache.
        # self.debug("Loading data from cache")
        # self.data_cache.load_cache()

        # self.write_to_log("Training emulator\n")

        
        # # Train the emulator.
        # input_data_raw = self.data_cache.get_parameters()
        # input_data_raw_jax = jnp.array(input_data_raw)

        # for quantity, emulator in self.emulators.items():
        #     self.debug("Start training emulator for quantity %s", quantity)
        #     output_data_raw = self.data_cache.get_quantities(quantity)

        #     output_data_raw_jax = jnp.array(output_data_raw)

        #     # load data into emulators
        #     emulator.load_data(input_data_raw_jax, output_data_raw_jax)

        #     del output_data_raw_jax
        #     del output_data_raw

        #     # compute normalization and compression and apply it to the data
        #     self.debug("Compute normalization and compression for quantity %s", quantity)
        #     emulator.data_processor.compute_normalization()
        #     emulator.data_processor.normalize_training_data()
        #     self.debug("Normalization done for quantity %s", quantity)

        #     emulator.data_processor.compute_compression()
        #     emulator.data_processor.compress_training_data()

        #     emulator.initialize_training() # fill basic structures so we can set errors accurately
            
        # self.set_error()

        # for quantity, emulator in self.emulators.items():
        #     self.debug("Train GP for quantity %s", quantity)

        #     emulator.finalize_training()

            

        # del input_data_raw_jax
        # del input_data_raw

        self.trained = True
        self.rejit_flag_emulator = True
        self.rejit_flag_sampling = True
        self.rejit_flag_likelihood_error = True

        jax.clear_backends()




        # if we have a plotting directory we plot the loglikelihood
        if (self.hyperparameters['plotting_directory'] is not None) and (get_mpi_rank() == 0):
            loglikes = jnp.array(self.data_cache.get_loglikes())
            parameters = jnp.array(self.data_cache.get_parameters()[0])

            for i in range(len(parameters[0])):
                plot_loglikes(loglikes[:], parameters[:,i], self.input_parameters[i], self.hyperparameters['working_directory'] + self.hyperparameters['plotting_directory']+'/loglike_'+str(i)+'.png')
        pass

        self.increment("train")

    def parallel_train(self):
        # this function looks into the emulator state file and checks which quantities are to be trained.
        # It then selects a quantity that is not trained yet, sets its flag to 1 and trains the emulator for this quantity.
        # It then stores the weights to the emulator state file and sets its flag to 2. 
        # If there are no quantities left to be trained, it returns False, otherwise True.

        my_training_quantity = None
        my_PCA_number = None

        # load the emulator state file
        with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] + ".lock"):
            with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'], 'rb') as f:
                emulator_state = pickle.load(f)

            # check which quantities are to be trained
            for quantity, emulator_instance in emulator_state['quantities'].items():
                for i in range(len(emulator_instance['GPs'])):
                    if emulator_instance['GPs'][i]['status'] == 0:
                        my_training_quantity = quantity
                        my_PCA_number = i
                        break

            self.debug("Quantity to be trained: %s on PCA component %d", my_training_quantity, my_PCA_number)

            # if there is no quantity to be trained, return False
            if my_training_quantity is None:
                return False
            
            # set flag to 1
            emulator_state['quantities'][my_training_quantity]['GPs'][my_PCA_number]['status'] = 1

            # pickle the emulator state with a lock. Overwrite it
            with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] , 'wb') as f:
                pickle.dump(emulator_state, f)

        # load the data from the cache.
        self.debug("Loading data from normalized cache")

        with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters["normalized_cache_file"] + ".lock"):
            with open(self.hyperparameters['working_directory'] + self.hyperparameters["normalized_cache_file"], "rb") as fp:
                data = pickle.load(fp)

        parameters = data['input_data_normalized']
        output_data = data[my_training_quantity][:,my_PCA_number]
        self.hyperparameters['white_noise_level'] = emulator_state['quantities'][my_training_quantity]['GPs'][my_PCA_number]['white_noise_level']

        # print current memory usage
        # import psutil
        # self.info("Current memory usage: %f GB", psutil.virtual_memory().used/1024**3)

        if not hasattr(self, 'my_test_GP'):
            self.my_test_GP = GP("Training GP " + my_training_quantity + " dim " + str(my_PCA_number),
                            **self.hyperparameters)
        else:
            self.my_test_GP.hyperparameters['white_noise_level'] = emulator_state['quantities'][my_training_quantity]['GPs'][my_PCA_number]['white_noise_level']
            self.my_test_GP.rename("Training GP " + my_training_quantity + " dim " + str(my_PCA_number))

        self.my_test_GP.load_data(parameters, output_data)
        self.my_test_GP.train()

        # store the weights to the emulator state file and set its flag to 2
        with fasteners.InterProcessLock(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] + ".lock"):
            with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'], 'rb') as f:
                emulator_state = pickle.load(f)

            emulator_state['quantities'][my_training_quantity]['GPs'][my_PCA_number]['status'] = 2
            emulator_state['quantities'][my_training_quantity]['GPs'][my_PCA_number]['kernel'] = self.my_test_GP.opt_posterior.prior.kernel
            
            # pickle the emulator state with a lock. Overwrite it
            with open(self.hyperparameters['working_directory'] + self.hyperparameters['emulator_state_file'] , 'wb') as f:
                pickle.dump(emulator_state, f)

        # a = time.time()
        jax.clear_caches()
        # print("Time to clear caches after train: ", time.time()-a)

        # remove GP from memory
        del parameters
        del output_data
        
        # self.info("Current memory usage: %f GB", psutil.virtual_memory().used/1024**3)


        return True




    def mpi_train(self):
        # in the case of mpi parallelization, we need to train the emulator in a different way.
        #
        # Rank 0:
        # 1. Copy the current state of the cache into a temporary cache that will be used for training.
        # 2. Do the PCA/data compression on the temporary cache and store a compressed version of the data.
        # 3. Create a emulator state file that holds the trained parameters of the emulator.
        # 4. Broadcast to the other ranks that both the compressed data and the emulator state file are ready, such that they can load it and train the different emulators.

        # Rank != 0:
        # 1. Wait until the broadcast from rank 0 is received.
        # 2. Load the emulator state file and check which quantities are to be trained. There is a flag that can either state 0: 'not trained yet', 1: 'currently training', 2: 'already trained'.
        # 3. Select a quantity that is not trained yet, set its flag to 1 and train the emulator for this quantity.
        # 4. Train the emulator for the quantity.
        # 5. Store your weights to the emulator state file and set its flag to 2. Repeat until all quantities are trained.
        # 6. If all quantities are trained, wait until you receive message from rank 0 that training is done.
        # 7. When message is received, load the compressed data and the emulator state file and update the emulators with the compressed data.






        return None



    def set_data_covmats(self, data_covmats):
        # set the data covmats for the different quantities
        for quantity_name, data_covmat in data_covmats.items():
            self.data_covmats[quantity_name] = data_covmat
            self.emulators[quantity_name].data_processor.data_covmat = data_covmat

    # @profile
    def emulate(self, parameters):

        # check (if we are using MPI) whether we need to train the emulator
        if get_mpi_rank() != 0:
            # check if there is a message from rank 0 that we need to train the emulator
            if get_mpi_comm().Iprobe(source=0, tag=24):
                message = get_mpi_comm().recv(source=0, tag=24)
                if message == "require_training":
                    self.train()
        else:
            # check mailbox 
            train_flag = False
            for i in range(1, get_mpi_size()):
                if get_mpi_comm().Iprobe(source=i, tag=24):
                    message = get_mpi_comm().recv(source=i, tag=24)
                    if message == "require_training":
                        train_flag = True
                    
            if self.broadcast_training_flag:
                train_flag = True
                self.broadcast_training_flag = False
            
            if train_flag:
                for i in range(1, get_mpi_size()):
                    get_mpi_comm().send("require_training", dest=i, tag=24)
                self.train()

        # start timer 
        self.start("emulate") 

        # Prepare output state
        output_state = self.ini_state.copy() # TODO Talk with christian about thish here

        # write to log
        self.write_parameter_dict_to_log(parameters)

        # if we jit the emulator, we use the jit version of the emulator
        if self.hyperparameters['jit'] and (self.continuous_successful_calls > self.hyperparameters['jit_threshold']):
            if self.rejit_flag_emulator: # if the emulator was updated/retrained, we rejit the emulator
                # if there is a old jitted function, we delete it
                if hasattr(self, 'jitted_emulation_function'):
                    jax.clear_backends()
                    jax.clear_caches()
                    self.jitted_emulation_function.clear_cache()
                    self.jitted_emulation_function = None
                    # del self.jitted_emulation_function
                    # clear memory
                    jax.clear_caches()
                    jax.clear_backends()
                    gc.collect()
                self.jitted_emulation_function = jax.jit(self.emulate_jit)
                output_state_emulator = self.jitted_emulation_function(parameters)
                self.rejit_flag_emulator = False
                jax.clear_caches()
            else:
                output_state_emulator = self.jitted_emulation_function(parameters)
        else:
            output_state_emulator = self.emulate_nojit(parameters)

        # overwrite the quantities which were emulated
        for quantity, emulator_output in output_state_emulator['quantities'].items():
            output_state['quantities'][quantity] = emulator_output

        # overwrite the parameters
        output_state['parameters'] = output_state_emulator['parameters']

        self.emulate_counter += 1

        # end timer
        self.increment("emulate")

        return output_state

    def emulate_jit(self, parameters):
        # Prepare output state
        output_state = {'parameters': {}, 'quantities':{}, 'loglike': {}, 'total_loglike': None} 
        output_state['parameters'] = parameters

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])
        for quantity, emulator in self.emulators.items():
            emulator_output = emulator.predict(input_data)
            output_state['quantities'][quantity] = emulator_output
        
        return output_state

    def emulate_nojit(self, parameters):
        # Prepare output state
        output_state = {'parameters': {}, 'quantities':{}, 'loglike': {}, 'total_loglike': None} 
        output_state['parameters'] = parameters

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])

        for quantity, emulator in self.emulators.items():
            emulator_output = emulator.predict(input_data)
            output_state['quantities'][quantity] = emulator_output
        
        return output_state
    
    def compute_error_from_differentiable_likelihood(self, parameters, include_white_noise = False):
        # Here we need to compute the error of the emulator via the differentiable likelihood.
        # Therefore, we get the errors on the GPs two layers below in the GPpredictor class.
        # Addtionally we need the pipeline from the GP prediction to the likelihood. 
        # Thus, computing the decompression of each emulated quantity and feed it through the likelihood collection.
        # Then we can use jit.grad to compute the gradient of the loglikelihood with respect to the GP prediction.
        # By multiplying this gradient with the error of the GP we get the error of the likelihood without some shooting method.

        # define the likelihood from GP function
        a = time.time()
        self.include_white_noise = include_white_noise
        
        if self.rejit_flag_likelihood_error:
            if self.jitted_likelihood_error_function is not None:
                self.jitted_likelihood_error_function.clear_cache()
                del self.jitted_likelihood_error_function
            # clear memory
            jax.clear_backends()
            self.jitted_likelihood_error_function = None
        

        if self.rejit_flag_likelihood_error:
            self.jitted_likelihood_error_function = jax.jit(self.jittable_likelihood_error_function)
            error = self.jitted_likelihood_error_function(parameters)
            self.rejit_flag_likelihood_error = False

        
        error = self.jitted_likelihood_error_function(parameters)
        b = time.time()
        print("Time for error computation: ", b-a)
        # return the loglikelihood
        return error
    

    # def the jittable version of the likelihood function
    def jittable_likelihood_error_function(self, parameters):
        output_state = {'parameters': {}, 'quantities':{}, 'loglike': {}, 'total_loglike': None} 
        output_state['parameters'] = parameters

        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])

        emulator_GP_uncertainy = {}
        emulator_GP_value = {}
        for quantity, emulator in self.emulators.items():
            vals, std = emulator.predict_GP_value_and_std(input_data, include_white_noise = self.include_white_noise)
            emulator_GP_value[quantity] = vals
            emulator_GP_uncertainy[quantity] = std

        def likelihood_from_GP(emulator_GPs):
            # decompress the data
            for quantity, emulator in self.emulators.items():
                output_state['quantities'][quantity] = emulator.predict_fromGP(emulator_GPs[quantity])

            # feed the data through the likelihood collection
            for likelihood_name, likelihood in self.likelihood_collection.items():
                output_state['loglike'][likelihood_name] = likelihood.loglike(output_state)

            output_state['total_loglike'] = sum(output_state['loglike'].values())

            return output_state['total_loglike'][0]

        # compute the gradient of the loglikelihood with respect to the GP prediction
        grad_loglike = jax.grad(likelihood_from_GP)(emulator_GP_value)

        # compute the error of the likelihood
        error = 0.
        for quantity, emulator in self.emulators.items():
            error += jnp.sum(grad_loglike[quantity]**2 * emulator_GP_uncertainy[quantity]**2)

        return jnp.sqrt(error)




    # function to get N samples from the same input parameters
    # @profile
    def emulate_samples(self, parameters, RNGkey,noise = 0):
        # add Sphinx documentation
        """
        Emulate N samples of the quantities for the given parameters.

        Parameters
        ----------
        parameters : dict
            The parameters for which the quantities are to be emulated.
        RNGkey : jax.random.PRNGKey
            The random number generator key.

        Returns
        -------
        list
            A list of N output states. Each output state is a dictionary with the following structure:
            {
                "parameters": {
                    "parameter1": [123],
                    "parameter2": [456],
                    ...
                },
                "quantities": {
                    "quantity1": [element1, element2, ...],
                    "quantity2": [element1, element2, ...],
                    ...
                }
                "loglike": {'name_of_experiment': 123, 'name_of_experiment2': 456},
                "total_loglike": 123456,
            }
        jax.random.PRNGKey
            The updated random number generator key.
        """
        # start timer
        self.start("emulate_samples")

        # Prepare list of N output states
        state = deepcopy(self.ini_state) # TODO Talk with christian about thish here
        state['parameters'] = parameters
        output_states = [deepcopy(state) for i in range(self.hyperparameters['N_quality_samples'])]

        # use jit or no jit version of the function
        if self.hyperparameters['jit'] and (self.continuous_successful_calls > self.hyperparameters['jit_threshold']):
            if self.rejit_flag_sampling: # if the emulator was updated/retrained, we rejit the emulator

                if hasattr(self, 'jitted_sampling_function'):
                    # clear memory
                    jax.clear_backends()
                    jax.clear_caches()
                    self.jitted_sampling_function.clear_cache()
                    self.jitted_sampling_function = None
                    del self.jitted_sampling_function
                    # clear memory
                    jax.clear_backends()
                    jax.clear_caches()
                    gc.collect()

                self.jitted_sampling_function = jax.jit(self.emulate_samples_jit)
                output_states_emulator, RNGkey = self.jitted_sampling_function(parameters, RNGkey,noise)
                self.rejit_flag_sampling = False
            else:
                output_states_emulator, RNGkey = self.jitted_sampling_function(parameters, RNGkey, noise)
        else:
            output_states_emulator, RNGkey = self.emulate_samples_nojit(parameters, RNGkey ,noise)

        # overwrite the quantities which were emulated
        for i in range(self.hyperparameters['N_quality_samples']):
            for quantity, emulator_output in output_states_emulator[i]['quantities'].items():
                output_states[i]['quantities'][quantity] = emulator_output

        # increment timer
        self.increment("emulate_samples")

        # start the likelihood_testing counter
        self.start("likelihood_testing")


        return output_states, RNGkey

    # @partial(jax.jit, static_argnums=0)
    def emulate_samples_jit(self, parameters, RNGkey, noise = 0):
        # Prepare list of N output states
        state = {'parameters': {}, 'quantities':{}, 'loglike': {}, 'total_loglike': None} 
        state['parameters'] = parameters
        output_states = [deepcopy(state) for i in range(self.hyperparameters['N_quality_samples'])]

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])

        for quantity, emulator in self.emulators.items():
            emulator_output, RNGkey = emulator.sample_prediction(input_data, N=self.hyperparameters['N_quality_samples'],noise=noise, RNGkey=RNGkey)
            
            for i in range(self.hyperparameters['N_quality_samples']):
                output_states[i]['quantities'][quantity] = emulator_output[i,:]

        return output_states, RNGkey

    def emulate_samples_nojit(self, parameters, RNGkey, noise = 0):
        # Prepare list of N output states
        state = {'parameters': {}, 'quantities':{}, 'loglike': {}, 'total_loglike': None} 
        state['parameters'] = parameters
        output_states = [deepcopy(state) for i in range(self.hyperparameters['N_quality_samples'])]

        # Emulate the quantities for the given parameters.
        input_data = jnp.array([[parameters[key][0] for key in self.input_parameters]])

        for quantity, emulator in self.emulators.items():
            emulator_output, RNGkey = emulator.sample_prediction(input_data, N=self.hyperparameters['N_quality_samples'],noise = noise, RNGkey=RNGkey)
            
            for i in range(self.hyperparameters['N_quality_samples']):
                output_states[i]['quantities'][quantity] = emulator_output[i,:]

        return output_states, RNGkey



    def set_error(self):

        if self.hyperparameters['sparse_GP_points'] > 0.:
            # require errors for sparse GP's
            if self.hyperparameters['white_noise_ratio'] == 0.:
                print("Sparse GP require a white noise error. I have set the noise ratio to one!") 
                self.hyperparameters['white_noise_ratio'] = 1.


        if self.hyperparameters['white_noise_ratio'] == 0.:
            for quantity_name in self.emulators.keys():
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
            self.debug('set the noise levels to ')
            for quantity_name, quantity in self.ini_state['quantities'].items():

                # check whether the quantity is in the veto list
                if self.hyperparameters['skip_emulation_quantities'] is not None:
                    if quantity_name in self.hyperparameters['skip_emulation_quantities']:
                        continue

                # Ali memory bug <3

                var = self.emulators[quantity_name].data_processor.explained_variance #* len(self.emulators[quantity_name].data_processor.output_data_emulator)
                # this is analytical for the variance of the components. since we compare to the total we might drop the len of data
                relative_importance = self.emulators[quantity_name].data_processor.relative_importance
                total_var = jnp.sum(self.emulators[quantity_name].data_processor.explained_variance)
                # variance_tolerance = 1. - self.emulators[quantity_name].data_processor.hyperparameters['explained_variance_cutoff'] # TODO: PDF: Check this please :)
                variance_tolerance = 1 - self.emulators[quantity_name].data_processor.cumulative_explained_variance[self.emulators[quantity_name].data_processor.output_data_emulator_dim]

                for i in range(self.emulators[quantity_name].num_GPs):
                    relative_variance = var[i]/total_var
                    #error = variance_tolerance / relative_variance * self.hyperparameters['noise_percentage']
                    # error = ((variance_tolerance / relative_variance) ** 2. ) / len(var)
                    error = ((variance_tolerance / relative_variance)  ) / len(var) # TODO: SG: Is that correct? Seems to me ...
                    
                    # experimental new option:
                    error = 1. / relative_importance[i]

                    # set it to an minimum value of 1e-14
                    error = max(error* self.hyperparameters['white_noise_ratio'], 1e-14) # this ensures some white kernel. Otherwise training might fail for deterministic data, like training H0 out of h etc ...
                    
                    self.emulators[quantity_name].GPs[i].hyperparameters['white_noise_level'] = error 
                    self.debug("Error tolerance for GP %d of quantity %s: %e" % (i, quantity_name, error))

    
    def check_quality_criterium(self, loglikes, parameters, reference_loglike = None, write_log = True):
        
        # stop the likelihood counter
        self.increment("likelihood_testing")
        # correct for the emulator runtime
        self.subtimer["likelihood_testing"].time_sum -= self.subtimer["emulate"].last_round

        # check whether the emulator is good enough to be used
        # if the emulator is not yet trained, we return False
        if not self.trained:
            self.continuous_successful_calls = 0
            self.quality_check_unsuccessful_counter += 1
            return False

        # check for nans in the loglikes
        if jnp.any(jnp.isnan(loglikes)):
            self.quality_check_unsuccessful_counter += 1
            # send warning
            self.warning("Loglike is NaN!!! Please ensure that your pipeline is working correctly.")
            self.log("Loglike is NaN!!! Please ensure that your pipeline is working correctly.")
            return False
        
        
        if reference_loglike is None:
            mean_loglike = jnp.mean(loglikes)
            if self.likelihood_collection_differentiable:
                raise ValueError("Reference loglike is not provided for Quality check!")
        else:
            mean_loglike = reference_loglike


        max_loglike = max(self.data_cache.max_loglike, self.max_loglike_encountered)

        # check if the loglike is too far away from the training data
        if jnp.any(jnp.abs(loglikes - max_loglike) > self.maximal_delta_loglike):
            self.quality_check_unsuccessful_counter += 1
            _ = "Loglikes are too far away from best-fit point. Not using OLE; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, Reference loglike: %f, delta loglikes: " % (max_loglike,mean_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
            self.write_to_log(_)
            return False

        # as long as mean_loglike is bugged we pass the correct one as the first point
        # BUG: fix refernce loglike (apears to be fixed)
        # mean_loglike = loglikes[0]


        # if the emulator is trained, we check the quality criterium
        # we check whether the loglikes are within the quality criterium
        if not self.likelihood_collection_differentiable:
            
            # old estimator
            # here we need to cut outlier of the chisquare distribution.
            #N_cut = max(1, int(self.hyperparameters['tail_cut_fraction'] * self.hyperparameters['N_quality_samples']))

            #std_loglike_precut = jnp.std(loglikes) # we use the standard deviation of the mean as the error
            # Thus we gonna cut the N_cut smallest values
            #loglikes = jnp.sort(loglikes)[N_cut:]

            
            #std_loglike = jnp.std(loglikes) # we use the standard deviation of the mean as the error
            # this scaling is wierd as more samples lead to a smaller error, but even if we know the mean very well, how is this important. 
            # We are then very certain tha the meam is the mean so ... 
            #if (std_loglike_precut - std_loglike)/std_loglike_precut * 100 > 85:
            #    print("error reduced byt cut ",(std_loglike_precut - std_loglike)/std_loglike_precut * 100)

            # new estimator here
            # this is based on using that we rolled all points at the 1 sigma level and thus we can use them to estimate the posterior
            # one sigma assuming that we stay in the tangent-space of the likeleehood
         
            #loglikesNew = loglikes[1:]
            variances_loglikes = ( loglikes - mean_loglike )**2
            std_loglike = jnp.sqrt(jnp.median(variances_loglikes))
            

        else:
            std_loglike = loglikes[0]


        delta_loglike = jnp.abs(mean_loglike - max_loglike)

        # write testing to log

        # if the mean loglike is above the maximum found loglike we only check the constant term
        if mean_loglike > max_loglike:
            if std_loglike > self.hyperparameters['quality_threshold_constant']:
                self.debug("Emulator quality criterium NOT fulfilled")
                _ = "Quality criterium NOT fulfilled; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, Reference loglike: %f, delta loglikes: " % (max_loglike,mean_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
                if write_log: 
                    self.write_to_log(_)
                self.quality_check_unsuccessful_counter += 1
                return False
        else:
            # calculate the absolute difference between the mean loglike and the maximum loglike
            delta_loglike = jnp.abs(mean_loglike - max_loglike)
            
            # the full criterium 
            if std_loglike > self.hyperparameters['quality_threshold_constant'] + self.hyperparameters['quality_threshold_linear']*delta_loglike + self.hyperparameters['quality_threshold_quadratic']*delta_loglike**2:
                self.debug("Emulator quality criterium NOT fulfilled")
                _ = "Quality criterium NOT fulfilled; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, Reference loglike: %f, delta loglikes: " % (max_loglike,mean_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
                if write_log: 
                    self.write_to_log(_)
                self.quality_check_unsuccessful_counter += 1
                return False

        self.debug("Emulator quality criterium fulfilled")
        _ = "Quality criterium fulfilled; "+"; ".join([key+ ': ' +str(value) for key, value in parameters.items()]) + " Max loglike: %f, Reference loglike: %f, delta loglikes: " % (max_loglike,mean_loglike) + " ".join([str(loglike) for loglike in loglikes]) + "\n"
        if write_log: 
            self.write_to_log(_)
        self.continuous_successful_calls += 1
        
        # if any of the loglikes is above the maximum loglike, we need to update the maximum loglike
        if jnp.any(mean_loglike > max_loglike):
            self.max_loglike_encountered = mean_loglike

        self.quality_check_successful_counter += 1
        return True
    
    def require_quality_check(self, parameters):
        # check whether the emulator is expected to perform well for the given parameters.
        # if we return False, the emulator is expected to perform well and we do not need to check the quality criterium
        # if we return True, the emulator is expected to perform poorly and we need to check the quality criterium

        # every now and then its nice to check if the cache did change and if new data points are available. In particular in the very late stage of the run
        if (self.continuous_successful_calls+1) % self.hyperparameters['check_cache_for_new_points'] == 0:
            if self.data_cache.check_for_new_points():
                self.debug("New points in cache. Quality check required.")
                self.update()
                

        self.print_status()

        # if we do not require a quality check, we return False
        if self.hyperparameters['testing_strategy'] == 'test_none':
            self.debug("Quality check not required. Test emulator is False")
            self.write_to_log("Quality check not required. Test emulator is False \n")
            self.continuous_successful_calls += 1
            return False

        # if we test all calls until the first 'test_early_points' points were consecutively successful. Stop testing afterwards.
        if (self.hyperparameters['testing_strategy'] == 'test_early') and (self.continuous_successful_calls > self.hyperparameters['test_early_points']):
            self.debug("Quality check not required. Test emulator is False")
            self.write_to_log("Quality check not required. Test emulator is False \n")
            self.continuous_successful_calls += 1
            return False

        # implement testing strategy
        if self.hyperparameters['testing_strategy'] == 'test_stochastic':
            if self.hyperparameters['dimensionality'] is not None:
                testing_prob = np.exp(-self.continuous_successful_calls/self.hyperparameters['test_stochastic_scale']/self.hyperparameters['dimensionality'])
            else:
                testing_prob = np.exp(-self.continuous_successful_calls/self.hyperparameters['test_stochastic_scale']/10.0)

            # set to groundlevel testing probability
            if self.hyperparameters['test_stochastic_rate'] == None:
                if self.groundlevel_testing_prob == None:
                    self.groundlevel_testing_prob = 1.0
                if ('likelihood_testing' in self.subtimer.keys()) and (self.emulate_counter<20):
                    self.groundlevel_testing_prob = 1 / self.hyperparameters['N_quality_samples'] * self.hyperparameters['test_stochastic_testing_time_fraction']
                if self.emulate_counter==20: 
                    self.debug("Groundlevel testing probability set to %f", self.groundlevel_testing_prob)
                    self.write_to_log("Groundlevel testing probability set to " + str(self.groundlevel_testing_prob) + "\n")
            testing_prob = max(testing_prob, self.groundlevel_testing_prob)
            if np.random.rand() > testing_prob:
                self.debug("Quality check not required. Test emulator is False")
                self.write_to_log("Quality check not required. Test emulator is False \n")
                self.continuous_successful_calls += 1
                return False


        # check whether the normalized input data is within a certain radius of the checked points
        if len(self.quality_points) == 0:
            self.debug("No quality points yet. Quality check required.")
            return True
        else:
            # The idea is that we collect all points which were checked by the quality criterium and then check whether the new point is within the convex hull of the checked points.
            input_data = jnp.array([[value[0] for key, value in parameters.items() if key in self.input_parameters]])

            # normalize the input data
            quantity = list(self.emulators.keys())[0]
            normalized_input_data = self.emulators[quantity].data_processor.normalize_input_data(input_data)
            
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
    
    def print_status(self):
        N = self.emulate_counter
        if 'theory_code' in self.subtimer.keys():
            N += self.subtimer['theory_code'].n
        if N%self.hyperparameters['status_print_frequency'] == 0:
            # print the status of the emulator
            self.info("Emulator status: [" + datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S") + "]")
            self.info("Number of data points in cache: %d", len(self.data_cache.states))
            self.info("Number of emulation calls: %d", self.emulate_counter)
            self.info("Number of quality check successful calls: %d", self.quality_check_successful_counter)
            self.info("Number of quality check unsuccessful calls: %d", self.quality_check_unsuccessful_counter)
            self.info("Number of not tested calls: %d", self.emulate_counter - self.quality_check_unsuccessful_counter - self.quality_check_successful_counter)
            self.log_all_times(self.logger)
            # also write the status to the log file
            self.write_to_log("Emulator status: [" + datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S") + "]\n")
            self.write_to_log("Number of data points in cache: %d\n" % len(self.data_cache.states))
            self.write_to_log("Number of emulation calls: %d\n" % self.emulate_counter)
            self.write_to_log("Number of quality check successful calls: %d\n" % self.quality_check_successful_counter)
            self.write_to_log("Number of quality check unsuccessful calls: %d\n" % self.quality_check_unsuccessful_counter)
            self.write_to_log("Number of not tested calls: %d\n" % (self.emulate_counter - self.quality_check_unsuccessful_counter - self.quality_check_successful_counter))
            self.write_to_log("Time spent in different parts of the code: \n")
            for name in list(self.subtimer.keys()):
                _ = f"Timing for {name}: total: {self.get_summed_time(name):.4f}s calls: {self.n(name)} avg: {self.get_time_avg(name):.4f}s \n"
                self.write_to_log(_)

    def log_all_times(self, logger):
        for name in self.subtimer:
            self.log_time(name, logger)

        
        
        
        pass

    def write_to_log(self, message):
        # write the message to the logfile
        if self.hyperparameters['logfile'] is None:
            return
        
        file_name = self.hyperparameters['working_directory'] + self.hyperparameters['logfile']+'_'+str(get_mpi_rank())+'.log'
        # check if file/directory exists, otherwise create it
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
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




