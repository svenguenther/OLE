from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
import sys
import time
import pickle
import fasteners
import copy

import jax.numpy as jnp

# /write a data cache which stores the data in a hdf5 file. The data cache should be able to handle the following operations:
# - write a state to the cache
# - read a state from the cache
# - store the cache in a hdf5 file
# - load the cache from a hdf5 file

# Path: OLE/data_cache.py

# The data cache is a dictionary of the individual states. The keys are the names of the states.


# A state dictionary is a nested dictionary with the following structure:
# state = {
#     "parameters": {
#         "parameter1": 123,
#         "parameter2": 456,
#         ...
#     },
#     "quantities": {
#         "quantity1": [element1, element2, ...],
#         "quantity2": [element1, element2, ...],
#         ...
#     },
#     "loglike": 123, (or None if not available)
# }

class DataCache(BaseClass):

    states: list

    hyperparameters: dict

    # the init comes with a dictionary of the hyperparameters
    def __init__(self, **kwargs):
        super().__init__("DataCache", **kwargs)

        defaulthyperparameters = {
            # the cache size is the number of states which are stored in the cache
            'cache_size': 1000,

            # the cache file is the hdf5 file in which the cache is stored
            'cache_file': 'cache.pkl',

            # load the cache from the cache file
            'load_cache': True,

            # delta loglike is the the maximum allowed difference of the loglike between two states which are stored in the cache.
            # It should prevent the cache from storing states which are outliers.
            'delta_loglike': 300,

            # the cache is stored in the hdf5 file after each state is added
            'store_cache': True,

            # learn about the actual emulation task to estimate the 'delta_loglike'.
            'N_sigma': 6,
            'dimensionality': None, # if we give the dimensionality, the code estimates 'delta_loglike' to ensure that all data cache points are within N_sigma of the likelihood.


        }

        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value


        # if 'dimensionality' is given we need to estimate the quality_threshold_quadratic
        if self.hyperparameters['dimensionality'] is not None:
            from scipy.stats import chi2
            # we need to estimate the quality_threshold_quadratic
            # we use the N_sigma to estimate the quality_threshold_quadratic
            # we estimate the quality_threshold_quadratic such that it becomes dominant over the linear term at the N_sigma point

            # up to which p value do we want to be accurate?
            p_val = chi2.cdf(self.hyperparameters['N_sigma']**2, 1)

            if p_val == 1.0:
                self.warning("N_sigma is too large. The p value is 1.0 due to double precision. The estimated delta_loglike is not accurate. Thus, N_sigma = 8")
                p_val = chi2.cdf(8**2, 1)

            # the corresponding loglike
            self.hyperparameters['delta_loglike'] = chi2.ppf(p_val, self.hyperparameters['dimensionality'])/2

            # print the estimated delta_loglike
            self.debug("Estimated delta_loglike: ", self.hyperparameters['delta_loglike'])

        self.states = []

        self.max_loglike = None

        # if the cache file exists, load the cache from the file
        if self.hyperparameters['load_cache']:
            if os.path.exists(self.hyperparameters['cache_file']):
                self.load_cache()
                self.max_loglike = max(self.get_loglikes())
        else:
            # delete old cache file
            try:
                os.remove(self.hyperparameters['cache_file'])
            except OSError:
                pass

        

        pass

    def initialize(self, ini_state):
        self.states.append(ini_state)

    def add_state(self, new_state):        
        # update cache
        if self.hyperparameters['load_cache']:
            if os.path.exists(self.hyperparameters['cache_file']):
                self.load_cache()


        # check if delta loglike is exceeded
        # returns True if the state is added to the cache, False otherwise
        new_loglike = new_state['loglike']
        self.max_loglike = max(self.get_loglikes())

        self.info("new_loglike: %f", new_loglike)
        self.info("max_loglike: %f", self.max_loglike)

        # check if the new loglike is larger than the maximum loglike
        if (self.max_loglike - new_loglike) > self.hyperparameters['delta_loglike']:
            self.debug("delta_loglike exceeded")
            return False
        
        # check if the new data point is already in the cache
        for state in self.states:
            if state['parameters'] == new_state['parameters']:
                self.debug("state already in cache")
                return False
        
        # check if the cache is full
        if len(self.states) >= self.hyperparameters['cache_size']:
            # remove the state with the smallest loglike if the new loglike is larger than the smallest loglike
            min_loglike = min(self.get_loglikes())
            if new_loglike > min_loglike:
                for i,state in enumerate(self.states):
                    if state['loglike'] == min_loglike:
                        self.states.pop(i) #
                        break
    
        # add a state to the cache
        self.states.append(new_state)

        self.info("Added state to cache: %s", new_state['parameters'])
        self.info("Cache size: %d/%d", len(self.states), self.hyperparameters['cache_size'])

        print("new_state: ", new_state)

        # if there exists a chache file, store the cache in the file
        if os.path.exists(self.hyperparameters['cache_file']):
            self.synchronize_to_cache(new_state)
        else:
            self.store_cache()

        return True

    def get_parameters(self):
        # returns all parameters
        return [[values[0] for parameter,values in state['parameters'].items()] for state in self.states]

    def get_quantities(self, quantity):
        # returns all quantities of the given name
        return [state['quantities'][quantity] for state in self.states]
    
    def get_loglikes(self):
        # returns all loglikes
        return [state['loglike'] for state in self.states]

    def store_cache(self):
        # store the cache in the hdf5 file

        # if MPI, we need to make sure that we have a loop to wait until the file is not locked anymore
        with fasteners.InterProcessLock(self.hyperparameters['cache_file'] + '.lock'):
            with open(self.hyperparameters['cache_file'], 'wb') as fp:
                pickle.dump(self.states, fp)

            with open(self.hyperparameters['cache_file'], 'rb') as fp:
                old = pickle.load(fp)

        self.debug("Stored cache in file: %s", self.hyperparameters['cache_file'])

    def synchronize_to_cache(self, new_state):
        # This function reads the old states from the cache file, adds the new state and stores the new states in the cache file.
        # This is useful if we want to store the cache from different processes.

        # if MPI, we need to make sure that we have a loop to wait until the file is not locked anymore
        with fasteners.InterProcessLock(self.hyperparameters['cache_file'] + '.lock'):
            with open(self.hyperparameters['cache_file'], 'rb') as fp:
                old = pickle.load(fp)

            old.append(new_state)

            # here we need to remove states if the cache is full
            if len(old) > self.hyperparameters['cache_size']:
                min_loglike = min([state['loglike'] for state in old])
                for i,state in enumerate(old):
                    if state['loglike'] == min_loglike:
                        old.pop(i)
                        break
            
            # remove states whose loglike is to far away from the maximum loglike
            max_loglike = max([state['loglike'] for state in old])

            valid_indices = []
            for i, state in enumerate(old):
                if not (abs(state['loglike'][0] - max_loglike) > self.hyperparameters['delta_loglike']):
                    # remove index from mask
                    valid_indices.append(i)

            # keep valid states
            old = [old[i] for i in valid_indices]

            with open(self.hyperparameters['cache_file'], 'wb') as fp:
                pickle.dump(old, fp)

            self.states = old

            self.max_loglike = max(self.get_loglikes())

    def load_cache(self):
        # load the cache from the hdf5 file
        self.states = []


        with fasteners.InterProcessLock(self.hyperparameters['cache_file'] + '.lock'):
            with open(self.hyperparameters['cache_file'], 'rb') as fp:
                self.states = pickle.load(fp)

        self.debug("Loaded cache from file: %s", self.hyperparameters['cache_file'])

