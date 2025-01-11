"""
Data Cache
"""

from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
import sys
import time
import pickle
import fasteners
import copy

import jax.numpy as jnp
import numpy as np

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
#     "loglike": {'likelihood1': 0.0, 'likelihood2': 0.0, ...}
#     "total_loglike": 0.0
# }


class DataCache(BaseClass):

    states: list

    hyperparameters: dict

    # the init comes with a dictionary of the hyperparameters
    def __init__(self, **kwargs):
        super().__init__("DataCache", **kwargs)

        defaulthyperparameters = {
            # the cache size is the number of states which are stored in the cache
            "cache_size": 1000,
            # the cache file is the hdf5 file in which the cache is stored
            "cache_file": "cache.pkl",
            "compressed_cache_file": "compressed_cache.pkl",
            # working directory
            "working_directory": './',
            # flag if we want to share the cache between different processes
            "share_cache": True,
            # load the cache from the cache file
            "load_cache": False,
            # delta loglike is the the maximum allowed difference of the loglike between two states which are stored in the cache.
            # It should prevent the cache from storing states which are outliers.
            "delta_loglike": 50,
            # learn about the actual emulation task to estimate the 'delta_loglike'.
            "N_sigma": 4,
            "dimensionality": None,  # if we give the dimensionality, the code estimates 'delta_loglike' to ensure that all data cache points are within N_sigma of the likelihood.
        }

        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        # if 'dimensionality' is given we need to estimate the quality_threshold_quadratic
        if self.hyperparameters["dimensionality"] is not None:
            from scipy.stats import chi2

            # we need to estimate the quality_threshold_quadratic
            # we use the N_sigma to estimate the quality_threshold_quadratic
            # we estimate the quality_threshold_quadratic such that it becomes dominant over the linear term at the N_sigma point

            # up to which p value do we want to be accurate?
            p_val = chi2.cdf(self.hyperparameters["N_sigma"] ** 2, 1)

            if p_val == 1.0:
                self.warning(
                    "N_sigma is too large. The p value is 1.0 due to double precision. The estimated delta_loglike is not accurate. Thus, N_sigma = 8"
                )
                p_val = chi2.cdf(8**2, 1)

            # the corresponding loglike
            self.hyperparameters["delta_loglike"] = (
                chi2.ppf(p_val, self.hyperparameters["dimensionality"]) / 2
            )

            # print the estimated delta_loglike
            self.debug(
                "Estimated delta_loglike: %f", self.hyperparameters["delta_loglike"]
            )

        self.states = []
        self.states_hashes = []

        self.max_loglike = None

        # if we do not want to share the cache we need to modify the cache file by the rank
        if not self.hyperparameters["share_cache"]:
            rank = get_mpi_rank()
            self.hyperparameters["cache_file"] = self.hyperparameters["cache_file"][:-4] + "_%d.pkl" % rank
            self.hyperparameters["compressed_cache_file"] = self.hyperparameters["compressed_cache_file"][:-4] + "_%d.pkl" % rank

        # if the cache file exists, load the cache from the file
        if self.hyperparameters["load_cache"]:
            if os.path.exists(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"]):
                self.load_cache()
                self.max_loglike = self.get_max_loglike()
        else:
            # delete old cache file
            try:
                os.remove(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"])
            except OSError:
                pass

        pass

    def initialize(self, ini_state):
        self.states.append(ini_state)

    def check_for_new_points(self):
        # this function checks if the cache changed since the last call
        old_hash = self.cache_hash

        # load the cache from the file
        self.load_cache()

        # check if the cache changed
        if old_hash != self.cache_hash:
            return True
        else:
            return False


    def add_state(self, new_state):
        # update cache
        if self.hyperparameters["load_cache"]:
            if os.path.exists(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"]):
                self.load_cache()

        # check if delta loglike is exceeded
        # returns True if the state is added to the cache, False otherwise
        new_loglike = new_state["total_loglike"]

        self.max_loglike = self.get_max_loglike()


        # check if new loglike is nan
        if jnp.isnan(new_loglike):
            self.error("LOGLIKE IS NAN!")
            return False

        self.debug(
            "Loglikelihood of incoming state: %f, Current bestfit Loglikelihood %s"
            % (new_loglike.sum(), self.max_loglike)
        )

        # check if the new loglike is larger than the maximum loglike
        if (self.max_loglike - new_loglike) > self.hyperparameters["delta_loglike"]:
            self.debug("delta_loglike exceeded")
            return False

        # check if the new data point is already in the cache
        for state in self.states:
            if str(state["parameters"]) == str(new_state["parameters"]):
                # update the loglike if the new loglike is larger
                if new_loglike > state["total_loglike"]:
                    state["total_loglike"] = new_loglike
                    self.debug("updated loglike in cache")
                self.debug("state already in cache")
                return False

        # check if the cache is full
        if len(self.states) >= self.hyperparameters["cache_size"]:
            # remove the state with the smallest loglike if the new loglike is larger than the smallest loglike
            min_loglike = min(self.get_loglikes())
            if new_loglike > min_loglike:
                for i, state in enumerate(self.states):
                    if state["total_loglike"] == min_loglike:
                        self.states.pop(i)  #
                        break


        self.debug("Added state to cache: %s", new_state["parameters"])

        # if there exists a chache file, store the cache in the file
        if os.path.exists(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"]):
            self.synchronize_to_cache(new_state)
        else:
            # add a state to the cache
            self.states.append(new_state)
            
            self.store_cache()

        self.info(
            "Cache size: %d/%d", len(self.states), self.hyperparameters["cache_size"]
        )

        self.update_hashes()

        return True

    def get_parameters(self, veto_hashes=[]):
        # get indices of veto_hashes which are not in self.states_hashes
        removal_indices = []
        for i, _hash in enumerate(veto_hashes):
            if _hash not in self.states_hashes:
                removal_indices.append(i)

        # get the new states which are not in the veto_hashes
        new_parameters = []
        for i, state in enumerate(self.states):
            if self.states_hashes[i] not in veto_hashes:
                _ = [values[0] for parameter, values in state["parameters"].items()]
                new_parameters.append(_)

        return new_parameters, removal_indices

    def get_quantities(self, quantity, veto_hashes=[]):
        # get indices of veto_hashes which are not in self.states_hashes
        removal_indices = []
        for i, _hash in enumerate(veto_hashes):
            if _hash not in self.states_hashes:
                removal_indices.append(i)

        # get the new states which are not in the veto_hashes
        new_quantity = []
        for i, state in enumerate(self.states):
            if self.states_hashes[i] not in veto_hashes:
                new_quantity.append(state["quantities"][quantity])

        return new_quantity, removal_indices

    def get_loglikes(self):
        # returns all loglikes     
        return jnp.array([state["total_loglike"] for state in self.states])
    
    def get_max_loglike(self):

        loglikes = self.get_loglikes()

        if len(loglikes) == 0:
            max_loglike = -jnp.inf
        else:
            max_loglike = jnp.max(loglikes)

        return max_loglike


    def store_cache(self):
        # store the cache in the hdf5 file

        # if MPI, we need to make sure that we have a loop to wait until the file is not locked anymore
        with fasteners.InterProcessLock(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"] + ".lock"):
            with open(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"], "wb") as fp:
                pickle.dump((self.states, 0), fp)

        self.debug("Stored cache in file: %s", self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"])

    def synchronize_to_cache(self, new_state):
        # This function reads the old states from the cache file, adds the new state and stores the new states in the cache file.
        # This is useful if we want to store the cache from different processes.

        # if MPI, we need to make sure that we have a loop to wait until the file is not locked anymore
        with fasteners.InterProcessLock(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"] + ".lock"):
            with open(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"], "rb") as fp:
                old, old_recently_added = pickle.load(fp)

            old.append(new_state)

            # here we need to remove states if the cache is full
            if len(old) > self.hyperparameters["cache_size"]:
                min_loglike = min([state["total_loglike"] for state in old])
                for i, state in enumerate(old):
                    if state["total_loglike"] == min_loglike:
                        old.pop(i)
                        break

            # remove states whose loglike is to far away from the maximum loglike
            max_loglike = max([state["total_loglike"] for state in old])

            valid_indices = []
            for i, state in enumerate(old):
                if not (
                    abs(state["total_loglike"] - max_loglike)
                    > self.hyperparameters["delta_loglike"]
                ):
                    # remove index from mask
                    valid_indices.append(i)

            # keep valid states
            old = [old[i] for i in valid_indices]

            with open(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"], "wb") as fp:
                pickle.dump((old, old_recently_added+1), fp)

            self.recently_added = old_recently_added + 1

            self.states = old

            self.max_loglike = self.get_max_loglike()

    def load_cache(self):
        # if we give the deployed_hashes, we only load the states which are not in the deployed_hashes
        remove_flag = np.zeros(len(self.states), dtype=bool)

        with fasteners.InterProcessLock(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"] + ".lock"):
            with open(self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"], "rb") as fp:
                stored_states, self.recently_added = pickle.load(fp)

                for state in stored_states:
                    _hash = hash(str(state["parameters"]))

                    if _hash not in self.states_hashes:
                        self.states.append(state)

                    for i, _ in enumerate(self.states_hashes):
                        if _hash == self.states_hashes[i]:
                            remove_flag[i] = True

        # now remove the data points which are not in the cache anymore
        for i in reversed(range(len(self.states_hashes))):
            if not remove_flag[i]:
                self.states.pop(i)

        self.update_hashes()

        self.debug("Loaded cache from file: %s", self.hyperparameters["working_directory"] + self.hyperparameters["cache_file"])

    def update_hashes(self):
        self.states_hashes = [hash(str(state["parameters"])) for state in self.states]
        self.cache_hash = hash(str(self.states_hashes))

    def load_compressed_cache(self):
        # if we give the deployed_hashes, we only load the states which are not in the deployed_hashes
        remove_flag = np.zeros(len(self.states), dtype=bool)

        with fasteners.InterProcessLock(self.hyperparameters["working_directory"] + self.hyperparameters["compressed_cache_file"] + ".lock"):
            with open(self.hyperparameters["working_directory"] + self.hyperparameters["compressed_cache_file"], "rb") as fp:
                stored_states = pickle.load(fp)

                for state in stored_states:
                    _hash = hash(str(state["parameters"]))

                    if _hash not in self.states_hashes:
                        self.states.append(state)

                    for i, _ in enumerate(self.states_hashes):
                        if _hash == self.states_hashes[i]:
                            remove_flag[i] = True

        # now remove the data points which are not in the cache anymore
        for i in reversed(range(len(self.states_hashes))):
            if not remove_flag[i]:
                self.states.pop(i)

        self.update_hashes()

        self.debug("Loaded cache from file: %s", self.hyperparameters["working_directory"] + self.hyperparameters["compressed_cache_file"])
