import jax.numpy as jnp
import jax
import numpy as np
import time
import logging
import cobaya as my_cobaya
from cobaya.theory import Theory
from cobaya.log import LoggedError, always_stop_exceptions

from cobaya.component import Timer

from functools import partial
import copy
import gc

# global force_acceptance # OLD 

def check_cache_and_compute(self, params_values_dict,
                                dependency_params=None, want_derived=False, cached=True):
    
    """
    Takes a dictionary of parameter values and computes the products needed by the
    likelihood, or uses the cached value if that exists for these parameters.
    If want_derived, the derived parameters are saved in the computed state
    (retrieved using current_derived).

    params_values_dict can be safely modified and stored.
    """
    # Try to build emulator from saved cobaya state and cache
        
    # There is a possibility when using the emulator to load an initial state from the cache of the emulator, 
    # which then allows to perform the sampling without a single call to the theory
    if self.emulate:
        # force_acceptance = False # OLD
        if self.emulator is None:
            # if self.emulator_settings has the key, 'load_initial_state', we can load the initial state from the emulator
            if 'load_initial_state' in self.emulator_settings.keys():
                if self.emulator_settings['load_initial_state']:

                    # we can only load the initial state if an cobaya state is given
                    if self.emulator_settings['cobaya_state_file'] is None:
                        self.log.info("No cobaya state file given. Cannot load initial state from emulator")
                    else:
                        # check that the file exists
                        import os
                        if not os.path.exists(self.emulator_settings['cobaya_state_file']):
                            self.log.info("The cobaya state file does not exist. Cannot load initial state from emulator. Running theory code to generate initial state")
                        else:
                            # load the initial state from the emulator which is a pickle
                            import pickle
                            with open(self.emulator_settings['cobaya_state_file'], 'rb') as f:
                                self.initial_cobaya_state = pickle.load(f)

                            # avoid stupid cobaya bug here (let theory)
                            self.timer = Timer()
                            self.timer.start()
                            self.timer.increment()

                            # import the emulator
                            from OLE.emulator import Emulator

                            # initialize the emulator
                            self.emulator = Emulator(**self.emulator_settings)
                            self.emulator.initialize(ini_state=None, **self.emulator_settings)

                            # jit the emulator functions                                         
                            del self.emulator_emulate_function
                            del self.emulator_sampling_function
                            del self.jit_emulate
                            del self.jit_emulator_samples
                            self.emulator_emulate_function = self.emulator.emulate
                            self.emulator_sampling_function = self.emulate_samples

                            # self.jit_emulate = jax.jit(self.emulator_emulate_function)
                            # self.jit_emulator_samples = jax.jit(self.emulator_sampling_function)
                            self.jit_emulate = self.emulator_emulate_function
                            self.jit_emulator_samples = self.emulator_sampling_function

                            self.log.info("Emulator trained")

                            self._current_state = self.initial_cobaya_state


    start = time.time()
    # This is a flag which is used when we want to compute the delta_loglikelihood for the emulator
    if self.skip_theory_state_from_emulator is not None:
        # here we are in the emulator and we need to use the emulator state
        state = self.skip_theory_state_from_emulator
        self.skip_theory_state_from_emulator = None
        self._current_state = state
        return True

    if self._input_params_extra:
        params_values_dict.update(
            zip(self._input_params_extra,
                self.provider.get_param(self._input_params_extra)))
    self.log.debug("Got parameters %r", params_values_dict)
    state = None
    if cached:
        for _state in self._states:
            # compare dictionaries elementwise
            same = True
            try: 
                if (params_values_dict != _state["params"]):
                    same = False
            except:
                for key, value in _state["params"].items():
                    if (value != params_values_dict[key]).any():
                        same = False


            if same and \
                    _state["dependency_params"] == dependency_params \
                    and (not want_derived or _state["derived"] is not None):
                state = _state
                self.log.debug("Re-using computed results")
                self._states.remove(_state)
                break
    if not state:
        successful_emulation = False
        if self.emulate and (self.emulator is not None):
            # try to emulate a state
            self.log.debug("Try emulating new state")

            # go to the emulator state
            state = {"params": params_values_dict,
                        "dependency_params": dependency_params,
                        "derived": {} if want_derived else None}
            
            emulator_state = translate_cobaya_state_to_emulator_state(state)

            # test whether the emulator can be used
            successful_emulation, emulator_state = self.test_emulator(emulator_state)

            if successful_emulation:
                # translate the emulator state back to the cobaya state
                state = translate_emulator_state_to_cobaya_state(self._current_state, emulator_state)
                self.log.debug("Emulation successful")
            else:
                self.log.debug("Emulation not successful")

        if not successful_emulation:
            self.log.debug("Computing new state")
            state = {"params": params_values_dict,
                        "dependency_params": dependency_params,
                        "derived": {} if want_derived else None}
            if self.timer:
                self.timer.start()
            try:
                if self.calculate(state, want_derived, **params_values_dict) is False:
                    return False
                else:
                    # we might want to store an cobaya state for the emulator such that we can reuse it when restarting the emulator\
                    if self.initial_cobaya_state is None:
                        if 'cobaya_state_file' in self.emulator_settings.keys():
                            if self.emulator_settings['cobaya_state_file'] is not None:
                                self.initial_cobaya_state = state
                                import pickle
                                with open(self.emulator_settings['cobaya_state_file'], 'wb') as f:
                                    pickle.dump(state, f)
            except always_stop_exceptions:
                raise
            except Exception as excpt:
                if self.stop_at_error:
                    self.log.error("Error at evaluation. See error information below.")
                    raise
                else:
                    self.log.debug(
                        "Ignored error at evaluation and assigned 0 likelihood "
                        "(set 'stop_at_error: True' as an option for this component "
                        "to stop here and print a traceback). Error message: %r", excpt)
                    return False
            if self.timer:
                self.timer.increment(self.log)
            if self.emulator is not None:
                # add the new state to the emulator
                emulator_state = translate_cobaya_state_to_emulator_state(state)

                # Be aware! This is recursive! Thus, we need to flag this out
                self.skip_theory_state_from_emulator = state
                likelihoods = list(self.provider.model._loglikes_input_params(self.provider.params, cached=False, return_derived = False))

                emulator_state['loglike'] = np.array([sum(likelihoods)])

                # import sys
                # sys.exit()
                added = self.emulator.add_state(emulator_state)

                # if the emulator is not trained, train it, if enough states are available
                if not self.emulator.trained:
                    if len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']:
                        # force_acceptance = True # OLD
                        self.emulator.train()

                        # jit the emulator functions
                        del self.emulator_emulate_function
                        del self.emulator_sampling_function
                        self.emulator_emulate_function = self.emulator.emulate
                        self.emulator_sampling_function = self.emulate_samples
                        gc.collect()       
                        # self.jit_emulate = jax.jit(self.emulator_emulate_function)
                        # self.jit_emulator_samples = jax.jit(self.emulator_sampling_function)

                        self.jit_emulate = self.emulator_emulate_function
                        self.jit_emulator_samples = self.emulator_sampling_function

                        

                        self.log.info("Emulator trained")
                else:
                    if added:
                        # force_acceptance = True #OLD

                        # here we need to re jit    
                        del self.jit_emulate
                        del self.jit_emulator_samples

                        # del self.emulator 

                        gc.collect()       

                        self.emulator_emulate_function = self.emulator.emulate
                        self.emulator_sampling_function = self.emulate_samples

                        # self.jit_emulate = jax.jit(self.emulator_emulate_function)
                        # self.jit_emulator_samples = jax.jit(self.emulator_sampling_function)

                        self.jit_emulate = self.emulator_emulate_function
                        self.jit_emulator_samples = self.emulator_sampling_function

    # transform each element in state["params"] to a float. Otherwise the cache wont work.
    for key, value in state["params"].items():
        state["params"][key] = float(value)

    # make this state the current one
    _ = copy.deepcopy(state)
    self._states.appendleft(_)
    self._current_state = state

    # if we want to use the emulator here, we need to initialize it here
    if self.emulate:
        if self.emulator is None:
            # import the emulator
            from OLE.emulator import Emulator

            # translate the cobaya state to the emulator state
            emulator_state = translate_cobaya_state_to_emulator_state(state)

            # Be aware! This is recursive! Thus, we need to flag this out
            self.skip_theory_state_from_emulator = state

            likelihoods = list(self.provider.model._loglikes_input_params(self.provider.params, cached=False, return_derived = False))

            emulator_state['loglike'] = np.array([sum(likelihoods)])

            self.initial_emulator_state = copy.deepcopy(emulator_state)

            # initialize the emulator
            self.emulator = Emulator(**self.emulator_settings)
            self.emulator.initialize(ini_state = self.initial_emulator_state, **self.emulator_settings)

            if self.emulator.trained:
                # jit the emulator functions
                del self.emulator_emulate_function
                del self.emulator_sampling_function
                del self.jit_emulate
                del self.jit_emulator_samples
                self.emulator_emulate_function = self.emulator.emulate
                self.emulator_sampling_function = self.emulate_samples

                # self.jit_emulate = jax.jit(self.emulator_emulate_function)
                # self.jit_emulator_samples = jax.jit(self.emulator_sampling_function)
                self.jit_emulate = self.emulator_emulate_function
                self.jit_emulator_samples = self.emulator_sampling_function

        # check if the emulator was used 'jit_threshold' times and retrain it
        if self.emulator.continuous_successful_calls-1 == self.emulator.hyperparameters['jit_threshold']:
            del self.emulator_emulate_function
            del self.emulator_sampling_function
            del self.jit_emulate
            del self.jit_emulator_samples
            self.emulator_emulate_function = self.emulator.emulate
            self.emulator_sampling_function = self.emulator.emulate_samples

            self.jit_emulate = jax.jit(self.emulator_emulate_function)
            self.jit_emulator_samples = jax.jit(self.emulator_sampling_function)


    stop = time.time()
    self.log.debug("Time for check_cache_and_compute: %f", stop-start)

    return True


# function that samples multiple emulated spectra and decides whether to accept the quality of the emulation. Additionally, it can return the emulated state
def test_emulator(self,emulator_state):


    # check whether there is an emulator
    if self.emulator is None:
        return False, None

    # check whether the emulator is trained
    if not self.emulator.trained:
        return False, None

    # first we ask the emulator whether it is requried to run a quality check
    a = time.time()
    if self.emulator.require_quality_check(emulator_state['parameters']):
        # if yes. We sample multiple emulator states and estiamte the accuracy on the loglikelihood prediction!
        # if not, we trust the emulator and we can just run it
        # sample multiple spectra
        #emulator_sample_states = self.emulator.emulate_samples(emulator_state['parameters'])
        local_key = jax.random.PRNGKey(time.time_ns())
        emulator_sample_states, _ = self.jit_emulator_samples(emulator_state['parameters'], local_key)

        # compute the likelihoods
        emulator_sample_loglikes = []
        for emulator_sample_state in emulator_sample_states:

            # Translate the emulator state to the cobaya state
            cobaya_sample_state = translate_emulator_state_to_cobaya_state(self._current_state, emulator_sample_state)

            self.skip_theory_state_from_emulator = cobaya_sample_state
            likelihoods = list(self.provider.model._loglikes_input_params(self.provider.params, cached = False, return_derived = False))
            emulator_sample_loglikes.append(sum(likelihoods))

        emulator_sample_loglikes = np.array(emulator_sample_loglikes)
        print('emulator_sample_loglikes')
        print(emulator_sample_loglikes)

        # we also need the reference likelihood which is going to be used eventually
        predictions = self.jit_emulate(emulator_state['parameters'])
        predictions_cobaya_state = translate_emulator_state_to_cobaya_state(self._current_state, predictions)

        self.skip_theory_state_from_emulator = predictions_cobaya_state

        reference_loglike = sum(self.provider.model._loglikes_input_params(self.provider.params, cached = False, return_derived = False))

        # check whether the emulator is good enough
        if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=emulator_state['parameters'], reference_loglike = reference_loglike):
            print("Emulator not good enough")
            return False, None
        else:
            print("Emulator good enough")
            # Add the point to the quality points
            self.emulator.add_quality_point(emulator_state['parameters'])

        b = time.time()
        print("Time for emulator test: ", b-a)

    else:
        # now run the emulator
        a = time.time()
        # predictions = self.emulator.emulate(emulator_state['parameters'])
        
        predictions = self.jit_emulate(emulator_state['parameters'])

        b = time.time()
        print("Run the emulator: ", b-a)

    # if we save a store theory data path we can save the emulator state
    if 'store_prediction' in self.emulator_settings.keys():
        # open file and store prediction dictionary with pickle
        import pickle

        # first check whether the file exists
        import os
        if os.path.exists(self.emulator_settings['store_prediction']):
            # if the file exists, we need to load the dictionary and add the new predictions
            with open(self.emulator_settings['store_prediction'], 'rb') as f:
                old_predictions = pickle.load(f)

            # add the new predictions
            old_predictions.append(predictions)

            if len(old_predictions) > 1000:
                # pop the first element
                old_predictions.pop(0)

            # store the new dictionary
            with open(self.emulator_settings['store_prediction'], 'wb') as f:
                pickle.dump(old_predictions, f)
            
        else:
            # if the file does not exist, we can just store the predictions
            with open(self.emulator_settings['store_prediction'], 'wb') as f:
                pickle.dump([predictions], f)


    self.emulator_counter += 1
    print('emulator_counter')
    print(self.emulator_counter)

    return True, predictions

#@partial(jax.jit, static_argnums=(0,))
def emulate_samples(self,parameters, key):
    return self.emulator.emulate_samples(parameters, key)

# This function calls all likelihoods and computes the total likelihood. It is used to compute the likelihood for new emulator states or to qualify the accuracy of the emulator
def evaluate_cobaya_pipeline(self,state):

    return state


# give the theory class the new attribute
Theory.emulate = False
Theory.emulator = None
Theory.emulator_counter = 0
Theory.skip_theory_state_from_emulator = None
Theory.emulator_settings = {}
Theory.emulator_sampling_function = None
Theory.emulator_emulate_function = None
Theory.jit_emulate = None
Theory.jit_emulator_samples = None
Theory.initial_cobaya_state = None

# Theory flag


# give the theory class the new method which incorporates the emulator
Theory.test_emulator = test_emulator
Theory.check_cache_and_compute = check_cache_and_compute
Theory.emulate_samples = emulate_samples


#
#  Auxiliary functions
#

def translate_cobaya_state_to_emulator_state(state):
    emulator_state = {}
    emulator_state['parameters'] = {}
    emulator_state['quantities'] = {}
    emulator_state['loglike'] = None

    # try to read the parameters and quantities from the cobaya state
    #try:
    for key, value in state.items():

        if key == 'params':

            # translate the parameters to the emulator state
            emulator_state['parameters'] = {}
            for key2, value2 in value.items():
                emulator_state['parameters'][key2] = jnp.array([value2])

            continue

        # if the value is a dictionary, we have to go one layer deeper
        if type(value) == dict:
            for key2, value2 in value.items():
                if type(value2) == dict:
                    for key3, value3 in value2.items():
                        emulator_state['quantities'][key3] = jnp.array([value3])
                else:
                    emulator_state['quantities'][key2] = jnp.array([value2])
        else:
            if len(value) > 1:
                emulator_state['quantities'][key] = jnp.array([value])

    # recheck the dimensionality of the emulator_state. All quantities should be 1D arrays
    for key, value in emulator_state['quantities'].items():
        if len(value.shape) > 1:
            emulator_state['quantities'][key] = jnp.array(value[0])
    for key, value in emulator_state['parameters'].items():
        if len(value.shape) > 1:
            emulator_state['parameters'][key] = jnp.array(value[0])


    return emulator_state

def translate_emulator_state_to_cobaya_state(old_cobaya_state, emulator_state):
    # this function uses the old cobaya state and the emulator state to create the new cobaya state.
    # we go through all keys in the old cobaya state and check whether they are in the emulator state

    # first update the parameters
    for key, value in emulator_state['parameters'].items():
        old_cobaya_state['params'][key] = value[0]

    # now we can go through the keys
    for key, value in old_cobaya_state.items():

        # if the value is a dictionary, we have to go one layer deeper
        if type(value) == dict:
            for key2, value2 in value.items():
                if type(value2) == dict:
                    for key3, value3 in value2.items():
                        # if the key is not in the emulator state, we can skip it
                        if key3 not in emulator_state['quantities'].keys():
                            continue

                        old_cobaya_state[key][key2][key3] = np.array(emulator_state['quantities'][key3])
                else:
                    # if the key is not in the emulator state, we can skip it
                    if key2 not in emulator_state['quantities'].keys():
                        continue

                    old_cobaya_state[key][key2] = np.array(emulator_state['quantities'][key2])
        else:

            # if the key is not in the emulator state, we can skip it
            if key not in emulator_state['quantities'].keys():
                continue

            old_cobaya_state[key] = np.array(emulator_state['quantities'][key])


    import sys 
    # sys.exit()

    return old_cobaya_state



from cobaya.model import LogPosterior


# REMOVED OLD CONCEPT OF FORCING ACCEPTANCE
'''
#
# Here we need to modify the MCMC 
#

def metropolis_accept(self, logp_trial, logp_current):
    global force_acceptance
    """
    Symmetric-proposal Metropolis-Hastings test.

    Returns
    -------
    ``True`` or ``False``.
    """
    if logp_trial == -np.inf:
        return False
    if force_acceptance:
        force_acceptance = False
        return True
    if logp_trial > logp_current:
        return True
    posterior_ratio = (logp_current - logp_trial) / self.temperature
    return self._rng.standard_exponential() > posterior_ratio


from cobaya.samplers.mcmc import MCMC

MCMC.metropolis_accept = metropolis_accept
'''