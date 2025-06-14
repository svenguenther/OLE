import jax.numpy as jnp
import jax
import numpy as np
import time
import logging
import cobaya as my_cobaya
from cobaya.theory import Theory
from cobaya.samplers.mcmc import MCMC
from cobaya.log import LoggedError, always_stop_exceptions

from cobaya.component import Timer

from functools import partial
import copy
import gc

global CAMB_flag_to_skip_CAMB_transfers
EMULATOR_UPDATED_FLAG = False
CAMB_flag_to_skip_CAMB_transfers = False # This ugly flag because CAMB is split into 2 theory codes...

try:
    import importlib
    import importlib.util
    import sys

    # Since we would like to use hte cobaya intrinsic 'boltzmannbase' theory, we need to adapt the source code of the cobaya intrinsic 'boltzmannbase' theory to use the OLE theory interface.
    # 
    # For all other cases, e.g. we define our own cobaya - Theory class, we can use the OLE theory interface directly.
    # We do this by replacing "from cobaya.theory import Theory" by "from OLE.interfaces.cobaya_interface import Theory

    spec = importlib.util.find_spec('cobaya.theories.cosmo.boltzmannbase', 'cobaya')
    source = spec.loader.get_source('cobaya.theories.cosmo.boltzmannbase')

    # write replace function to let boltzmannbase not import 'from cobaya.theory import Theory', but 'from OLE.theory import Theory'
    def replace(source):
        source = source.replace("from cobaya.theory import Theory", "from OLE.interfaces.cobaya_interface import Theory")
        return source

    source = replace(source)
    module = importlib.util.module_from_spec(spec)
    codeobj = compile(source, module.__spec__.origin, 'exec')
    exec(codeobj, module.__dict__)
    sys.modules['cobaya'].theories.cosmo.boltzmannbase = module

    import cobaya 
    cobaya.theories.cosmo.boltzmannbase = module
    importlib.reload(cobaya)

    # Additionally, due to some incompatibility of the logging systems, we do need to manually silence some of the logging output of the jax and matplotlib libraries.
    import logging
    logger = logging.getLogger('root')
    logger.disabled = True
    logger = logging.getLogger('jax._src.dispatch')
    logger.disabled = True
    logger = logging.getLogger('jax._src.compiler')
    logger.disabled = True
    logger = logging.getLogger('jax.experimental.host_callback')
    logger.disabled = True
    logger = logging.getLogger('jax._src.xla_bridge')
    logger.disabled = True
    logger = logging.getLogger('jax._src.interpreters.pxla')
    logger.disabled = True
    logger = logging.getLogger('matplotlib.font_manager')
    logger.disabled = True
    logger = logging.getLogger('jax._src.interpreters.pxla')
    logger.disabled = True
except:
    print("Could not replace the cobaya.theory import. This is not a problem if you are not using the boltzmannbase theory.")



EMULATOR_INITIALLY_TRAINED_FLAG = False
from cobaya.samplers.mcmc.proposal import BlockedProposer
# The 'emulator_settings' key is a dictionary which contains the settings for the emulator. If not given the default settings are used which can be found in the corresponding OLE python files.
def OLE_callback_function(self):
    global EMULATOR_INITIALLY_TRAINED_FLAG
    if EMULATOR_INITIALLY_TRAINED_FLAG:
        sampled_params_list = list(self.model.parameterization.sampled_params())
        blocks_indices = [[sampled_params_list.index(p) for p in b] for b in self.blocks]
        # append all lists to one
        blocks_indices = [item for sublist in blocks_indices for item in sublist]
        self.blocks = tuple([[sampled_params_list[i] for i in blocks_indices]])
        self.oversampling_factors = [1]

        self.drag = False

        self.proposer = BlockedProposer(
                [blocks_indices], self._rng,
                oversampling_factors=self.oversampling_factors,
                i_last_slow_block=(self.i_last_slow_block if self.drag else None),
                proposal_scale=self.proposal_scale)
        
        self.set_proposer_initial_covmat(load=True)

        self.cycle_length = sum(len(b) * o for b, o in
                                    zip([blocks_indices], self.oversampling_factors))

        if self.oversample_thin:
            self.current_point.output_thin = int(
                np.round(
                    sum(
                        len(b) * o for b, o in zip(
                            self.blocks,
                            self.oversampling_factors,
                        )
                    ) / self.model.prior.d()
                )
            )        

        for number in self._quants_d_units:
            number.set_scale(self.cycle_length // self.current_point.output_thin)

        EMULATOR_INITIALLY_TRAINED_FLAG = False





def check_cache_and_compute(self, params_values_dict,
                                dependency_params=None, want_derived=False, cached=True):
    global CAMB_flag_to_skip_CAMB_transfers
    global EMULATOR_UPDATED_FLAG
    global EMULATOR_INITIALLY_TRAINED_FLAG
    for param in params_values_dict.keys():
        self.theory_params[param] = params_values_dict[param]

    """
    Takes a dictionary of parameter values and computes the products needed by the
    likelihood, or uses the cached value if that exists for these parameters.
    If want_derived, the derived parameters are saved in the computed state
    (retrieved using current_derived).

    params_values_dict can be safely modified and stored.
    """
    if (self._name == 'camb.transfers') and CAMB_flag_to_skip_CAMB_transfers:
        self._current_state = {'params': {}, 'derived': {}, 'results': {}}
        return True
    
    # This is a flag which is used when we want to compute the delta_loglikelihood for the emulator
    if self.skip_theory_state_from_emulator is not None:
        # here we are in the emulator and we need to use the emulator state
        state = self.skip_theory_state_from_emulator
        self.skip_theory_state_from_emulator = None
        self._current_state = state
        return True

    # start timer for likelihood computation/oversampling
    if self.emulator is not None:
        self.emulator.increment("likelihood")

    
    # Try to build emulator from saved cobaya state and cache
        
    # There is a possibility when using the emulator to load an initial state from the cache of the emulator, 
    # which then allows to perform the sampling without a single call to the theory
    if self.emulate:
        # force_acceptance = False # OLD
        if self.emulator is None:

            # read out cobaya settings to update OLE settings 
            # output
            if 'working_directory' not in self.emulator_settings.keys():
                global GLOBAL_COBAYA_OUTPUT_DIRECTORY
                self.emulator_settings['working_directory'] = GLOBAL_COBAYA_OUTPUT_DIRECTORY

            # get the number of total varied parameters
            if 'dimensionality' not in self.emulator_settings.keys():
                self.emulator_settings['dimensionality'] = len(self.provider.model.parameterization.sampled_params())

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
                            self.log.info("Emulator trained")

                            self._current_state = self.initial_cobaya_state


    start = time.time()


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
            # we have to make this work with CAMB
            for key, value in params_values_dict.items():
                if (value != _state["params"][key]):
                    same = False

            if same and (_state["derived"] is not None): # Here we dont check for derived because CAMB. Lets hope for the best
                state = _state
                self.log.debug("Re-using computed results")
                self._states.remove(_state)
                break
    if not state: 
        successful_emulation = False
        if self.emulate and (self.emulator is not None):
            # try to emulate a state
            self.log.debug("Try emulating new state")

            if hasattr(self, '_camb_transfers'):
                params_values_dict.update(self._camb_transfers.theory_params)

            state = {"params": params_values_dict,
                        "dependency_params": dependency_params,
                        "derived": {} if want_derived else None}
            
            emulator_state = translate_cobaya_state_to_emulator_state(state)

            # test whether the emulator can be used
            successful_emulation, emulator_state = self.test_emulator(emulator_state)

            if successful_emulation:
                # translate the emulator state back to the cobaya state
                state = translate_emulator_state_to_cobaya_state(self._current_state, emulator_state)
                CAMB_flag_to_skip_CAMB_transfers = True
                self.log.debug("Emulation successful")
            else:
                CAMB_flag_to_skip_CAMB_transfers = False # When using CAMB we need to reset this flag such that CAMB_transfers is computed
                # raise error
                self.log.debug("Emulation not successful")

                if self._name == 'camb':
                    # delete all keys which are not in the theory_params
                    new_params_values_dict = copy.deepcopy(params_values_dict)
                    for key in params_values_dict.keys():
                        if key not in self._camb_transfers.theory_params.keys():
                            del new_params_values_dict[key]

                    # weird hack
                    self._camb_transfers.check_cache_and_compute(new_params_values_dict, dependency_params, want_derived, cached) 

        if not successful_emulation:
            self.log.debug("Computing new state")

            if self.emulator is not None:
                self.emulator.start("theory_code")

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
                self.emulator.increment("theory_code")
                self.emulator.print_status()

                # add the new state to the emulator
                emulator_state = translate_cobaya_state_to_emulator_state(state)

                # Be aware! This is recursive! Thus, we need to flag this out
                self.skip_theory_state_from_emulator = state
                likelihoods = list(self.provider.model._loglikes_input_params(self.provider.params, cached=False, return_derived = False))

                emulator_state['total_loglike'] = np.array([sum(likelihoods)])

                # import sys
                # sys.exit()
                added, EMULATOR_UPDATED_FLAG = self.emulator.add_state(emulator_state)

                # if the emulator is not trained, train it, if enough states are available
                if not self.emulator.trained:
                    if len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']:
                        # force_acceptance = True # OLD
                        self.emulator.train()
                        EMULATOR_UPDATED_FLAG = True
                        EMULATOR_INITIALLY_TRAINED_FLAG = True

                        self.log.info("Emulator trained")

                    gc.collect()  

    # transform each element in state["params"] to a float. Otherwise the cache wont work.
    for key, value in state["params"].items():
        state["params"][key] = np.array(value)

    # make this state the current one
    try:
        _ = copy.deepcopy(state) # deepcopy to keep it in the cache :)
    except:
        _ = state # we do this try because of CAMB FORTRAN things

    self._states.appendleft(_)
    self._current_state = _

    if EMULATOR_UPDATED_FLAG:
        # delete cache
        self._states.clear()

    # if we want to use the emulator here, we need to initialize it here
    if self.emulate:
        if self.emulator is None:
            # import the emulator
            from OLE.emulator import Emulator

            # here we need to make an CAMB specific hack ...
            if hasattr(self, '_camb_transfers'):
                state['params'].update(self._camb_transfers.theory_params) 

            # translate the cobaya state to the emulator state
            emulator_state = translate_cobaya_state_to_emulator_state(state)

            # Be aware! This is recursive! Thus, we need to flag this out
            self.skip_theory_state_from_emulator = state

            likelihoods = list(self.provider.model._loglikes_input_params(self.provider.params, cached=False, return_derived = False))

            emulator_state['total_loglike'] = np.array([sum(likelihoods)])

            self.initial_emulator_state = copy.deepcopy(emulator_state)

            # initialize the emulator
            self.emulator = Emulator(**self.emulator_settings)
            self.emulator.initialize(ini_state = self.initial_emulator_state, **self.emulator_settings)


    stop = time.time()
    self.log.debug("Time for check_cache_and_compute: %f", stop-start)


    # start timer for likelihood computation/oversampling
    if self.emulator is not None:
        self.emulator.start("likelihood")


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
    if self.emulator.require_quality_check(emulator_state['parameters']):
        # if yes. We sample multiple emulator states and estiamte the accuracy on the loglikelihood prediction!
        # if not, we trust the emulator and we can just run it
        # sample multiple spectra
        #emulator_sample_states = self.emulator.emulate_samples(emulator_state['parameters'])
        local_key = jax.random.PRNGKey(time.time_ns())
        emulator_sample_states, _ = self.emulator.emulate_samples(emulator_state['parameters'], local_key)

        # compute the likelihoods
        emulator_sample_loglikes = []
        self.emulator.start("likelihood_testing")
        for i, emulator_sample_state in enumerate(emulator_sample_states):

            # Translate the emulator state to the cobaya state
            cobaya_sample_state = translate_emulator_state_to_cobaya_state(self._current_state, emulator_sample_state)

            # Be aware! This is recursive! Thus, we need to flag this out
            self.skip_theory_state_from_emulator = cobaya_sample_state

            params_local = copy.deepcopy(self.provider.params)
            # for para, value in params_local.items():
            #     params_local[para] = value*(1.0 + 1.e-5*(i+1))


            likelihoods = list(self.provider.model._loglikes_input_params(params_local, cached = False, return_derived = False))
            emulator_sample_loglikes.append(sum(likelihoods))


        emulator_sample_loglikes = np.array(emulator_sample_loglikes)
        self.emulator.increment("likelihood_testing")

        # we also need the reference likelihood which is going to be used eventually
        predictions = self.emulator.emulate(emulator_state['parameters'])
        predictions_cobaya_state = translate_emulator_state_to_cobaya_state(self._current_state, predictions)

        # Be aware! This is recursive! Thus, we need to flag this out
        self.skip_theory_state_from_emulator = predictions_cobaya_state

        self.emulator.start("likelihood")
        reference_loglike = sum(self.provider.model._loglikes_input_params(self.provider.params, cached = False, return_derived = False))
        self.emulator.increment("likelihood")

        # check whether the emulator is good enough
        if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=emulator_state['parameters'], reference_loglike = reference_loglike):
            return False, None
        else:
            # Add the point to the quality points
            self.emulator.add_quality_point(emulator_state['parameters'])

    else:
        # now run the emulator
        predictions = self.emulator.emulate(emulator_state['parameters'])

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
Theory.skip_theory_state_from_emulator = None
Theory.emulator_settings = {}
Theory.theory_params = {}
Theory.initial_cobaya_state = None

# Theory flag


# give the theory class the new method which incorporates the emulator
Theory.test_emulator = test_emulator
Theory.check_cache_and_compute = check_cache_and_compute
Theory.emulate_samples = emulate_samples



def metropolis_accept(self, logp_trial, logp_current):
    """
    Symmetric-proposal Metropolis-Hastings test.

    Returns
    -------
    ``True`` or ``False``.
    """
    global EMULATOR_UPDATED_FLAG

    # if EMULATOR_UPDATED_FLAG:
    #     EMULATOR_UPDATED_FLAG = False
    #     print("FORCES ACCEPT!")
    #     return True

    if self.current_point.weight >420:
        if logp_trial == -np.inf:
            return False
        else:
            self.current_point.weight = 0.1
            return True



    if logp_trial == -np.inf:
        return False
    if logp_trial > logp_current:
        return True
    posterior_ratio = (logp_current - logp_trial) / self.temperature
    return self._rng.standard_exponential() > posterior_ratio

# MCMC.metropolis_accept = metropolis_accept

#
#  Auxiliary functions
#

def translate_cobaya_state_to_emulator_state(state):
    emulator_state = {}
    emulator_state['parameters'] = {}
    emulator_state['quantities'] = {}
    emulator_state['total_loglike'] = None

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
                    if type(value2) not in [int, float]:
                        emulator_state['quantities'][key2] = jnp.array([value2.flatten()])
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

                        if type(value3) not in [int, float]:
                            old_cobaya_state[key][key2][key3] = np.array(emulator_state['quantities'][key3].reshape(value3.shape))
                        else:

                            old_cobaya_state[key][key2][key3] = np.array(emulator_state['quantities'][key3])
                else:
                    # if the key is not in the emulator state, we can skip it
                    if key2 not in emulator_state['quantities'].keys():
                        continue
                    if type(value2) not in [int, float]:
                        old_cobaya_state[key][key2] = np.array(emulator_state['quantities'][key2].reshape(value2.shape))
                    else:
                        old_cobaya_state[key][key2] = np.array(emulator_state['quantities'][key2])
        else:

            # if the key is not in the emulator state, we can skip it
            if key not in emulator_state['quantities'].keys():
                continue
            
            old_cobaya_state[key] = np.array(emulator_state['quantities'][key])


    return old_cobaya_state




# Weird Hack to obtain direcory of Cobaya
from cobaya.sampler import Sampler
from cobaya.output import OutputDummy, Output

global GLOBAL_COBAYA_OUTPUT_DIRECTORY

@property
def output(self) -> Output:
    global GLOBAL_COBAYA_OUTPUT_DIRECTORY
    GLOBAL_COBAYA_OUTPUT_DIRECTORY = self._output.folder
    return self._output

Sampler.output = output
