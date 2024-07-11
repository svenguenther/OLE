# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
# The NUTS sampler is based upon jaxnuts: https://github.com/guillaume-plc/jaxnuts
from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.theory import Theory
from OLE.likelihood import Likelihood
from OLE.emulator import Emulator
from OLE.utils.mpi import *
from OLE.plotting import data_covmat_plot, covmat_diagonal_plot
from scipy.optimize import minimize

from functools import partial
from typing import Tuple
import jax.numpy as jnp
import jax
import fasteners
from jax.numpy import ndarray
import jax.lax as lax
import random
import copy
import os

import numpy as np
import jax.numpy as jnp

import jax.random as random

import time 

# The sampler class connects the theory, the likelihood, and the emulator.
# It can be also used to interact with the emulator also without giving a specific theory oder likelihood instance.

class Sampler(BaseClass):

    theory: Theory
    likelihood: Likelihood
    parameter_dict: dict

    emulator: Emulator

    
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, 
                   parameters, 
                   likelihood=None, 
                   theory=None,
                   emulator=None, 
                   emulator_settings={},
                   likelihood_settings={},
                    theory_settings={},
                    sampling_settings={},
                   **kwargs):

        # Store Likelihood and initialize if possible
        self.likelihood = likelihood
        if self.likelihood is not None:
            self.likelihood_settings = likelihood_settings
            self.likelihood.initialize(**likelihood_settings)

        # update theory settings by likelihood
        if theory is not None:
            theory_settings = self.likelihood.update_theory_settings(theory_settings)

        # Store Theory and initialize if possible
        self.theory = theory
        if self.theory is not None:
            self.theory_settings = theory_settings
            self.theory.initialize(**theory_settings)

        # Store Emulator and initialize if possible
        self.emulator = None
        if emulator is not None:
            self.emulator = emulator

        # load default parameter dictionary from likelihood
        if self.likelihood is not None:
            self.parameter_dict = self.likelihood.nuisance_sample_dict

        # Update on parameters if given
        if parameters is not None:
            self.parameter_dict.update(parameters)   

        defaulthyperparameters = {
            # output directory for the chain
            'output_directory': 'output',

            # force overwrite
            'force': False, #TODO: implement this

            # load (parameter) covmat from file from a path
            'covmat': None,

            # compute data covmat. This only works for differentiable likelihoods
            'compute_data_covmat': False,

            # plotting directory
            'plotting_directory': None,

            # status print frequency
            'status_print_frequency': 50,

        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in sampling_settings.items():
            self.hyperparameters[key] = value

        # create output directory
        import os
        if not os.path.exists(self.hyperparameters['output_directory']):
            os.makedirs(self.hyperparameters['output_directory'])

        # save proposal lengths. They are useful to normalize the parameters
        self.proposal_lengths = jnp.ones(len(self.parameter_dict))
        self.proposal_means = jnp.zeros(len(self.parameter_dict))
        for i, key in enumerate(self.parameter_dict.keys()):
            if type(self.parameter_dict[key]) is list:
                continue 
            elif 'proposal' in list(self.parameter_dict[key].keys()):
                self.proposal_lengths = self.proposal_lengths.at[i].set(self.parameter_dict[key]['proposal'])
            else:
                raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)
            
            if 'ref' in list(self.parameter_dict[key].keys()):
                self.proposal_means = self.proposal_means.at[i].set(self.parameter_dict[key]['ref']['mean'])
            else:
                self.proposal_means = self.proposal_means.at[i].set(0.5*(self.parameter_dict[key]['prior']['min']+self.parameter_dict[key]['prior']['max']))

        

        # generate the covariance matrix
        self.covmat = self.generate_covmat()

        # go into eigenspace of covmat and build the inverse of the eigenvectors
        self.eigenvalues, self.eigenvectors = jnp.linalg.eigh(self.covmat)
        self.inv_eigenvectors = jnp.linalg.inv(self.eigenvectors)

        # we can now transform the parameters into the (normalized) eigenspace

        # remove the parameters from the test state which are not in self.theory.requirements

        # the likelihood will be initialized once again in the emulator. As a consequence we need to provide the emulator settings with the likelihood settings
        emulator_settings['likelihood_settings'] = likelihood_settings

        # if emulator was not loaded, we need to create one here:
        if self.emulator is None:
            # initialize the emulator
            self.emulator = Emulator(**emulator_settings)

            test_state = None

            if 'load_initial_state' not in emulator_settings:
                test_state = self.test_pipeline()
            else:
                if not emulator_settings['load_initial_state']:
                    test_state = self.test_pipeline()

            # here we check the requirements of the theory code. If they are specified, we initialize the emulator with the requirements
            # otherwise the emulator is inificalized with all parameters which could also include likelihood parameters!
            if len(self.theory.requirements) > 0:
                self.emulator.initialize(self.likelihood, test_state, input_parameters=self.theory.requirements, **emulator_settings)
            else:
                self.emulator.initialize(self.likelihood, test_state, **emulator_settings)


        self.nuisance_parameters = list(self.parameter_dict.keys())
        #remove keys which are in self.theory.requirements
        for key in self.theory.requirements:
            if key in self.nuisance_parameters:
                self.nuisance_parameters.remove(key)

        if len(self.nuisance_parameters) != 0:
            # get nuisance mean and std from the parameter_dict
            self.nuisance_means = jnp.zeros(len(self.nuisance_parameters))
            self.nuisance_stds = jnp.zeros(len(self.nuisance_parameters))

            for i, key in enumerate(self.nuisance_parameters):
                self.nuisance_means = self.nuisance_means.at[i].set(self.parameter_dict[key]['ref']['mean'])
                self.nuisance_stds = self.nuisance_stds.at[i].set(self.parameter_dict[key]['ref']['std'])

        pass

    def update_covmat(self, covmat):
        # this function updates the covmat
        self.covmat = covmat

        # go into eigenspace of covmat and build the inverse of the eigenvectors
        self.eigenvalues, self.eigenvectors = jnp.linalg.eigh(self.covmat)

        # search for negative eigenvalues
        if jnp.any(self.eigenvalues < 0):
            self.error("Covmat contains negative eigenvalues")
            raise ValueError("Covmat contains negative eigenvalues")
        
        # search for nan or inf in the eigenvalues and eigenvectors\
        if jnp.isnan(self.eigenvalues).any() or jnp.isinf(self.eigenvalues).any():
            self.error("Covmat contains nans or infs")
            raise ValueError("Covmat contains nans or infs")
        
        if jnp.isnan(self.eigenvectors).any() or jnp.isinf(self.eigenvectors).any():
            self.error("Covmat contains nans or infs")
            raise ValueError("Covmat contains nans or infs")

        self.inv_eigenvectors = self.eigenvectors.T        

        return 0

    def print_status(self, i, chain):
        if i==0:
            return 0
        # this function computes the effective sample size and prints the status of the chain after the i-th interation. This is only done when i%100==0
        if ((i) % self.hyperparameters['status_print_frequency']) == 0:
            ess = self.estimate_effective_sample_size(chain[:i,:])
            mean_ess = np.mean(ess)
            self.info("Iteration %d: Mean ESS = %f", i, mean_ess)
            self.info("Iteration %d: Mean ESS/sek = %f", i, mean_ess/self.time_from_start())
            
        return 0

    def estimate_effective_sample_size(self, chain):
        ESS = np.zeros(chain.shape[1])

        for i in range(chain.shape[1]):
            mean = np.mean(chain[:,i])
            autocorrelation_1 = np.sum((chain[1:,i]-mean)*(chain[:-1,i]-mean))/np.sum((chain[:,i]-mean)**2)

            ESS[i] = len(chain) / ( (1 + autocorrelation_1)/(1-autocorrelation_1) )

        return ESS

    def denormalize_inv_hessematrix(self, matrix):
        # this function denormalizes the matrix

        a = jnp.dot(self.eigenvectors,  
                    jnp.dot( 
                        jnp.dot(
                            jnp.dot( jnp.diag(jnp.sqrt(self.eigenvalues)),matrix), jnp.diag(jnp.sqrt(self.eigenvalues))), self.eigenvectors.T))

        return a

    def transform_parameters_into_normalized_eigenspace(self, parameters):
        # this function transforms the parameters into the normalized eigenspace
        return jnp.dot( self.inv_eigenvectors, ( parameters - self.proposal_means ))/jnp.sqrt(self.eigenvalues)
    
    def retranform_parameters_from_normalized_eigenspace(self, parameters):
        # this function transforms the parameters back from the normalized eigenspace
        return jnp.dot(self.eigenvectors, parameters * jnp.sqrt(self.eigenvalues) ) + self.proposal_means

    def generate_covmat(self):
        # this function generates the covariance matrix either by loading it from a file or by using the proposal lengths. Note that the covmat might miss some entries which are to be filled with the proposal lengths.
        
        # first: create diagonal covmat from proposal lengths
        covmat = np.zeros((len(self.parameter_dict), len(self.parameter_dict)))
        for i in range(len(self.parameter_dict)):
            covmat[i,i] = self.proposal_lengths[i]**2

        # second if the covmat is given (at least for some parameters), we need to fill the covmat with the given values
        if self.hyperparameters['covmat'] is not None:
            # load the covmat from a file. The first line is the strings of the parameters, the rest is the covmat
            with open(self.hyperparameters['covmat'], 'r') as f:
                lines = f.readlines()
                parameters = lines[0].split()
                covmat_loaded = np.zeros((len(parameters), len(parameters)))
                for i, line in enumerate(lines[1:]):
                    covmat_loaded[i] = np.array([float(_) for _ in line.split()])

            self.info("Loaded covmat from file: %s", self.hyperparameters['covmat'])
            self.info("Loaded parameters: %s", parameters)

            # fill the covmat with the given values
            for i, key in enumerate(self.parameter_dict.keys()):
                if key in parameters:
                    for j, key2 in enumerate(self.parameter_dict.keys()):
                        if key2 in parameters:
                            covmat[j,i] = covmat_loaded[parameters.index(key2), parameters.index(key)]
        
        # create the covmat from the proposal lengths
        return covmat

    def get_initial_position(self, N=1, normalized=False):
        # this function returns N initial positions for the sampler. It samples from the 'ref' values of the parameters.
        positions = []
        for i in range(N):
            position = []
            for key, value in self.parameter_dict.items():
                if 'ref' in list(value.keys()):
                    # create candidate from normal distribution with mean and std from the 'ref' values but ensure that it is within the prior
                    while True:
                        candidate = value['ref']['mean'] + value['ref']['std']*np.random.normal()
                        if candidate > value['prior']['min'] and candidate < value['prior']['max']:
                            break
                    position.append(candidate)
                elif 'prior' in list(value.keys()):
                    position.append(value['prior']['min'] + (value['prior']['max']-value['prior']['min'])*np.random.uniform())
                else:
                    raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)
            positions.append(position)

        if normalized:
            for i in range(N):
                positions[i] = self.transform_parameters_into_normalized_eigenspace(np.array(positions[i]))
            return positions
        else:
            return np.array(positions)

    def get_bounds(self, normalized=False):
        # this function returns the bounds of the parameters
        bounds = []
        for key, value in self.parameter_dict.items():
            if 'prior' in list(value.keys()):
                bounds.append((value['prior']['min'], value['prior']['max']))
            else:
                raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)
        
        if normalized:
            lower_bound = np.array([_[0] for _ in bounds])
            upper_bound = np.array([_[1] for _ in bounds])

            normalized_bounds = np.vstack([self.transform_parameters_into_normalized_eigenspace(lower_bound),
                                           self.transform_parameters_into_normalized_eigenspace(upper_bound)])
            # put the bounds in the right order\
            bounds = []
            for i in range(len(normalized_bounds[0])):
                bounds.append((normalized_bounds[0,i], normalized_bounds[1,i]))

            return bounds
        else:
            return bounds

    def test_pipeline(self):
        # Create a test state from the given parameters.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None, 'loglike_gradient': {}}

        RNG = jax.random.PRNGKey(int(time.time())+get_mpi_rank())

        # translate the parameters to the state
        for key, value in self.parameter_dict.items():
            RNG, subkey = jax.random.split(RNG)
            if type(value) is list:
                state['parameters'][key] = jnp.array(value)
            elif 'ref' in list(value.keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict[key]['ref']['mean'] + self.parameter_dict[key]['ref']['std']*jax.random.normal(subkey)])
            elif 'prior' in list(value.keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict[key]['prior']['min'] + (self.parameter_dict[key]['prior']['max']-self.parameter_dict[key]['prior']['min'])*jax.random.uniform(subkey)]) # 
            else:
                raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)

        # compute the observables
        state = self.theory.compute(state)

        # compute the loglikelihood
        state = self.likelihood.loglike_state(state)

        self.debug("Test pipeline:")
        self.debug(state)

        return state


    def calculate_data_covmat(self, parameters):
        # This function computes the hessian of the likelihood 
        # with respect to a certain point in parameter space. This can be understtod as an effective covmat of the data.
        # The hessian is calculated by jax.hessian

        # Run the emulator to compute the observables.
        state = self.emulator.emulate(parameters)

        data_covmats = {}

        # for each observable we create a function that computes the hessian with respect to the observable.
        for observable in state['quantities'].keys():

            observable_values = jnp.array(state['quantities'][observable])

            # function to compute likelihood as a function of the observable
            def hessian_observable(x):
                local_state = copy.deepcopy(state)
                local_state['quantities'][observable] = x
                return -self.likelihood.loglike(local_state)[0]

            # compute hessian of observables
            observable_hessian = jnp.asarray(jax.hessian(hessian_observable)(observable_values))

            # find entries which do not constiute to the loglike
            diag_hessian = jnp.diag(observable_hessian)

            mask = np.zeros(len(observable_values))
            mask[diag_hessian!=0.0]=1

            # set index = indices where mask is 1
            index = np.where(mask==1)[0]
            indices = jnp.meshgrid(index, index)

            # observable_hessian is a n times n matrix. We need to remove the entries which do not contribute to the loglike
            original_shape = observable_hessian.shape

            observable_covmat = jnp.linalg.inv(observable_hessian[mask==1][:,mask==1])

            # create the data_covmat
            data_covmats[observable] = jnp.zeros(original_shape)
            data_covmats[observable] = data_covmats[observable].at[indices[0],indices[1]].set(observable_covmat)

            # check for nans or infs
            if jnp.isnan(data_covmats[observable]).any() or jnp.isinf(data_covmats[observable]).any():
                self.error("Data covariance matrix for observable %s contains nans or infs", observable)
                raise ValueError("Data covariance matrix for observable %s contains nans or infs"%observable)

            # store the diagonal of the data_covmat as txt file in the output directory
            if not os.path.exists(self.hyperparameters['output_directory']+'/data_covmats'):
                os.makedirs(self.hyperparameters['output_directory']+'/data_covmats')
            np.savetxt(self.hyperparameters['output_directory']+'/data_covmats/'+observable+'.txt', jnp.diag(data_covmats[observable]))

        if self.hyperparameters['plotting_directory'] is not None:
            for observable in state['quantities'].keys():
                # check if plotting directory exists, otherwise create it
                if not os.path.exists(self.hyperparameters['plotting_directory']+'/data_covmats'):
                    os.makedirs(self.hyperparameters['plotting_directory']+'/data_covmats')

                data_covmat_plot(data_covmats[observable], 'data covariance matrix '+ observable, self.hyperparameters['plotting_directory']+'/data_covmats/'+observable+'.png')

                covmat_diagonal_plot(data_covmats[observable], 'data diagonal covariance matrix '+ observable, self.hyperparameters['plotting_directory']+'/data_covmats/'+observable+'_diagonal.png')


        return data_covmats     

    # This function computes the loglikelihoods for given parameters.
    # If possible it uses the emulator to speed up the computation.
    # If the emulator is not trained or the emulator is not good enough, the theory code is used.
    def compute_loglike_from_normalized_parameters(self, parameters):
        self.start()

        RNGkey = jax.random.PRNGKey(int(time.clock_gettime_ns(0)))

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)
        
        # every 100 self.n calls, state the n
        if self.n % 100 == 0:
            self.info("Current sampler call: %d", self.n)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        self.debug("Calculating loglike for parameters: %s", state['parameters'])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])

        # compute the observables. First check whether emulator is already trained
        if not self.emulator.trained:

            self.debug("emulator not trained yet -> use theory code")
            state = self.theory.compute(state)
            self.debug("state after theory: %s for parameters: %s", state['quantities'], state['parameters'])
            
            state = self.likelihood.loglike_state(state)
            self.debug("loglike after theory: %f for parameters: %s", state['loglike'][0], state['parameters'])

            state['loglike'] = state['loglike'] + logprior
            self.emulator.add_state(state)
        else:
            # here we need to test the emulator for its performance
            emulator_sample_states, RNGkey = self.emulator.emulate_samples(state['parameters'], RNGkey=RNGkey)
            emulator_sample_loglikes = jnp.array([self.likelihood.loglike_state(_)['loglike'] for _ in emulator_sample_states])
            print("emulator_sample_loglikes: ", emulator_sample_loglikes)
            # check whether the emulator is good enough
            if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=state['parameters']):
                print("Emulator not good enough")
                state = self.theory.compute(state)
                state = self.likelihood.loglike_state(state)
                state['loglike'] = state['loglike'] + logprior
                self.emulator.add_state(state)
            else:
                print("Emulator good enough")
                # Add the point to the quality points
                self.emulator.add_quality_point(state['parameters'])
            
                self.debug("emulator available - check emulation performance")
                state = self.emulator.emulate(state['parameters'])
                self.debug("state after emulator: %s for parameters: %s", state['quantities'], state['parameters'])

                state = self.likelihood.loglike_state(state)
                self.debug("loglike after theory: %f for parameters: %s", state['loglike'][0], state['parameters'])

                state['loglike'] = state['loglike'] + logprior
                self.debug("emulator prediction: %s", state['quantities'])


        # if we have a minimal number of states in the cache, we can train the emulator
        if (len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']) and not self.emulator.trained:
            self.debug("Training emulator")
            self.emulator.train()
            self.debug("Emulator trained")

        self.increment(self.logger)
        return state['loglike']


    def compute_theory_from_normalized_parameters(self, parameters):
        self.start()

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        self.debug("Calculating loglike for parameters: %s", state['parameters'])

        # compute the observables. First check whether emulator is already trained
        if not self.emulator.trained:

            self.debug("emulator not trained yet -> use theory code")
            state = self.theory.compute(state)
            self.debug("state after theory: %s for parameters: %s", state['quantities'], state['parameters'])
        else:
            # here we need to test the emulator for its performance
            emulator_sample_states, RNGkey = self.emulator.emulate_samples(state['parameters'], RNGkey=RNGkey)
            emulator_sample_loglikes = jnp.array([self.likelihood.loglike_state(_)['loglike'] for _ in emulator_sample_states])
            print("emulator_sample_loglikes: ", emulator_sample_loglikes)
            # check whether the emulator is good enough
            if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=state['parameters']):
                print("Emulator not good enough")
                state = self.theory.compute(state)
            else:
                print("Emulator good enough")
                # Add the point to the quality points
                self.emulator.add_quality_point(state['parameters'])
            
                self.debug("emulator available - check emulation performance")
                state = self.emulator.emulate(state['parameters'])
                self.debug("state after emulator: %s for parameters: %s", state['quantities'], state['parameters'])

        # if we have a minimal number of states in the cache, we can train the emulator
        if (len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']) and not self.emulator.trained:
            self.debug("Training emulator")
            self.emulator.train()
            self.debug("Emulator trained")

        self.increment(self.logger)
        return state
    
    @partial(jax.jit, static_argnums=(0))
    def likelihood_function(self, parameter_values, local_state):
        for i, key in enumerate(self.nuisance_parameters):
            local_state['parameters'][key] = jnp.array([parameter_values[i]])*self.nuisance_stds[i] + self.nuisance_means[i]
        local_state = self.likelihood.loglike_state(local_state)
        logprior = self.compute_logprior(local_state)
        local_state['loglike'] = local_state['loglike'] + logprior
        return -local_state['loglike'][0]
    
    

    def nuisance_minimization(self, state):
        
        local_state = copy.deepcopy(state)
        
        # minimize the nuisance parameters
        x0 = (jnp.array([state['parameters'][key][0] for key in self.nuisance_parameters])-self.nuisance_means)/self.nuisance_stds

        state['loglike'] = -self.jit_likelihood_function(x0, local_state)
        # check for nan
        if jnp.isnan(state['loglike']):
            self.error("Nuisance minimization: loglike is nan")
            return state

        result = minimize(self.jit_likelihood_function, x0, jac=self.jit_gradient_likelihood_function, method='TNC', args=(local_state), options={'disp': False}, tol=1e-1)
        self.debug("minimization result: %s", result)

        state['loglike'] = jnp.array([float(-result.fun)])
        for i, key in enumerate(self.nuisance_parameters):
            state['parameters'][key] = jnp.array([result.x[i]])*self.nuisance_stds[i] + self.nuisance_means[i]

        return state

    # This function emulates the loglikelihoods for given parameters.
    def emulate_loglike_from_normalized_parameters_differentiable(self, parameters_norm):
        # this function is similar to the compute_loglike_from_parameters, but it returns the loglike and does not automaticailly add the state to the emulator. Thus it is differentiable.
        # if RNG is not None, we provide the mean estimate, otherwise we sample from the emulator
        self.start()

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters_norm)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])
        
        self.debug("emulator available - check emulation performance")
        state = self.emulator.emulate(state['parameters'])
        self.debug("state after emulator: %s for parameters: %s", state['quantities'], state['parameters'])

        state = self.likelihood.loglike_state(state)
        self.debug("loglike after theory: %f for parameters: %s", state['loglike'][0], state['parameters'])

        state['loglike'] = state['loglike'] + logprior
        self.debug("emulator prediction: %s", state['quantities'])

        parameters = self.transform_parameters_into_normalized_eigenspace(parameters)

        self.increment(self.logger)
        return state['loglike']



    def sample_emulate_loglike_from_normalized_parameters_differentiable(self, parameters, N=1, RNGkey=jax.random.PRNGKey(int(time.time()))):
        # this function is similar to the compute_loglike_from_parameters, but it returns the loglike and does not automaticailly add the state to the emulator. Thus it is differentiable.
        # if RNG is not None, we provide the mean estimate, otherwise we sample from the emulator
        self.start()

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])

        self.debug("emulator available - check emulation performance")

        states, RNGkey = self.emulator.emulate_samples(state['parameters'], RNGkey)

        loglikes = [self.likelihood.loglike_state(_)['loglike'] + logprior for _ in states]

        self.increment(self.logger)
        return loglikes

    def sample_emulate_loglike_from_normalized_parameters_differentiable_noiseFree(self, parameters, N=1, RNGkey=jax.random.PRNGKey(int(time.time()))):
        # this function is similar to the compute_loglike_from_parameters, but it returns the loglike and does not automaticailly add the state to the emulator. Thus it is differentiable.
        # if RNG is not None, we provide the mean estimate, otherwise we sample from the emulator
        self.start()

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])

        self.debug("emulator available - check emulation performance")

        states, RNGkey = self.emulator.emulate_samples_noiseFree(state['parameters'], RNGkey)

        loglikes = [self.likelihood.loglike_state(_)['loglike'] + logprior for _ in states]

        self.increment(self.logger)
        return loglikes

    def compute_total_loglike_from_normalized_parameters(self, parameters):

        res = self.compute_loglike_from_normalized_parameters(parameters)
        return res.sum()
    
    def emulate_total_loglike_from_parameters_differentiable(self, parameters):
        res = self.emulate_loglike_from_normalized_parameters_differentiable(parameters)
        return res.sum()
    
    def sample_emulate_total_loglike_from_parameters_differentiable(self, parameters):
        N = self.emulator.hyperparameters['N_quality_samples']
        res = [_.sum() for _ in self.sample_emulate_loglike_from_normalized_parameters_differentiable(parameters, N=N)]
        return res

    def sample_emulate_total_loglike_from_parameters_differentiable_noiseFree(self, parameters):
        N = self.emulator.hyperparameters['N_quality_samples']
        res = [_.sum() for _ in self.sample_emulate_loglike_from_normalized_parameters_differentiable_noiseFree(parameters, N=N)]
        return res


    def emulate_total_minusloglike_from_parameters_differentiable(self, parameters):
        res = -self.emulate_loglike_from_normalized_parameters_differentiable(parameters)
        return res.sum()
    
    def sample_emulate_total_minusloglike_from_parameters_differentiable(self, parameters):
        N = self.emulator.hyperparameters['N_quality_samples']
        res = [-_.sum() for _ in self.sample_emulate_loglike_from_normalized_parameters_differentiable(parameters, N=N)]
        return res

    def compute_logprior(self, state):
        # To be implemented.
        # if we have a flat prior:
        log_prior = 0.0
        for key, value in self.parameter_dict.items():

            
            # if mean and std are given, we use a gaussian prior
            if 'mean' in value['prior'].keys():
                log_prior += -0.5*(state['parameters'][key][0]-value['prior']['mean'])**2/value['prior']['std']**2 - 0.5*jnp.log(2*jnp.pi*value['prior']['std']**2)
            # else we sticke with the flat prior
            else:
                log_prior+= jnp.log(1.0/(value['prior']['max']-value['prior']['min']))

            log_prior -= jnp.heaviside(value['prior']['min'] - state['parameters'][key][0],  1.0) * 99999999999999999999999.  + jnp.heaviside(state['parameters'][key][0] - value['prior']['max'], 1.0) * 99999999999999999999999.

        return log_prior
    
    def sample(self):
        # Run the sampler.
        
        return 