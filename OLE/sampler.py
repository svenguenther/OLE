# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
# The NUTS sampler is based upon jaxnuts: https://github.com/guillaume-plc/jaxnuts
from .utils.base import BaseClass
from .utils.mpi import *
from .theory import Theory
from .likelihood import Likelihood
from .emulator import Emulator
from .utils.mpi import *

from functools import partial
from typing import Tuple
import jax.numpy as jnp
import jax
import fasteners
from jax.numpy import ndarray
import jax.lax as lax
import random

import numpy as np

import jax.random as random

import time 


class Sampler(BaseClass):

    theory: Theory
    likelihood: Likelihood
    parameter_dict: dict

    emulator: Emulator

    
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, theory, likelihood, parameters, **kwargs):
        self.theory = theory
        self.theory.initialize(**kwargs)
        self.likelihood = likelihood
        self.likelihood.initialize(**kwargs)
        self.parameter_dict = parameters

        


        # remove the parameters from the test state which are not in self.theory.requirements
        # initialize the emulator
        self.emulator = Emulator(**kwargs)

        test_state = None

        if 'load_initial_state' not in kwargs:
            test_state = self.test_pipeline()
        else:
            if not kwargs['load_initial_state']:
                test_state = self.test_pipeline()

        if len(self.theory.requirements) > 0:
            self.emulator.initialize(test_state, input_parameters=self.theory.requirements, **kwargs)
        else:
            self.emulator.initialize(test_state, **kwargs)


        defaulthyperparameters = {
            # output directory for the chain
            'output_directory': 'output',

            # force overwrite
            'force': False,
        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
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


        pass

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

        self.info("Test pipeline:")
        self.info(state)

        return state
    
    def compute_loglike_from_parameters(self, parameters):
        self.start()

        RNGkey = jax.random.PRNGKey(time.clock_gettime_ns(0))

        # rescale the parameters
        parameters = parameters * self.proposal_lengths + self.proposal_means

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
            emulator_sample_states, RNGkey = self.emulator.emulate_samples(state['parameters'], self.emulator.hyperparameters['N_quality_samples'], RNGkey=RNGkey)
            emulator_sample_loglikes = jnp.array([self.likelihood.loglike_state(_)['loglike'] for _ in emulator_sample_states])
            print("emulator_sample_loglikes: ", emulator_sample_loglikes)
            # check whether the emulator is good enough
            if not self.emulator.check_quality_criterium(emulator_sample_loglikes):
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

    def emulate_loglike_from_parameters_differentiable(self, parameters):
        # this function is similar to the compute_loglike_from_parameters, but it returns the loglike and does not automaticailly add the state to the emulator. Thus it is differentiable.
        # if RNG is not None, we provide the mean estimate, otherwise we sample from the emulator
        self.start()

        # rescale the parameters
        parameters = parameters * self.proposal_lengths + self.proposal_means

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

        self.increment(self.logger)
        return state['loglike']



    def sample_emulate_loglike_from_parameters_differentiable(self, parameters, N=1, RNGkey=jax.random.PRNGKey(int(time.time()))):
        # this function is similar to the compute_loglike_from_parameters, but it returns the loglike and does not automaticailly add the state to the emulator. Thus it is differentiable.
        # if RNG is not None, we provide the mean estimate, otherwise we sample from the emulator
        self.start()

        # rescale the parameters
        parameters = parameters * self.proposal_lengths + self.proposal_means

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])
        
        self.debug("emulator available - check emulation performance")
        states, RNGkey = self.emulator.emulate_samples(state['parameters'], N, RNGkey=RNGkey)

        loglikes = [self.likelihood.loglike_state(_)['loglike'] + logprior for _ in states]

        self.increment(self.logger)
        return loglikes




    def compute_total_loglike_from_parameters(self, parameters):
        res = self.compute_loglike_from_parameters(parameters)
        return res.sum()
    
    def emulate_total_loglike_from_parameters_differentiable(self, parameters):
        res = self.emulate_loglike_from_parameters_differentiable(parameters)
        return res.sum()
    
    def sample_emulate_total_loglike_from_parameters_differentiable(self, parameters):
        N = self.emulator.hyperparameters['N_quality_samples']
        res = [_.sum() for _ in self.sample_emulate_loglike_from_parameters_differentiable(parameters, N=N)]
        return res

    def compute_logprior(self, state):
        # To be implemented.
        # if we have a flat prior:
        log_prior = 0.0
        for key, value in self.parameter_dict.items():


            log_prior+= jnp.log(1.0/(value['prior']['max']-value['prior']['min']))

            # if we are outside the prior, return (almost) -inf / Make more beautiful
            # if (state['parameters'][key][0] < value['prior']['min']) or (state['parameters'][key][0] > value['prior']['max']):
            #     return -jnp.inf



            log_prior -= jnp.heaviside(value['prior']['min'] - state['parameters'][key][0],  1.0) * 99999999999999999999999.  + jnp.heaviside(state['parameters'][key][0] - value['prior']['max'], 1.0) * 99999999999999999999999.

        return log_prior
    
    def sample(self):
        # Run the sampler.
        
        return 
    

import emcee
import time 

class EnsembleSampler(Sampler):

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs else 10
        self.ndim = len(self.parameter_dict)

        # initialize the sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.compute_total_loglike_from_parameters)

        pass
    
    
    def run_mcmc(self, nsteps, **kwargs):
        # Run the sampler.
        # Initialize the position of the walkers.            
        pos = jnp.zeros((self.nwalkers, self.ndim))


        initial_seed = int(time.time())+get_mpi_rank()

        for i, key in enumerate(self.parameter_dict.keys()):
            for j in range(self.nwalkers):

                if 'ref' in list(self.parameter_dict[key].keys()):
                    RNG = jax.random.PRNGKey(initial_seed+i*self.nwalkers+j)
                    proposal= self.parameter_dict[key]['ref']['mean'] + self.parameter_dict[key]['ref']['std']*jax.random.normal(RNG)
                    pos = pos.at[j,i].add((proposal-self.proposal_means[i])/self.proposal_lengths[i])
                else:
                    RNG = jax.random.PRNGKey(initial_seed+i*self.nwalkers+j)
                    proposal = self.parameter_dict[key]['prior']['min'] + (self.parameter_dict[key]['prior']['max']-self.parameter_dict[key]['prior']['min'])*jax.random.uniform(RNG)
                    pos = pos.at[j,i].add((proposal-self.proposal_means[i])/self.proposal_lengths[i])

        self.sampler.run_mcmc(pos, nsteps, **kwargs, tune=True)

        # save the chain and the logprobability
        self.chain = self.sampler.get_chain()*self.proposal_lengths + self.proposal_means # rescale chains

        self.logprobability = self.sampler.get_log_prob()

        # Append the chains to chain.txt using a lock
        lock = fasteners.InterProcessLock('chain.txt.lock')
        with lock:
            _ = self.hyperparameters['output_directory'] + '/chain_%d.txt'%get_mpi_rank()
            with open(self.hyperparameters['output_directory'] + '/chain.txt', 'ab') as f:
                np.savetxt(f, np.hstack([self.logprobability.reshape(-1)[:,None], self.chain.reshape((-1,self.ndim))]) )
            with open(_, 'ab') as f:
                np.savetxt(f, np.hstack([self.logprobability.reshape(-1)[:,None], self.chain.reshape((-1,self.ndim))]) )


class NUTSSampler(Sampler):
    # the nuts sampler samples with an Ensemble sampler and once the emulator is trained, it will switch to the NUTS sampler to make use of the gradients.

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs else 10
        self.ndim = len(self.parameter_dict)

        # read target_acceptanc, M_adapt, delta_max
        self.target_acceptance = kwargs['target_acceptance'] if 'target_acceptance' in kwargs else 0.5
        self.M_adapt = kwargs['M_adapt'] if 'M_adapt' in kwargs else 1000
        self.delta_max = kwargs['delta_max'] if 'delta_max' in kwargs else 1000

        self.NUTS_batch_size = kwargs['NUTS_batch_size'] if 'NUTS_batch_size' in kwargs else 10

        pass
    
    
    def run_mcmc(self, nsteps, **kwargs):
        # Run the sampler.
        # Initialize the position of the walkers.
        pos = jnp.zeros((self.nwalkers, self.ndim))

        import time 

        initial_seed = int(time.time()+get_mpi_rank())
        RNG = jax.random.PRNGKey(initial_seed)
        for i, key in enumerate(self.parameter_dict.keys()):
            for j in range(self.nwalkers):
                # not implemented yet!
                if 'ref' in list(self.parameter_dict[key].keys()):
                    RNG, subkey = jax.random.split(RNG)
                    proposal= self.parameter_dict[key]['ref']['mean'] + self.parameter_dict[key]['ref']['std']*jax.random.normal(subkey)
                    pos = pos.at[j,i].add((proposal-self.proposal_means[i])/self.proposal_lengths[i])
                else:
                    RNG, subkey = jax.random.split(RNG)
                    proposal = self.parameter_dict[key]['prior']['min'] + (self.parameter_dict[key]['prior']['max']-self.parameter_dict[key]['prior']['min'])*jax.random.uniform(subkey)
                    pos = pos.at[j,i].add((proposal-self.proposal_means[i])/self.proposal_lengths[i])

        # run the warmup
        current_loglikes = -jnp.inf*jnp.ones(self.nwalkers)
        
        max_loglike = -jnp.inf
        bestfit = None

        if not self.emulator.trained:
            while not self.emulator.trained:
                # We perform vanilla MH sampling for one step
                # The covmat is the identity matrix
                RNG, subkey = jax.random.split(RNG)
                step = pos+jnp.ones((self.nwalkers,self.ndim)) * jax.random.normal(subkey, shape=(self.nwalkers,self.ndim))
                loglikes = jnp.array([self.compute_total_loglike_from_parameters(step[i]) for i in range(self.nwalkers)])
                # update bestfit
                if jnp.max(loglikes) > max_loglike:
                    max_loglike = jnp.max(loglikes)
                    bestfit = step[jnp.argmax(loglikes)]

                # accept or reject
                for i in range(self.nwalkers):
                    if loglikes[i] > current_loglikes[i]:
                        pos = pos.at[i].set(step[i])
                        current_loglikes = current_loglikes.at[i].set(loglikes[i])
                    else:
                        RNG, subkey = jax.random.split(RNG)
                        if jnp.log(jax.random.uniform(subkey)) < loglikes[i]-current_loglikes[i]:
                            pos = pos.at[i].set(step[i])
                            current_loglikes = current_loglikes.at[i].set(loglikes[i])

        else:
            # if the emulator is already trained, we can directly start with the NUTS sampler
            bestfit = pos[0]
        self.info("Emulator trained - Start NUTS sampler now!")

        # create differentiable loglike
        self.logp_and_grad = jax.jit(jax.value_and_grad(self.emulate_total_loglike_from_parameters_differentiable))     # this is the differentiable loglike
        self.logp_sample = jax.jit(self.sample_emulate_total_loglike_from_parameters_differentiable)                    # this samples N realizations from the emulator to estimate the uncertainty

        self.theta0 = bestfit

        # we search a reasonable epsilon (step size)
        RNG, eps = self._findReasonableEpsilon(bestfit, RNG)

        # Initialize variables
        mu = jnp.log(10 * eps)
        eps_bar = 1.0
        H_bar = 0.0
        gamma = .05
        t_0 = 10
        kappa = .75

        # puntjes
        thetas = np.zeros((self.M_adapt + nsteps + 1, self.ndim))
        logps = np.zeros(self.M_adapt + nsteps + 1)

        # number of rounds for the NUTS sampler to run and check the performance of the emulator
        nrounds = int(np.ceil((self.M_adapt + nsteps + 1)/self.NUTS_batch_size))

        thetas[0] = bestfit

        testing_flag = True

        # run the warmup
        for i in range(nrounds):
            for j in range(self.NUTS_batch_size):
                # If all the samples are done, break
                if i*self.NUTS_batch_size+j+1 >= self.M_adapt + nsteps + 1:
                    break

                # Initialize momentum and pick a slice, record the initial log joint probability
                RNG, r, u, logjoint0 = self._init_iteration(thetas[i*self.NUTS_batch_size+j], RNG)

                # Initialize the trajectory
                theta_m, theta_p, r_m, r_p = thetas[i*self.NUTS_batch_size+j], thetas[i*self.NUTS_batch_size+j], r, r
                k = 0 # Trajectory iteration
                n = 1 # Length of the trajectory
                s = 1 # Stop indicator
                thetas[i*self.NUTS_batch_size+j+1] = thetas[i*self.NUTS_batch_size+j] # If everything fails, the new sample is our last position
                start = time.time()
                while s == 1:
                    # Choose a direction
                    RNG, v = self._draw_direction(RNG)
                    
                    # Double the trajectory length in that direction
                    if v == -1:
                        RNG, theta_m, r_m, _, _, theta_f, n_f, s_f, alpha, n_alpha = self._build_tree(theta_m, r_m, u, v, k, eps, logjoint0, RNG)
                    else:
                        RNG, _, _, theta_p, r_p, theta_f, n_f, s_f, alpha, n_alpha = self._build_tree(theta_p, r_p, u, v, k, eps, logjoint0, RNG)

                    # Update theta with probability n_f / n, to effectively sample a point from the trajectory;
                    # Update the current length of the trajectory;
                    # Update the stopping indicator: check it the trajectory is making a U-turn.
                    RNG, thetas[i*self.NUTS_batch_size+j+1], n, s, k = self._trajectory_iteration_update(thetas[i*self.NUTS_batch_size+j+1], n, s, k, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, RNG)

                if i*self.NUTS_batch_size+j+1 <= self.M_adapt:
                    # Dual averaging scheme to adapt the step size 'epsilon'.
                    H_bar, eps_bar, eps = self._adapt(mu, eps_bar, H_bar, t_0, kappa, gamma, alpha, n_alpha, i*self.NUTS_batch_size+j+1)
                elif i*self.NUTS_batch_size+j+1 == self.M_adapt + 1:
                    eps = eps_bar

                print("Round %d/%d, sample %d/%d, time: %f"%(i+1, nrounds, j+1, self.NUTS_batch_size, time.time()-start))


                start = time.time()
                # sample multiple points from the emulator and check the performance
                # note that the runtime of testing is about the same as the runtime of the emulator for one round 
                if testing_flag:
                    # if the emulator is not good enough, we need to run the theory code and add the state to the emulator
                    state = {'parameters': {}, 'quantities': {}, 'loglike': None}

                    # translate the parameters to the state
                    for k, key in enumerate(self.parameter_dict.keys()):
                        state['parameters'][key] = jnp.array([thetas[i*self.NUTS_batch_size+j][k]*self.proposal_lengths[k] + self.proposal_means[k]])

                    if self.emulator.require_quality_check(state['parameters']):
                        # here we need to test the emulator for its performance
                        loglikes = self.logp_sample(thetas[i*self.NUTS_batch_size+j])

                        # check whether the emulator is good enough
                        if not self.emulator.check_quality_criterium(jnp.array(loglikes)):

                            state = self.theory.compute(state)
                            state = self.likelihood.loglike_state(state)
                            logprior = self.compute_logprior(state)
                            state['loglike'] = state['loglike'] + logprior
                            self.emulator.add_state(state)

                            print("Emulator not good enough")

                            # update the differential loglikes
                            self.logp_and_grad = jax.jit(jax.value_and_grad(self.emulate_total_loglike_from_parameters_differentiable))     # this is the differentiable loglike
                            self.logp_sample = jax.jit(self.sample_emulate_total_loglike_from_parameters_differentiable)                    # this samples N realizations from the emulator to estimate the uncertainty
                        else:
                            print("Emulator good enough")
                            # Add the point to the quality points
                            self.emulator.add_quality_point(state['parameters'])

                print("Testing time: ", time.time()-start)
        print(thetas)

        # save the chain and the logprobability
        self.chain = thetas*self.proposal_lengths + self.proposal_means # rescale chains

        # Append the chains to chain.txt using a lock
        lock = fasteners.InterProcessLock('chain.txt.lock')

        self.logprobability = np.zeros(len(thetas))

        with lock:
            _ = self.hyperparameters['output_directory'] + '/chain_%d.txt'%get_mpi_rank()
            with open(self.hyperparameters['output_directory'] + '/chain.txt', 'ab') as f:
                np.savetxt(f, np.hstack([self.logprobability.reshape(-1)[:,None], self.chain.reshape((-1,self.ndim))]) )
            with open(_, 'ab') as f:
                np.savetxt(f, np.hstack([self.logprobability.reshape(-1)[:,None], self.chain.reshape((-1,self.ndim))]) )

    # supportive functions for the sampler
    def _findReasonableEpsilon(self, theta: ndarray, key: ndarray) -> Tuple[ndarray, float]:
        """Heuristic to find a reasonable initial value for the step size.
        
        Finds a reasonable value for the step size by scaling it until the acceptance probability
        crosses 0.5 .

        Parameters
        ----------
        theta : ndarray
            The initial sample position.
        key : ndarray
            PRNG key

        Returns
        -------
        key : ndarray
            The updated PRNG key
        eps : float
            A reasonable initial value for the step size
        """
        eps = 1
        key, subkey = random.split(key)
        r = random.normal(subkey, shape=theta.shape)

        logp, gradlogp = self.logp_and_grad(theta)
        if jnp.isnan(logp):
            raise ValueError("log probability of initial value is NaN.")

        theta_f, r_f, logp, gradlogp = self._leapfrog(theta, r, eps)

        # First make sure that the step is not too large i.e. that we do not get diverging values.
        while jnp.isnan(logp) or jnp.any(jnp.isnan(gradlogp)):
            eps /= 2
            theta_f, r_f, logp, gradlogp = self._leapfrog(theta, r, eps)
        
        # Then decide in which direction to move
        logp0, _ = self.logp_and_grad(theta)
        logjoint0 = logp0 - .5 * jnp.dot(r, r)
        logjoint = logp - .5 * jnp.dot(r_f, r_f)
        a = 2 * (logjoint - logjoint0 > jnp.log(.5)) - 1
        # And successively scale epsilon until the acceptance rate crosses .5
        while a * (logp - .5 * jnp.dot(r_f, r_f) - logjoint0) > a * jnp.log(.5):
            eps *= 2.**a
            theta_f, r_f, logp, _ = self._leapfrog(theta, r, eps)
        return key, eps

    @partial(jax.jit, static_argnums=0)
    def _leapfrog(self, theta: ndarray, r: ndarray, eps: float) -> Tuple[ndarray, ndarray, float, ndarray]:
        """Perform a leapfrog step.
        
        Parameters
        ----------
        theta : ndarray
            Initial sample position.
        r : ndarray
            Initial momentum.
        eps : float
            Step size;

        Returns
        -------
        theta : ndarray
            New sample position
        r : ndarray
            New momentum
        logp : float
            Log probability of the new position.
        gradlogp : ndarray
            Gradient of the log probability evaluated
            at the new position.
        """
        logp, gradlogp = self.logp_and_grad(theta)
        r = r + .5 * eps * gradlogp
        theta = theta + eps * r
        logp, gradlogp = self.logp_and_grad(theta)
        r = r + .5 * eps * gradlogp
        return theta, r, logp, gradlogp

    @partial(jax.jit, static_argnums=0)
    def _init_iteration(self, theta: ndarray, key: ndarray) -> Tuple[ndarray, ndarray, float, float]:
        """Initialize the sampling iteration

        Parameters
        ----------
        theta : ndarray
            The previous sample
        key : ndarray
            The PRNG key.

        Returns
        -------
        key : ndarray
            The updated PRNG key.
        r : ndarray
            The initial momentum vector.
        u : float
            The slice for this iteration.
        logjoint : float
            The logarithm of the joint probability p(theta, r)
        """
        key, *subkeys = random.split(key, 3)
        r = random.normal(subkeys[0], shape=self.theta0.shape)
        logprob, _ = self.logp_and_grad(theta)
        logjoint = logprob - .5 * jnp.dot(r, r)
        u = random.uniform(subkeys[1]) * jnp.exp(logjoint)
        return key, r, u, logjoint

    @partial(jax.jit, static_argnums=0)
    def _draw_direction(self, key: ndarray) -> Tuple[ndarray, int]:
        """Draw a random direction (-1 or 1)"""
        key, subkey = random.split(key)
        v = 2 * random.bernoulli(subkey) - 1
        return key, v

    @partial(jax.jit, static_argnums=0)
    def _trajectory_iteration_update(self, theta: ndarray, n: int, s: int, j: int, 
                                     theta_m: ndarray, r_m: ndarray, theta_p: ndarray, r_p: ndarray, 
                                     theta_f: ndarray, n_f: int, s_f: int, key: int) -> Tuple[ndarray, ndarray, int, int, int]:
        """Update trajectory parameters

        Parameters
        ----------
        theta : ndarray
            The previous sample.
        n : int
            Previous length of the trajectory.
        s : int
            Previous stopping indicator. 
        j : int
            Trajectory iteration index
        theta_m : ndarray
            Trajectory tail.
        r_m : ndarray
            Tail momentum.
        theta_p : ndarray
            Trajectory head.
        r_p : ndarray
            Head momentum
        theta_f : ndarray
            Sample from the last trajectory sub-tree.
        n_f : int
            Size of the last trajectory sub-tree
        s_f : int
            Stopping indicator of the last trajectory sub-tree
        key : ndarray
            PRNG key.

        Returns
        -------
        key : ndarray
            Updated PRNG key.
        theta : ndarray
            Updated sample
        n : int
            Updated trajectory size.
        s : int
            Updated stopping indicator.
        j : int
            Updated iteration index.
        """
        # If we are not stopping here, update theta with probability n_f / n
        operand = key, theta, theta_f, n, n_f
        key, theta = lax.cond(s_f == 1, self._draw_theta , lambda op: op[:2], operand)
        # Update the trajectory length
        n += n_f
        # Check if we are making a U-turn
        s = s_f * (jnp.dot(theta_p - theta_m, r_m) >= 0) * (jnp.dot(theta_p - theta_m, r_p) >= 0)
        # Update iteration index
        j += 1
        return key, theta, n, s, j

    @partial(jax.jit, static_argnums=0)
    def _draw_theta(self, operand: Tuple[ndarray, ndarray, ndarray, int, int]) -> Tuple[ndarray, ndarray]:
        """Replace the last sample with the new one with probability n_f / n.
        
        Parameters
        ----------
        operand : Tuple[ndarray, ndarray, ndarray, int, int]
            A tuple containing the PRNG key, the old sample, the new one, the previous total
            length of the trajectory and the length of the new sub-tree.

        Returns
        -------
        result : Tuple[ndarray, ndarray]
            A tuple containing the updated PRNG key and the chosen sample.
        """
        key, theta, theta_f, n, n_f = operand
        key, subkey = random.split(key)
        return lax.cond(random.uniform(subkey) < lax.min(1., n_f.astype(float) / n),
               lambda op: (op[0], op[2]), lambda op: op[:2], (key, theta, theta_f, n, n_f))

    @partial(jax.jit, static_argnums=0)
    def _adapt(self, mu: float, eps_bar: float, H_bar: float, t_0: float, 
               kappa: float, gamma: float, alpha: float, n_alpha: int, m: int) -> Tuple[float, float, float]:
        """Update the step size according to the dual averaging scheme.
        
        Parameters
        ----------
        mu : float
            Value towards which the iterates (epsilon) are shrinked.
        eps_bar : float
            Averaged iterate of the dual-averaging scheme.
        eps : float
            Iterate of the dual-averaging scheme.
        H_bar : float
            Averaged difference of the current pseudo-acceptance rate to the desired one.
        t_0 : float
            Free parameter to stabilize the initial iterations.
        kappa : float
            Power of the step size schedule.
        gamma : float
            Free parameter that controls the amount of shrinkage towards mu.
        alpha : float
            Pseudo-acceptance probability of the last trajectory.
        n_alpha : float
            Size of the last trajectory.
        m : int
            Iteration index

        Returns
        -------
        H_bar : float
            Updated averaged difference of the current pseudo-acceptance rate to the desired one.
        eps_bar : float
            Updated averaged iterate.
        eps : float
            Updated iterate.
        """
        eta = 1 / (m + t_0)
        H_bar = (1 - eta) * H_bar + eta * (self.target_acceptance - alpha / n_alpha)
        mkappa = m**(-kappa)
        eps = jnp.exp(mu - (jnp.sqrt(m) / gamma) * H_bar)
        eps_bar = jnp.exp(mkappa * jnp.log(eps) + (1 - mkappa) * jnp.log(eps_bar))
        return H_bar, eps_bar, eps

    def _build_tree(self, theta: ndarray, r: ndarray, u: float, v: int, j: int, 
                   eps: float, logjoint0: ndarray, key: ndarray) -> Tuple[ndarray, 
                   ndarray, ndarray, ndarray, ndarray, ndarray, int, int, float, int]:
        """Recursively build the trajectory binary tree.
        
        Parameters
        ----------
        theta : ndarray
            Sample position.
        r : ndarray
            Sample momentum.
        u : float
            Slice position.
        v : int
            Direction to take.
        j : int
            Iteration index of the trajectory.
        eps : float
            Step size.
        logjoint0 : ndarray
            Logarithm of the joint probability p(theta, r) of the
            original sample.
        key : ndarray
            PRNG key.
        
        Returns
        -------
        key : ndarray
            Updated PRNG key
        theta_m : ndarray
            Tail position
        r_m : ndarray
            Tail momentum
        theta_p : ndarray
            Head position
        r_p : ndarray
            Head momentum
        theta_f : ndarray
            Sampled position
        n_f : int
            Slice set size.
        s_f : int
            Stop indicator.
        alpha_f : float
            Pseudo acceptance rate.
        n_alpha_f : int
            Total set size.
        """
        if j == 0: # Initialize the tree
            return self._init_build_tree(theta, r, u, v, j, eps, logjoint0, key)
        else: # Recurse
            key, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, alpha_f, n_alpha_f = self._build_tree(theta, r, u, v, j - 1, eps, logjoint0, key)
            if s_f == 1: # If no early stopping, recurse.
                if v == -1:
                    key, theta_m, r_m, _, _, theta_ff, n_ff, s_ff, alpha_ff, n_alpha_ff = self._build_tree(theta_m, r_m, u, v, j - 1, eps, logjoint0, key)
                else:
                    key, _, _, theta_p, r_p, theta_ff, n_ff, s_ff, alpha_ff, n_alpha_ff = self._build_tree(theta_p, r_p, u, v, j - 1, eps, logjoint0, key)
                
                # Update theta with probability n_ff / (n_f + n_ff);
                # Update the stopping indicator (U-turn);
                # Update the pseudo-acceptance;
                # Update the length counters;
                key, theta_f, n_f, s_f, alpha_f, n_alpha_f = self._update_build_tree(theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, alpha_f, n_alpha_f, theta_ff, n_ff, s_ff, alpha_ff, n_alpha_ff, key)
            return key, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, alpha_f, n_alpha_f

    @partial(jax.jit, static_argnums=0)
    def _init_build_tree(self, theta : ndarray, r : ndarray, u : float, v : int, j : int, 
                         eps : float, logjoint0 : float, key : ndarray) -> Tuple[ndarray, 
                         ndarray, ndarray, ndarray, ndarray, ndarray, int, int, float, int]:
        """Initialize the trajectory binary tree."""
        # Perform one leapfrog step
        theta, r, logp, _ = self._leapfrog(theta, r, v * eps)
        logjoint = logp - .5 * jnp.dot(r, r)
        # Check if the step is within the slice
        n_f = (jnp.log(u) <= logjoint).astype(int)
        # Check that we are not completely off-track
        s_f = (jnp.log(u) < logjoint + self.delta_max).astype(int)
        # Compute the acceptance rate
        prob_ratio = jnp.exp(logjoint - logjoint0)
        alpha_f = lax.cond(jnp.isnan(prob_ratio), lambda _: 0., lambda _: lax.min(prob_ratio, 1.), None) # Presumably if the probability ratio diverges,
                                                                                                         # it is because the log-joint probability diverges, i.e. the probability tends
                                                                                                         # to zero, so its log is -infinity. Then the acceptance rate of this step,
                                                                                                         # which is what 'alpha_f' stands for, should be zero.
        # Total set size
        n_alpha_f = 1
        return key, theta, r, theta, r, theta, n_f, s_f, alpha_f, n_alpha_f

    @partial(jax.jit, static_argnums=0)
    def _update_build_tree(self, theta_m: ndarray, r_m: ndarray, theta_p: ndarray, r_p: ndarray, 
                           theta_f: ndarray, n_f: int, s_f: int, alpha_f: float, n_alpha_f: int, 
                           theta_ff: ndarray, n_ff: int, s_ff: int, alpha_ff: float, n_alpha_ff: int, 
                           key: ndarray) -> Tuple[ndarray, ndarray, int, int, float, int]:   
        """Updates the tree parameters.
        
        Parameters
        ----------
        theta_m : ndarray
            Tail position
        r_m : ndarray
            Tail momentum
        theta_p : ndarray
            Head position
        r_p : ndarray
            Head momentum
        theta_f : ndarray
            First sampled position
        n_f : int
            First slice set size.
        s_f : int
            First stop indicator.
        alpha_f : float
            First pseudo acceptance rate.
        n_alpha_f : int
            First total set size.
        theta_ff : ndarray
            Second sampled position
        n_ff : int
            Second slice set size.
        s_ff : int
            Second stop indicator.
        alpha_ff : float
            Second pseudo acceptance rate.
        n_alpha_ff : int
            Second total set size.

        Returns
        -------
        key : ndarray
            Updated PRNG key
        theta_f : ndarray
            Sampled position
        n_f : int
            Slice set size.
        s_f : int
            Stop indicator.
        alpha_f : float
            Pseudo acceptance rate.
        n_alpha_f : int
            Total set size.
        """
        key, subkey = random.split(key)
        update = random.uniform(subkey)
        theta_f = lax.cond(update <= n_ff / lax.max(n_f + n_ff, 1), lambda _: theta_ff, lambda _: theta_f, None)
        alpha_f += alpha_ff
        n_alpha_f += n_alpha_ff
        s_f = s_ff * (jnp.dot(theta_p - theta_m, r_m) >= 0).astype(int) * (jnp.dot(theta_p - theta_m, r_p) >= 0).astype(int)
        n_f += n_ff
        return key, theta_f, n_f, s_f, alpha_f, n_alpha_f

# This class can take one or multiple input parameters and evaluate the likelihood for them. Additionally we can turn the emulator on and off.
class EvaluateSampler(Sampler):
    
        def __init__(self, name=None, **kwargs):
            super().__init__(name, **kwargs)
        
        def initialize(self, **kwargs):
            super().initialize(**kwargs)
    
            # flag whether to use the emulator or not
            self.use_emulator = kwargs['use_emulator'] if 'use_emulator' in kwargs else True

            # flag wether to give also the uncertainty estimate
            self.return_uncertainty = kwargs['return_uncertainty'] if 'return_uncertainty' in kwargs else False

            # check wether we want to calculate the loglikelihood or the logposterior
            self.logposterior = kwargs['logposterior'] if 'logposterior' in kwargs else False

            # number of samples to estimate the uncertainty
            self.nsamples = kwargs['nsamples'] if 'nsamples' in kwargs else 20

            # run evaluation once to train the emulator
            if self.use_emulator:
                self.info("Training emulator")
                parameter_list = jnp.array([self.parameter_dict[key]['ref']['mean'] for key in self.parameter_dict.keys()])
                self.compute_total_loglike_from_parameters(parameter_list/self.proposal_lengths)
                self.info("Emulator trained")

            # output error if we use the emulator but it is not trained
            if self.use_emulator and not self.emulator.trained:
                self.error("Emulator is not trained yet. Please train the emulator first or set use_emulator=False")
                raise ValueError
        
            pass
        
        
        def sample(self, input_dicts, **kwargs):
            # Run the sampler.
            # Initialize the position of the walkers.
            RNGkey = jax.random.PRNGKey(random.randint(0, 10000000))

            # check whether input_dicts is a list or a single dictionary
            if isinstance(input_dicts, list):
                pass
            else:
                input_dicts = [input_dicts]

            # Number of evaluations
            n = len(input_dicts)

            # initialize the output
            output = jnp.zeros(n)

            # initialize the output uncertainty
            output_uncertainty = jnp.zeros(n)



            # loop over the input_dicts
            for i, input_dict in enumerate(input_dicts):
                # translate the parameters to the state
                state = {'parameters': {}, 'quantities': {}, 'loglike': None}
                for key, value in input_dict.items():
                    state['parameters'][key] = jnp.array(value)

                # compute logprior
                if self.logposterior:
                    logprior = self.compute_logprior(state)
                else:
                    logprior = 0.0

                # compute the observables. First check whether emulator is already trained
                if not self.emulator.trained or not self.use_emulator:
                    state = self.theory.compute(state)
                    state = self.likelihood.loglike_state(state)
                    state['loglike'] = state['loglike'] + logprior
                else:
                    state = self.emulator.emulate(state['parameters'])
                    state = self.likelihood.loglike_state(state)
                    state['loglike'] = state['loglike'] + logprior

                # save the output
                output = output.at[i].set(state['loglike'][0])

                # save the uncertainty
                if self.return_uncertainty:
                    # initialize the output
                    output_samples = jnp.zeros(self.nsamples)

                    # loop over the nsamples
                    for j in range(self.nsamples):
                        # compute the observables. First check whether emulator is already trained
                        if not self.emulator.trained or not self.use_emulator:
                            state = self.theory.compute(state)
                            state = self.likelihood.loglike_state(state)
                            state['loglike'] = state['loglike'] + logprior
                        else:
                            state, RNGkey = self.emulator.emulate_samples(state['parameters'],1,RNGkey=RNGkey)[0]
                            state = self.likelihood.loglike_state(state)
                            state['loglike'] = state['loglike'] + logprior

                        # save the output
                        output_samples = output_samples.at[j].set(state['loglike'][0])

                    # save the uncertainty
                    output_uncertainty = output_uncertainty.at[i].set(jnp.std(output_samples))

            # return the output
            return output, output_uncertainty










