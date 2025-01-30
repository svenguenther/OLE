# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
# The NUTS sampler is based upon jaxnuts: https://github.com/guillaume-plc/jaxnuts
from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.theory import Theory
from OLE.likelihood import Likelihood
from OLE.emulator import Emulator
from OLE.utils.mpi import *
from OLE.sampler.base import Sampler
from OLE.sampler.minimize import MinimizeSampler
from scipy.optimize import minimize

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

import copy

import time 


class NUTSSampler(Sampler):
    # the nuts sampler samples with an Ensemble sampler and once the emulator is trained, it will switch to the NUTS sampler to make use of the gradients.

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, **kwargs):
        emulator_settings = kwargs['emulator_settings'] if 'emulator_settings' in kwargs else {}
        emulator_settings['testing_strategy'] = 'test_all'
        # update kwargs
        kwargs['emulator_settings'] = emulator_settings
        super().initialize(**kwargs)

        # read the hyperparameters and add some NUTS specific hyperparameters
            
        self.hyperparameters['nwalkers'] = 10 if 'nwalkers' not in self.hyperparameters.keys() else self.hyperparameters['nwalkers']
        self.nwalkers = self.hyperparameters['nwalkers']
        self.ndim = len(self.parameter_dict)

        # read target_acceptanc, M_adapt, delta_max
        self.hyperparameters['target_acceptance'] = 0.5 if 'target_acceptance' not in self.hyperparameters else self.hyperparameters['target_acceptance']
        self.target_acceptance = self.hyperparameters['target_acceptance']
        self.hyperparameters['M_adapt'] = 200 if 'M_adapt' not in self.hyperparameters else self.hyperparameters['M_adapt']
        self.M_adapt = self.hyperparameters['M_adapt']
        self.hyperparameters['delta_max'] = 1000 if 'delta_max' not in self.hyperparameters else self.hyperparameters['delta_max']
        self.delta_max = self.hyperparameters['delta_max']

        # minimize nuisance parameters
        self.hyperparameters['minimize_nuisance_parameters'] = True if 'minimize_nuisance_parameters' not in self.hyperparameters else self.hyperparameters['minimize_nuisance_parameters']
        self.minimize_nuisance_parameters = self.hyperparameters['minimize_nuisance_parameters']

        pass
    
    def _check_correct_bestfit(self, bestfit):
        # this function checks whether the bestfit is on the edge of the prior. If it is, it will shift it slightly inside the prior.
        # enumerate over the parameters
        for i, key in enumerate(self.parameter_dict.keys()):
            eps = 1e-1
            if bestfit[i] <= self.parameter_dict[key]['prior']['min']+eps*self.parameter_dict[key]['proposal']:
                # set warning
                self.warning("Bestfit parameter %s is on the lower edge of the prior (%f). Shifting it slightly inside the prior."%(key,self.parameter_dict[key]['prior']['min']))
                bestfit = bestfit.at[i].set(self.parameter_dict[key]['prior']['min'] + self.parameter_dict[key]['proposal']*eps)
            if bestfit[i] >= self.parameter_dict[key]['prior']['max']-eps*self.parameter_dict[key]['proposal']:
                # set warning
                self.warning("Bestfit parameter %s is on the upper edge of the prior (%f). Shifting it slightly inside the prior."%(key,self.parameter_dict[key]['prior']['max']))
                bestfit = bestfit.at[i].set(self.parameter_dict[key]['prior']['max'] - self.parameter_dict[key]['proposal']*eps)

        return bestfit

    def nuisance_minimization(self, state):
        """
        This function minimizes the nuisance parameters for a given state. 
        It uses the minimize function from scipy.optimize to minimize the logposterior.
        Additionally, it utilizes the gradient of the logposterior to speed up the minimization.
        It uses the logposterior_function to compute the logposteriors for given parameters and a given theory state.

        Parameters
        --------------
        state : dict
            The state of the theory code.

        Returns
        --------------
        dict :
            The state after the minimization of the nuisance parameters
        """
        
        local_state = copy.deepcopy(state)
        
        # minimize the nuisance parameters
        x0 = (jnp.array([state['parameters'][key][0] for key in self.nuisance_parameters])-self.nuisance_means)/self.nuisance_stds

        state['total_loglike'] = jnp.array([-self.jit_logposterior_function(x0, local_state)])
        # check for nan
        if jnp.isnan(state['total_loglike']):
            self.error("Nuisance minimization: total_loglike is nan")
            return state

        result = minimize(self.jit_logposterior_function, x0, jac=self.jit_gradient_logposterior_function, method='TNC', args=(local_state), options={'disp': False}, tol=1e-1)
        self.debug("minimization result: %s", result)

        state['total_loglike'] = jnp.array([jnp.array(float(-result.fun))])
        for i, key in enumerate(self.nuisance_parameters):
            state['parameters'][key] = jnp.array([result.x[i]])*self.nuisance_stds[i] + self.nuisance_means[i]

        return state

    
    def run_mcmc(self, nsteps, **kwargs):
        # Run the sampler.

        # Ensure Number of steps requested is an integer
        nsteps = int(nsteps)

        # Initialize the position of the walkers.

        self.logger.info("Starting NUTS")

        pos = jnp.zeros((self.nwalkers, self.ndim))

        self.write_to_log("Starting NUTS \n")

        import time 

        initial_seed = int(time.time()+get_mpi_rank())
        RNG = jax.random.PRNGKey(initial_seed)

        pos = jax.random.normal(RNG, (self.nwalkers, self.ndim))
        step = jnp.zeros((self.nwalkers, self.ndim))

        # run the warmup
        current_loglikes = -jnp.inf*jnp.ones(self.nwalkers)
        
        max_loglike = -jnp.inf
        bestfit = None

        if not self.debug_mode:
            self.jit_logposterior_function = jax.jit(self.logposterior_function)
            self.jit_gradient_logposterior_function = jax.jit(jax.grad(self.logposterior_function))
        else:
            self.jit_logposterior_function = self.logposterior_function
            self.jit_gradient_logposterior_function = jax.grad(self.logposterior_function)

        if not self.emulator.trained:
            self.write_to_log("NUTS: Initial Metropolis-Hastings \n")
            while not self.emulator.trained:
                # We perform vanilla MH sampling for one step
                # The covmat is the identity matrix

                for i in range(self.nwalkers):

                    proposal_found = False 
                    counter = 0
                    while not proposal_found:
                        RNG, subkey = jax.random.split(RNG)
                        step = step.at[i].set(pos[i]+jnp.ones(self.ndim) * jax.random.normal(subkey, shape=(self.ndim,)))

                        step_untransformed = self.retranform_parameters_from_normalized_eigenspace(step[i] )

                        # Run the sampler.
                        state_test = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None} 
                        # translate the parameters to the state
                        for j, key in enumerate(self.parameter_dict.keys()):
                            state_test['parameters'][key] = jnp.array([step_untransformed[j]])

                        # compute the prior
                        logprior = self.compute_logprior(state_test)

                        if logprior > -1e20:
                            proposal_found = True
                        else:
                            counter += 1
                            if counter > 1000:
                                self.error("Could not find a proposal in 1000 trials. Check the prior boundaries.")
                                raise ValueError


                # if we have a large number of nuisance parameters, this phase can take a lot of time if no covmat is given. 
                # Here we provide the user with the possibility to minimize the nuisance parameters for each theory step.
                if (self.minimize_nuisance_parameters) and (len(self.nuisance_parameters)>0):
                    theory_states = [self.compute_theory_from_normalized_parameters(step[i]) for i in range(self.nwalkers)]

                    # minimize the nuisance parameters. It takes the state with the calculated theory and minimizes the nuisance parameters
                    for i in range(self.nwalkers):
                        theory_states[i] = self.nuisance_minimization(theory_states[i])
                        self.emulator.add_state(theory_states[i])

                        if (len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']) and not self.emulator.trained:
                            self.debug("Training emulator")
                            self.emulator.train()
                            self.debug("Emulator trained")

                    loglikes = jnp.array([theory_states[i]['total_loglike'][0] for i in range(self.nwalkers)])

                else:
                    loglikes = jnp.array([self.compute_total_logposterior_from_normalized_parameters(step[i]) for i in range(self.nwalkers)])

                # update bestfit
                if jnp.max(loglikes) > max_loglike:
                    
                    max_loglike = jnp.max(loglikes)
                    bestfit = self.retranform_parameters_from_normalized_eigenspace(step[jnp.argmax(loglikes)])
                
                # accept or reject
                for i in range(self.nwalkers):
                    if loglikes[i] > current_loglikes[i]:
                        pos = pos.at[i].set(step[i])
                        current_loglikes = current_loglikes.at[i].set(loglikes[i])
                    else:
                        RNG, subkey = jax.random.split(RNG)
                        # Metroplis-Hastings acceptance
                        if jnp.log(jax.random.uniform(subkey)) < loglikes[i]-current_loglikes[i]:
                            pos = pos.at[i].set(step[i])
                            current_loglikes = current_loglikes.at[i].set(loglikes[i])

        else:
            # if the emulator is already trained, we can directly start with the NUTS sampler
            bestfit = self.retranform_parameters_from_normalized_eigenspace(pos[0])

        # do minimization here to get bestfit and covariance matrix (fisher matrix)
        self.info("Starting minimization after first emulator training to find bestfit and covmat.")
        minimizer_params = {**{'method': 'TNC', 'use_emulator': True, 'use_gradients': True, 'logposterior': True}, **self.hyperparameters}
        minimizer = MinimizeSampler()

        minimizer.initialize(parameters=self.parameter_dict, 
                            theory=self.theory, 
                            likelihood_collection=self.likelihood_collection, 
                            emulator=self.emulator, 
                            likelihood_collection_settings=self.likelihood_collection_settings,
                            theory_settings=self.theory_settings,
                            emulator_settings=self.emulator.hyperparameters,
                            sampling_settings=minimizer_params,
                            **self.kwargs)
        minimizer.minimize()

        bestfit = minimizer.bestfit


        # update the covmat
        if minimizer.res.success:
            # check if there are any NaNs in the covmat
            if jnp.any(jnp.isnan(minimizer.inv_hessian)):
                self.info("Minimization failed. Nans in covmat. Not updating covmat.")
            # check if any of the diagonal elements are zero or negative
            elif jnp.any(jnp.diag(minimizer.inv_hessian)<=0):
                self.info("Minimization failed. Eigenvalues are <=0. Not updating covmat.")
            else:
                self.info("Minimization after first emulator training finds:")
                self.info("Bestfit: ")
                # make a string with all parameter keys and their bestfit values
                _ = " ".join([f"{key}: {bestfit[i]}" for i, key in enumerate(self.parameter_dict.keys())])
                self.info(_)
                self.debug("Covmat: ")
                self.debug(minimizer.inv_hessian)
                self.info("Updating covmat with Fisher matrix")
                self.update_covmat(minimizer.inv_hessian)
        else:
            self.info("Minimization failed. Not updating covmat.")
            
        # Ensure that the bestfit is not on the edge of the prior (or outside). It it is we shift it slightly inside the prior for better initial sampling!
        bestfit_shift = self._check_correct_bestfit(bestfit) 
            
        # Run the sampler.
        if self.hyperparameters['compute_data_covmat']:
            # compute the data covariance matrix
            bestfit_state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

            # translate the parameters to the state
            for i, key in enumerate(self.parameter_dict.keys()):
                bestfit_state['parameters'][key] = jnp.array([bestfit[i]])

            data_covmats = self.calculate_data_covmat(bestfit_state['parameters'])

            # set the data covmats in the emulator
            self.emulator.set_data_covmats(data_covmats)

            # we need to retrain the emulator 
            self.emulator.train()

        self.info("Emulator trained - Start NUTS sampler now!")

        self.write_to_log("NUTS: Find stepsize \n")

        # create differentiable loglike
        self.logp_and_grad = jax.jit(jax.value_and_grad(self.emulate_total_logposterior_from_normalized_parameters_differentiable))     # this is the differentiable loglike
        
        if not self.emulator.likelihood_collection_differentiable: # if we do not have differentaibel likelihoods, we need to sample from the emulator
            self.logp_sample = self.sample_emulate_total_logposterior_from_normalized_parameters 

        self.theta0 = self.transform_parameters_into_normalized_eigenspace(bestfit_shift)

        # we search a reasonable epsilon (step size)
        self.info("Searching for a reasonable step sizes")

        # If we are not in debug mode, we jit the functions
        if not self.debug_mode:
            self._leapfrog = jax.jit(self._leapfrog)
            self._init_build_tree = jax.jit(self._init_build_tree)
            self._update_build_tree = jax.jit(self._update_build_tree)
            self._trajectory_iteration_update = jax.jit(self._trajectory_iteration_update)
            self._adapt = jax.jit(self._adapt)
            self._draw_direction = jax.jit(self._draw_direction)
            self._init_iteration = jax.jit(self._init_iteration)
            self._draw_theta = jax.jit(self._draw_theta)

        RNG, eps = self._findReasonableEpsilon(self.theta0, RNG)
        self.info("Found reasonable step sizes: %f", eps)

        # Initialize variables
        mu = jnp.log(10 * eps)
        eps_bar = 1.0
        H_bar = 0.0
        gamma = .05
        t_0 = 10
        kappa = .75

        # puntjes
        thetas = np.zeros((self.M_adapt + nsteps + 1, self.ndim))

        thetas[0] = self.transform_parameters_into_normalized_eigenspace(bestfit_shift)
        testing_flag = True

        self.write_to_log("NUTS: Start Warm-up \n")

        # run the warmup
        for i in range(self.M_adapt + nsteps):
            # Initialize momentum and pick a slice, record the initial log joint probability
            if i == self.M_adapt-1:
                self.write_to_log("NUTS: Start Sampling \n")

            valid_point_flag = False # we need to check whether the point is inside the prior
            while not valid_point_flag:
            
                RNG, r, u, logjoint0, loglike0 = self._init_iteration(thetas[i], RNG)

                # Initialize the trajectory
                theta_m, theta_p, r_m, r_p = thetas[i], thetas[i], r, r
                k = 0 # Trajectory iteration
                n = 1 # Length of the trajectory
                s = 1 # Stop indicator
                thetas[i+1] = thetas[i] # If everything fails, the new sample is our last position
                start = time.time()

                self.emulator.start('NUTS')
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
                    RNG, thetas[i+1], n, s, k = self._trajectory_iteration_update(thetas[i+1], n, s, k, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, RNG)

                if i+1 <= self.M_adapt:
                    # Dual averaging scheme to adapt the step size 'epsilon'.
                    H_bar, eps_bar, eps = self._adapt(mu, eps_bar, H_bar, t_0, kappa, gamma, alpha, n_alpha, i+1)
                elif i+1 == self.M_adapt + 1:
                    eps = eps_bar

                self.emulator.increment('NUTS')

                # Check if the new sample is inside the prior
                parameters_untransformed = {'parameters': {key: jnp.array([self.retranform_parameters_from_normalized_eigenspace(thetas[i+1])[j]]) for j, key in enumerate(self.parameter_dict.keys())}}
                check_logprior = self.compute_logprior(parameters_untransformed)
                if check_logprior > -1e20:
                    valid_point_flag = True
                else:
                    self.warning("Sample %d/%d is outside the prior. Sampling again."%(i+1, self.M_adapt + nsteps +1))



            self.debug("Sample %d/%d, time: %f"%(i+1, self.M_adapt + nsteps +1, time.time()-start))



            start = time.time()
            # sample multiple points from the emulator and check the performance
            # note that the runtime of testing is about the same as the runtime of the emulator for one round 
            if testing_flag:
                # if the emulator is not good enough, we need to run the theory code and add the state to the emulator
                state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

                theta_untransformed = self.retranform_parameters_from_normalized_eigenspace(thetas[i+1])

                # translate the parameters to the state
                for k, key in enumerate(self.parameter_dict.keys()):
                    state['parameters'][key] = jnp.array([theta_untransformed[k]])

                if self.emulator.require_quality_check(state['parameters']) and (thetas[i+1]!=thetas[i]).all():
                    # here we need to test the emulator for its performance
                    # noiseFree is only required if a noise term is used at all !!
                    self.emulator.start("likelihood_testing")
                    if not self.emulator.likelihood_collection_differentiable:
                        loglikes = self.logp_sample(thetas[i+1])

                        self.debug('dumping loglikes emulated')
                        # self.debug(jnp.std(jnp.array(loglikes)))
                        self.debug(jnp.std(jnp.array(loglikes)))

                    else:
                        loglikes = jnp.array([self.compute_loglike_uncertainty_for_differentiable_likelihood_from_normalized_parameters(thetas[i+1])])
                    self.emulator.increment("likelihood_testing")

                    # remove this after enough testing
                    #for quantity, emulatorX in self.emulator.emulators.items():
                      
                    #    input_data = jnp.array([[value[0] for key, value in state['parameters'].items() if key in self.emulator.input_parameters]])
                    #    parameters_normalized = self.emulator.emulators[quantity].data_processor.normalize_input_data(input_data)
                    #    for i in range(emulatorX.num_GPs):
                    #        emulatorX.GPs[i].predicttest(parameters_normalized)

                    # we need to compute the logposterior for checking the quality of the emulator
                    self.emulator.start('emulate')
                    reference_loglike, _ = self.logp_and_grad(thetas[i+1])
                    self.emulator.increment('emulate')
                    self.emulator.emulate_counter += 1
                    self.emulator.print_status()


                    # check whether the emulator is good enough
                    if not self.emulator.check_quality_criterium(jnp.array(loglikes), reference_loglike=reference_loglike, parameters=state['parameters']):
                        
                        self.emulator.start('theory_code')
                        state = self.theory.compute(state)
                        self.emulator.increment('theory_code')

                        self.emulator.start('likelihood')
                        for key in self.likelihood_collection.keys():
                            state = self.likelihood_collection[key].loglike_state(state)
                        logprior = self.compute_logprior(state)
                        self.emulator.increment('likelihood')

                        state['total_loglike'] = jnp.array([jnp.array(list(state['loglike'].values())).sum() + logprior])
                        a,rejit_required = self.emulator.add_state(state)

                        self.debug("Emulator not good enough")

                        # update the differential loglikes
                        if rejit_required:
                            self.logp_and_grad = jax.jit(jax.value_and_grad(self.emulate_total_logposterior_from_normalized_parameters_differentiable))     # this is the differentiable loglike
                            if not self.emulator.likelihood_collection_differentiable: # if we do not have differentaibel likelihoods, we need to sample from the emulator
                                self.logp_sample = self.sample_emulate_total_logposterior_from_normalized_parameters                  # this samples N realizations from the emulator to estimate the uncertainty
                            
                            # self._leapfrog = jax.jit(self._leapfrog)
                            self._init_build_tree = jax.jit(self._init_build_tree)
                            self._update_build_tree = jax.jit(self._update_build_tree)
                            self._init_iteration = jax.jit(self._init_iteration)
                            self._draw_theta = jax.jit(self._draw_theta)
                    else:
                        
                        # if self.emulator.hyperparameters['test_noise_levels_counter'] > 0 and self.emulator.hyperparameters['white_noise_ratio'] != 0.:
                        #     self.emulator.hyperparameters['test_noise_levels_counter'] -= 1

                        #     self.emulator.start("likelihood_testing")
                        #     if not self.emulator.likelihood_collection_differentiable:
                        #         loglikes_withNoise = self.logp_sample(thetas[i],noise = 1.)
                        #     else:
                        #         loglikes_withNoise = jnp.array([self.compute_loglike_uncertainty_for_differentiable_likelihood_from_normalized_parameters(thetas[i])])
                    
                        #     if not self.emulator.check_quality_criterium(jnp.array(loglikes_withNoise), reference_loglike=reference_loglike, parameters=state['parameters'], write_log=False):
                        #         # if the emulator passes noiseFree but fails with noise then the noise is too large
                        #         self.info('!!!!noise levels too large for convergence, reduce explained_variance_cutoff !!!!')
                        #         # note that it is normal to trigger this from time to time. for acceptable noise at the edge of interpolation area it can happen
                        self.debug("Emulator good enough")
                        # Add the point to the quality points
                        self.emulator.add_quality_point(state['parameters'])
                        
            self.debug("Testing time: "+ str(time.time()-start))

            self.print_status(i, thetas)

        self.write_to_log("NUTS: Finished Sampling \n")

        # save the chain and the logprobability
        # initialize the chain
        self.chain = np.zeros((self.M_adapt + nsteps + 1, self.ndim))

        # first get normalized chains and rescale them
        for i in range(self.M_adapt + nsteps + 1):
            self.chain[i] = self.retranform_parameters_from_normalized_eigenspace(thetas[i])

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

        self.emulator.start('emulate')
        logp, gradlogp = self.logp_and_grad(theta)
        self.emulator.emulate_counter += 1
        self.emulator.increment('emulate')
        self.emulator.print_status()


        if self.debug_mode:
            theta_untransformed = self.retranform_parameters_from_normalized_eigenspace(theta)
            _ = "Logposterior: "+ str(logp) + ' at ' + " ".join([self.parameter_names[i] + ": " + str(theta_untransformed[i]) for i in range(len(theta_untransformed))]) + "\n"
            self.write_to_log(_)

        if jnp.isnan(logp):
            raise ValueError("log probability of initial value is NaN.")
        if jnp.any(jnp.isnan(gradlogp)):
            raise ValueError("Gradient of log probability of initial value is NaN.")

        theta_f, r_f, logp, gradlogp = self._leapfrog(theta, r, eps)

        # First make sure that the step is not too large i.e. that we do not get diverging values.
        while jnp.isnan(logp) or jnp.any(jnp.isnan(gradlogp)):
            eps /= 2
            theta_f, r_f, logp, gradlogp = self._leapfrog(theta, r, eps)
            
        
        # Then decide in which direction to move
        self.emulator.start('emulate')
        logp0, _ = self.logp_and_grad(theta)
        self.emulator.emulate_counter += 1
        self.emulator.increment('emulate')
        self.emulator.print_status()


        if self.debug_mode:
            theta_untransformed = self.retranform_parameters_from_normalized_eigenspace(theta)
            _ = "Logposterior: "+ str(logp0) + ' at ' + " ".join([self.parameter_names[i] + ": " + str(theta_untransformed[i]) for i in range(len(theta_untransformed))]) + "\n"
            self.write_to_log(_)

        logjoint0 = logp0 - .5 * jnp.dot(r, r)
        logjoint = logp - .5 * jnp.dot(r_f, r_f)
        a = 2 * (logjoint - logjoint0 > jnp.log(.5)) - 1
        # And successively scale epsilon until the acceptance rate crosses .5
        while a * (logp - .5 * jnp.dot(r_f, r_f) - logjoint0) > a * jnp.log(.5):
            eps *= 2.**a
            theta_f, r_f, logp, _ = self._leapfrog(theta, r, eps)


        return key, eps

    # @partial(jax.jit, static_argnums=0)
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
        self.emulator.start('emulate')
        logp, gradlogp = self.logp_and_grad(theta)
        self.emulator.emulate_counter += 1
        self.emulator.increment('emulate')
        self.emulator.print_status()


        if self.debug_mode:
            theta_untransformed = self.retranform_parameters_from_normalized_eigenspace(theta)
            _ = "Logposterior: "+ str(logp) + ' at ' + " ".join([self.parameter_names[i] + ": " + str(theta_untransformed[i]) for i in range(len(theta_untransformed))]) + "\n"
            self.write_to_log(_)
        
        r = r + .5 * eps * gradlogp
        theta = theta + eps * r

        self.emulator.start('emulate')
        logp, gradlogp = self.logp_and_grad(theta)
        self.emulator.emulate_counter += 1
        self.emulator.increment('emulate')
        self.emulator.print_status()

        if self.debug_mode:
            theta_untransformed = self.retranform_parameters_from_normalized_eigenspace(theta)
            _ = "Logposterior: "+ str(logp) + ' at ' + " ".join([self.parameter_names[i] + ": " + str(theta_untransformed[i]) for i in range(len(theta_untransformed))]) + "\n"
            self.write_to_log(_)
            
        r = r + .5 * eps * gradlogp
        return theta, r, logp, gradlogp

    # @partial(jax.jit, static_argnums=0)
    def _init_iteration(self, theta: ndarray, key: ndarray) -> Tuple[ndarray, ndarray, float, float, float]:
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

        self.emulator.start('emulate')
        logprob, _ = self.logp_and_grad(theta)
        self.emulator.emulate_counter += 1
        self.emulator.increment('emulate')
        self.emulator.print_status()

        if self.debug_mode:
            theta_untransformed = self.retranform_parameters_from_normalized_eigenspace(theta)
            _ = "Logposterior: "+ str(logprob) + ' at ' + " ".join([self.parameter_names[i] + ": " + str(theta_untransformed[i]) for i in range(len(theta_untransformed))]) + "\n"
            self.write_to_log(_)
        
        logjoint = logprob - .5 * jnp.dot(r, r)
        u = random.uniform(subkeys[1]) * jnp.exp(logjoint)
        return key, r, u, logjoint, logprob

    # @partial(jax.jit, static_argnums=0)
    def _draw_direction(self, key: ndarray) -> Tuple[ndarray, int]:
        """Draw a random direction (-1 or 1)"""
        key, subkey = random.split(key)
        v = 2 * random.bernoulli(subkey) - 1
        return key, v

    # @partial(jax.jit, static_argnums=0)
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

    # @partial(jax.jit, static_argnums=0)
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

    # @partial(jax.jit, static_argnums=0)
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

    # @partial(jax.jit, static_argnums=0)
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

    # @partial(jax.jit, static_argnums=0)
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
