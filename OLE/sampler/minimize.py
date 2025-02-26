# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.theory import Theory
from OLE.likelihood import Likelihood
from OLE.emulator import Emulator
from OLE.utils.mpi import *
from OLE.sampler.base import Sampler

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

# Minimize Sampler. This sampler is used to minimize the likelihood or the posterior. It uses scipy.optimize.minimize to do so.
# We can set it to also use gradients that we obtain from the jax jit grad emulator.
class MinimizeSampler(Sampler):
    
        def __init__(self, name=None, **kwargs):
            super().__init__(name, **kwargs)
        
        def initialize(self, **kwargs):
            super().initialize(**kwargs)
    
            # flag whether to use the emulator or not
            self.use_emulator = self.hyperparameters['use_emulator'] if 'use_emulator' in self.hyperparameters else True

            # check wether we want to calculate the loglikelihood or the logposterior
            self.logposterior = self.hyperparameters['logposterior'] if 'logposterior' in self.hyperparameters else False

            # check whether to use the gradients
            self.use_gradients = self.hyperparameters['use_gradients'] if 'use_gradients' in self.hyperparameters else True

            # number of restarts for the minimization
            self.n_restarts = self.hyperparameters['n_restarts'] if 'n_restarts' in self.hyperparameters else 1

            # set the method for the minimization
            self.method = 'TNC' if 'method' not in self.hyperparameters else self.hyperparameters['method']

            # set flag if fisher matrix should be computed
            self.compute_fisher = self.hyperparameters['compute_fisher'] if 'compute_fisher' in self.hyperparameters else True

            # flag if to store results
            self.store_results = self.hyperparameters['store_results'] if 'store_results' in self.hyperparameters else True

            # falg if we want to check the minimization result
            self.check_minimum = self.hyperparameters['check_minimum'] if 'check_minimum' in self.hyperparameters else True

            # we cannot use gradients if we do not use the emulator
            if not self.use_emulator:
                self.use_gradients = False

            # run evaluation once to train the emulator
            if self.use_emulator and not self.emulator.trained:
                self.info("Training emulator")
                parameter_list = jnp.array([self.parameter_dict[key]['ref']['mean'][0] for key in self.parameter_dict.keys()])
                parameter_list = self.transform_parameters_into_normalized_eigenspace(parameter_list)
                self.compute_total_logposterior_from_normalized_parameters(parameter_list)
                self.info("Emulator trained")

            # output error if we use the emulator but it is not trained
            if self.use_emulator and not self.emulator.trained:
                self.error("Emulator is not trained yet. Please train the emulator first or set use_emulator=False")
                raise ValueError

            # import scipy.optimize
            import scipy.optimize as opt

            self.optimizer = opt.minimize
        
            pass
        
        
        def minimize(self):
            # Run the sampler.

            self.write_to_log("Starting minimization \n")

            # get the initial guess
            initial_position = self.get_initial_position(N=self.hyperparameters['n_restarts'], normalized=True)
            # get the bounds
            bounds = self.get_bounds(normalized=True)

            self.results = []
            self.bestfits = []
            self.max_loglikes = []

            minimum_checked = False
            counter = 0 # we improve only once!

            while (not minimum_checked) and (counter <= 1):
                counter += 1

                # check if jitted functions are already defined otherwise define them
                # jit functions
                if self.use_emulator:
                    # if not self.debug_mode:
                    if not self.debug_mode:
                        self.f = jax.jit(self.emulate_total_minuslogposterior_from_normalized_parameters_differentiable)
                        if self.use_gradients:
                            self.grad_f = jax.jit(jax.grad(self.emulate_total_minuslogposterior_from_normalized_parameters_differentiable))
                            if self.compute_fisher:
                                self.hessian_f = (jax.hessian(self.emulate_total_minuslogposterior_from_normalized_parameters_differentiable))
                    else:
                        self.f = self.emulate_total_minuslogposterior_from_normalized_parameters_differentiable
                        if self.use_gradients:
                            self.grad_f = jax.grad(self.emulate_total_minuslogposterior_from_normalized_parameters_differentiable)
                            if self.compute_fisher:
                                self.hessian_f = (jax.hessian(self.emulate_total_minuslogposterior_from_normalized_parameters_differentiable))
                else:
                    self.f = self.compute_total_minuslogposterior_from_normalized_parameters



                for i in range(self.n_restarts):

                    if self.use_gradients:
                        self.method = 'TNC'
                        res = self.optimizer(self.f, 
                                            initial_position[i], method=self.method, bounds=bounds, 
                                            jac=self.grad_f, 
                                            options={'disp': False, 'maxfun':2000, 'accuracy': 0.01,'eps':0.01},#, 'ftol': 1e-20, 'gtol': 1e-10 },
                                            )
                    else:
                        if self.use_emulator:
                            res = self.optimizer(self.f,
                                                initial_position[i], method=self.method, bounds=bounds, 
                                                options={'disp': False,'eps':0.01})
                        else:
                            res = self.optimizer(self.f,
                                                initial_position[i], method=self.method, bounds=bounds, 
                                                options={'disp': False,'eps':0.01})

                    self.results.append(res)
                    self.bestfits.append(self.retranform_parameters_from_normalized_eigenspace(res.x))
                    self.max_loglikes.append(res.fun)

                # write to info
                self.info("Minimization results of " + str(self.n_restarts) + " restarts")
                mean_loglike = np.mean(self.max_loglikes)
                std_loglike = np.std(self.max_loglikes)
                self.info("Mean loglike: " + str(mean_loglike) + " +- " + str(std_loglike))

                # select best fit
                idx = np.argmin(self.max_loglikes)
                self.res = self.results[idx]
                self.bestfit = self.bestfits[idx]
                self.max_loglike = self.max_loglikes[idx]

                # check performance by sampling the likelihood at the bestfit
                self.emulator.start("emulate")
                self.emulator.increment('emulate')
                self.emulator.start("likelihood_testing")
                if self.use_emulator:
                    # go throught the likelihoods and check if all of them are differentiable
                    # if self.emulator.likelihood_collection_differentiable:
                    #     self.uncertainty = self.compute_loglike_uncertainty_for_differentiable_likelihood_from_normalized_parameters(self.res.x)
                    # else:
                    #     loglikes_withNoise = self.sample_emulate_total_logposterior_from_parameters_differentiable(self.bestfit,noise = 1.)
                    loglikes_withNoise = self.sample_emulate_total_logposterior_from_parameters(self.bestfit)
                    self.uncertainty = np.std(loglikes_withNoise)
                else:
                    self.uncertainty = 0.0            

                # check if the minimum is correct
                if self.use_emulator and self.check_minimum:
                    parameters={key: jnp.array([self.bestfit[i]]) for i, key in enumerate(self.parameter_dict.keys())}
                    # update with constant parametesr
                    parameters.update({key: jnp.array([self.parameter_dict_constant[key]['value']]) for key in self.parameter_dict_constant.keys()})
                    
                    minimum_checked = self.emulator.check_quality_criterium(jnp.array(loglikes_withNoise), reference_loglike=self.max_loglike, parameters=parameters)
                    if not minimum_checked:

                        if counter == 1:
                            self.info("Quality criterium not fulfilled for best fit! Adding new state to emulator")
                            state = {'parameters': parameters, 'quantities': {}, 'loglike': {}, 'total_loglike': None}
                            state = self.theory.compute(state)
                            for likelihood in self.likelihood_collection.keys():
                                state = self.likelihood_collection[likelihood].loglike_state(state)
                            state['total_loglike'] = jnp.array([jnp.array(list(state['loglike'].values())).sum() + self.compute_logprior(state)])
                            added_flag, _ = self.emulator.add_state(state)
                            if not added_flag:
                                self.info("Could not add state to emulator! Check if we prevent this state to be added. Accepting large error and continuing with minimization")
                                minimum_checked = True
                            else:
                                del self.f
                                del self.grad_f
                        else:
                            # we need to send a warning.
                            minimum_checked = True
                            self.warning("Quality criterium not fulfilled for best fit after adding a new state to the emulator.")

                    else:
                        self.info("Minimum checked. Finished minimization")
                else:
                    minimum_checked = True


            # comptue the inverse hessian
            if self.compute_fisher:
                if self.use_gradients:
                    self.inv_hessian = self.denormalize_inv_hessematrix( np.linalg.inv(self.hessian_f(self.res.x)) )
                else:
                    try:
                        self.inv_hessian = self.denormalize_inv_hessematrix( self.res.hess_inv.todense() )
                    except:
                        self.inv_hessian = None

                # now we want to store inv_hessian in self.hyperparameters['output_directory']/fisher.covmat as txt file. First line will be the name of the parameters
                if self.inv_hessian is not None and self.store_results:
                    with open(self.hyperparameters['output_directory'] + '/fisher.covmat', 'w') as f:
                        f.write(' '.join([key for key in self.parameter_dict.keys()]) + '\n')
                        for row in self.inv_hessian:
                            f.write(' '.join([str(val) for val in row]) + '\n')
            
            if self.store_results:
                # now store bestfit with max likelihood
                with open(self.hyperparameters['output_directory'] + '/bestfit.txt', 'w') as f:
                    f.write(' '.join([key for key in self.parameter_dict.keys()]) + '\n')
                    f.write(' '.join([str(val) for val in self.bestfit]) + '\n')
                    f.write(' '.join([str(self.max_loglike)]) + '\n')
                    f.write(' '.join([str(self.uncertainty)]) + '\n')






            


