# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.theory import Theory
from OLE.likelihood import Likelihood
from OLE.emulator import Emulator
from OLE.utils.mpi import *
from OLE.sampler.base import Sampler
from OLE.sampler.minimize import MinimizeSampler

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
class ProfileLikelihoodSampler(Sampler):
    
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
            
            # flag if to store results
            self.store_results = self.hyperparameters['store_results'] if 'store_results' in self.hyperparameters else True

            # information about the profiled parameter
            if not 'profiled_parameter' in self.hyperparameters:
                self.error("Please provide a profiled parameter")
                raise ValueError
            self.profiled_parameter = self.hyperparameters['profiled_parameter']

            # they come in the shape: {'name': 'h', 'range': [0.6, 0.8], 'n_samples': 10}

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

            self.write_to_log("Profile Likelihood computation \n")

            # we will profile the parameter in the range
            profiled_parameter = self.profiled_parameter['name']
            range_ = self.profiled_parameter['range']
            n_samples = self.profiled_parameter['n_samples']
            samples = np.linspace(range_[0], range_[1], n_samples)

            # then we will create a minimizer sampler and reuse if for each sample. Additionally we will store the results
            results = jnp.zeros((n_samples, 3)) # the value of the profiled parameter, the logposterior and the uncertainty

            # update the parameter dict such that the profiled parameter has the value of the sample

            minimizer_settings = self.sampling_settings.copy()
            minimizer_settings['compute_fisher'] = False
            minimizer_settings['store_results'] = False


            for i, sample in enumerate(samples):
                self.parameter_dict[profiled_parameter]['value'] = sample

                my_minizer = MinimizeSampler() 

                my_minizer.initialize(theory=self.theory, 
                                    likelihood_collection=self.likelihood_collection, 
                                    parameters=self.parameter_dict, 
                                    emulator_settings = self.emulator_settings,
                                    likelihood_collection_settings = self.likelihood_collection_settings,
                                    theory_settings = self.theory_settings,
                                    sampling_settings = minimizer_settings,
                                    emulator=self.emulator, 
                                    parameter_dict=self.parameter_dict)
                
                # if hasattr(self, 'f'):
                #     my_minizer.f = self.f
                #     my_minizer.logp_sample = self.logp_sample
                #     if self.use_gradients:
                #         my_minizer.grad_f = self.grad_f
                
                my_minizer.minimize()

                results = results.at[i, 0].set(sample)
                results = results.at[i, 1].set(my_minizer.max_loglike)
                results = results.at[i, 2].set(my_minizer.uncertainty)

                my_minizer.initialized = False

                # del my_minizer.logger TODO: Some memory leak here
                # # del my_minizer.emulator
                # del my_minizer.theory
                # del my_minizer.likelihood_collection
                # del my_minizer.parameter_dict
                # del my_minizer.f 
                # del my_minizer.grad_f
                # del my_minizer.logp_sample
                # del my_minizer






            print(results)

            # store the results
            self.results = results

            if self.store_results:
                # now store bestfit with max likelihood
                with open(self.hyperparameters['output_directory'] + '/profile_likelihood_results.txt', 'w') as f:
                    f.write(profiled_parameter + ' max_like sigma_like \n')
                    for row in results:
                        f.write(' '.join([str(val) for val in row]) + '\n')

            pass
        


        