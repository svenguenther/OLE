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

            # set the method for the minimization
            self.method = 'TNC' if 'method' not in self.hyperparameters else self.hyperparameters['method']

            # 

            # run evaluation once to train the emulator
            if self.use_emulator and not self.emulator.trained:
                self.info("Training emulator")
                parameter_list = jnp.array([self.parameter_dict[key]['ref']['mean'][0] for key in self.parameter_dict.keys()])
                parameter_list = self.transform_parameters_into_normalized_eigenspace(parameter_list)
                self.compute_total_loglike_from_normalized_parameters(parameter_list)
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

            # get the initial guess
            initial_position = self.get_initial_position(N=1, normalized=True)[0]
            initial_position1 = self.get_initial_position(N=1, normalized=False)[0]

            # get the bounds
            bounds = self.get_bounds(normalized=True)
            bounds1 = self.get_bounds(normalized=False)

            if self.use_gradients:
                # create differentiable loglike
                f = jax.jit(self.emulate_total_minusloglike_from_parameters_differentiable)     # this is the differentiable loglike
                grad_f = jax.jit(jax.grad(self.emulate_total_minusloglike_from_parameters_differentiable))
                hessian_f = (jax.hessian(self.emulate_total_minusloglike_from_parameters_differentiable))

                print(f(initial_position))
                print(grad_f(initial_position))

                self.method = 'TNC'
                res = self.optimizer(f, 
                                    initial_position, method=self.method, bounds=bounds, 
                                    jac=grad_f, 
                                    options={'disp': True, 'maxfun':2000, 'accuracy': 0.01},#, 'ftol': 1e-20, 'gtol': 1e-10 },
                                    )
                
                self.inv_hessian = self.denormalize_inv_hessematrix( np.linalg.inv(hessian_f(res.x)) )
            
            else:
                f = jax.jit(self.emulate_total_minusloglike_from_parameters_differentiable)

                res = self.optimizer(f,
                                    initial_position, method=self.method, bounds=bounds, 
                                    options={'disp': True})
                try:
                    self.inv_hessian = self.denormalize_inv_hessematrix( res.hess_inv.todense() )
                except:
                    self.inv_hessian = None
                
            self.res = res
            self.bestfit = self.retranform_parameters_from_normalized_eigenspace(res.x)
            self.max_loglike = res.fun

            # now we want to store inv_hessian in self.hyperparameters['output_directory']/fisher.covmat as txt file. First line will be the name of the parameters
            if self.inv_hessian is not None:
                with open(self.hyperparameters['output_directory'] + '/fisher.covmat', 'w') as f:
                    f.write(' '.join([key for key in self.parameter_dict.keys()]) + '\n')
                    for row in self.inv_hessian:
                        f.write(' '.join([str(val) for val in row]) + '\n')

            # now store bestfit with max likelihood
            with open(self.hyperparameters['output_directory'] + '/bestfit.txt', 'w') as f:
                f.write(' '.join([key for key in self.parameter_dict.keys()]) + '\n')
                f.write(' '.join([str(val) for val in self.bestfit]) + '\n')
                f.write(' '.join([str(self.max_loglike)]) + '\n')





            


