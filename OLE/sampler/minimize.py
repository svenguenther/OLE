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
            self.use_emulator = kwargs['use_emulator'] if 'use_emulator' in kwargs else True

            # check wether we want to calculate the loglikelihood or the logposterior
            self.logposterior = kwargs['logposterior'] if 'logposterior' in kwargs else False

            # check whether to use the gradients
            self.use_gradients = kwargs['use_gradients'] if 'use_gradients' in kwargs else True

            # set the method for the minimization
            self.method = 'L-BFGS-B' if 'method' not in kwargs else kwargs['method']

            # 

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

            # import scipy.optimize
            import scipy.optimize as opt

            self.optimizer = opt.minimize
        
            pass
        
        
        def minimize(self):
            # Run the sampler.

            # get the initial guess
            initial_position = self.get_initial_position(N=1, noramlized=True)[0]
            initial_position1 = self.get_initial_position(N=1, noramlized=False)[0]

            # get the bounds
            bounds = self.get_bounds(normalized=True)
            bounds1 = self.get_bounds(normalized=False)


            print(initial_position)
            print(initial_position1)
            print(bounds)
            print(bounds1)

            if self.use_gradients:
                # create differentiable loglike
                f = jax.jit(self.emulate_total_minusloglike_from_parameters_differentiable)     # this is the differentiable loglike
                grad_f = jax.jit(jax.grad(self.emulate_total_minusloglike_from_parameters_differentiable))
                hessian_f = jax.jit(jax.hessian(self.emulate_total_minusloglike_from_parameters_differentiable))

                res = self.optimizer(f, 
                                    initial_position, method=self.method, bounds=bounds, 
                                    jac=grad_f, 
                                    options={'disp': True},#, 'ftol': 1e-20, 'gtol': 1e-10 },
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
            self.bestfit = self.denormalize_parameters(res.x)
            self.max_loglike = res.fun

            print(self.bestfit)
            print(self.max_loglike)
            print(self.inv_hessian)
            


