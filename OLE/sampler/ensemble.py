# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
# The NUTS sampler is based upon jaxnuts: https://github.com/guillaume-plc/jaxnuts
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

import emcee
import time 

class EnsembleSampler(Sampler):

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs else 50
        self.ndim = len(self.parameter_dict)

        # initialize the sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.compute_total_logposterior_from_normalized_parameters)

        pass
    
    
    def run_mcmc(self, nsteps, **kwargs):
        # Run the sampler.
        # Initialize the position of the walkers.            
        pos = jnp.zeros((self.nwalkers, self.ndim))


        initial_seed = int(time.time())+get_mpi_rank()

        # sample the initial positions of the walkers from unit gaussian
        pos = jax.random.normal(jax.random.PRNGKey(initial_seed), (self.nwalkers, self.ndim))

        self.sampler.run_mcmc(pos, nsteps, **kwargs, tune=True)

        # save the chain and the logprobability
        # first get normalized chains and rescale them
        self.chain = self.sampler.get_chain()
        
        # flatten first and second dimensions
        self.chain = self.chain.reshape((-1, self.ndim))

        for i in range(len(self.chain)):
            self.chain[i] = self.retranform_parameters_from_normalized_eigenspace(self.chain[i])


        self.logprobability = self.sampler.get_log_prob()

        # Append the chains to chain.txt using a lock
        lock = fasteners.InterProcessLock('chain.txt.lock')
        with lock:
            _ = self.hyperparameters['output_directory'] + '/chain_%d.txt'%get_mpi_rank()
            with open(self.hyperparameters['output_directory'] + '/chain.txt', 'ab') as f:
                np.savetxt(f, np.hstack([self.logprobability.reshape(-1)[:,None], self.chain.reshape((-1,self.ndim))]) )
            with open(_, 'ab') as f:
                np.savetxt(f, np.hstack([self.logprobability.reshape(-1)[:,None], self.chain.reshape((-1,self.ndim))]) )
