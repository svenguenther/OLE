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
