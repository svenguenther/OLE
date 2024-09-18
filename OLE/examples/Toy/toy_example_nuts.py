# This is a toy example of the differential NUTS sampler in combination with OLE.

# 1. Define a theory. It has 2 functions: initialize and compute. 
#    The initialize function sets the requirements of the theory. 
#    The compute function computes the observables for the given parameters. The input is a dictionary with the parameters and the output is a dictionary with the observables.
#    It has the attributes 'requirements' which states the parameters it requires to compute the observables.

# 2. Define a likelihood. It has 2 functions: initialize and loglike.
#    The initialize function sets the requirements of the likelihood.
#    The loglike function computes the loglikelihood for the given parameters.

# 3. Define the settings for the emulator.

# 4. Define the sampler. It has 2 functions: initialize and run_mcmc.

# 5. Set the parameters for the sampler. It has the attributes 'prior', 'ref', and 'proposal'. 
#    The prior is the prior distribution of the parameter. The ref is the reference value of the parameter from where the sampling starts. The proposal is the proposal length of std of the parameter. It allows for faster burn in.


# state has the following structure:
# state = {
#     "parameters": {
#         "x1": [123],
#         "x2": [456],
#         "x3": [789],
#     },
#     "quantities": {
#         "y_array": [element1, element2, ...],
#         "y_scalar": [element1],
#     },
#     "loglike": 123, (or None if not available)
# }

from OLE.theory import Theory
from OLE.likelihood import Likelihood
import jax.numpy as jnp
import matplotlib.pyplot as plt


class my_theory(Theory):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)   

        # input parameters of the theory
        self.requirements = ['x1', 'x2', 'x3']

    def compute(self, state):
        # Compute the observable for the given parameters.

        # parameters x1,x2,x3
        # observables y_array, y_scalar
        self.requirements = ['x1', 'x2', 'x3']

        state['quantities']['y_array'] = jnp.array([0.3*jnp.sin(2.*state['parameters']['x1'][0])+state['parameters']['x1'][0], 0.5*state['parameters']['x3'][0]+state['parameters']['x2'][0], state['parameters']['x3'][0]])
        state['quantities']['y_scalar'] = jnp.array([jnp.sum(state['quantities']['y_array'])])
        return state


class my_likelihood(Likelihood):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)   

        # input parameters of the theory
        self.requirements = ['y_array', 'y_scalar']


    def loglike(self, state):
        # Compute the loglikelihood for the given parameters.
        loglike = -10.*jnp.sum((state['quantities']['y_array'])**2) - 10.*(state['quantities']['y_scalar']-0.5)**2 - state['quantities']['y_array'][0]*state['quantities']['y_array'][1]
        return loglike
    
    #def loglike_gradient(self, state, type): # define a function for each quantity.
            # Compute the gradient of the loglikelihood for the given parameters.
        
    #    if type == 'y_array':
    #        return [
    #            -20.*(state['quantities']['y_array']-jnp.ones(3))[0],
    #            -20.*(state['quantities']['y_array']-jnp.ones(3))[1],
    #            -20.*(state['quantities']['y_array']-jnp.ones(3))[2]
    #            ]
    #    elif type == 'y_scalar':
    #        return [
    #            -20.*(state['quantities']['y_scalar']-3.0)
    #            ]
          
    #    return []


# init theory and likelihood
my_theory = my_theory()
my_likelihood = my_likelihood()

likelihood_settings = {}
sampling_settings = {
    'output_directory': './toy',
}
theory_settings = {}
emulator_settings = {
    'logfile': './toy/log.txt',
    'min_variance_per_bin': 1e-6,
    'cache_file': './toy/cache.pkl',
    'dimensionality': 3,

    'test_emulator': True,
}

# load sampler 
from OLE.sampler import EnsembleSampler, Sampler, NUTSSampler
my_sampler = NUTSSampler()


# set parameters
my_parameters = {'x1': {'prior': {'min': 0.0, 'max': 3.0, 'type': 'uniform'},
                        'ref': {'mean': 0.2, 'std': 0.1},
                        'proposal': 1.0,},
                    'x2': {'prior': {'min': 0.0, 'max': 3.0, 'type': 'uniform'},
                        'ref': {'mean': 0.2, 'std': 0.1},
                        'proposal': 1.0,},
                    'x3': {'prior': {'min': 0.0, 'max': 3.0, 'type': 'uniform'},
                        'ref': {'mean': 0.2, 'std': 0.1},
                        'proposal': 1.0,},
                           } 

# initialize sampler
my_sampler.initialize(theory=my_theory, likelihood=my_likelihood, parameters=my_parameters, emulator_settings=emulator_settings, likelihood_settings=likelihood_settings, theory_settings=theory_settings, sampling_settings=sampling_settings)

# Note the total run steps are   (nsteps * nwalkers * MPI_size)
n_steps = 1000
my_sampler.run_mcmc(n_steps)

chain = my_sampler.chain

# make corner
import numpy as np
import corner
fig = corner.corner(np.array(chain))
plt.savefig('corner_nuts.png')
