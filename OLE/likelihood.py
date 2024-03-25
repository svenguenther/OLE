# This file contains a plain likelihood class that can be used to compute the loglikelihood.
from OLE.utils.base import BaseClass

class Likelihood(BaseClass):
    
        def __init__(self, name=None, **kwargs):
            super().__init__(name, **kwargs)
    
        def initialize(self, **kwargs):
            pass
    
        def loglike(self, state):
            # Compute the loglikelihood for the given parameters.
            loglike = 0.0
    
            return loglike
    
        def loglike_state(self, state):
            # Compute the loglikelihood for the given parameters.
            state['loglike'] = self.loglike(state)

            return state
        
        def loglike_gradient(self, state):
            # Compute the gradient of the loglikelihood for the given parameters.

            return state
