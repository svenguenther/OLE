# This file contains a plain theory class that can be used to compute observables.
from OLE.utils.base import BaseClass

class Theory(BaseClass):

    # list with input parameters
    requirements: list

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)

    def initialize(self, **kwargs):
        pass

    def compute(self, state):
        # Compute the observable for the given parameters.

        return state
    
    def requirements(self):
        # Compute the observable for the given parameters.

        return self.requirements