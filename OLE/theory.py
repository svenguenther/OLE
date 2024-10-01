"""
Theory
"""

# This file contains a plain theory class that can be used to compute observables.
from OLE.utils.base import BaseClass


class Theory(BaseClass):

    # list with input parameters
    requirements: list
    hyperparameters: dict

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)

        self.requirements = []

    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        if 'requirements' in kwargs:
            self.requirements = kwargs["requirements"] 

        if 'parameters' in kwargs:
            self.parameters = kwargs["parameters"]

        pass

    def compute(self, state):
        # Compute the observable for the given parameters.

        return state
    
    def required_parameters(self):
        # Compute the observable for the given parameters.

        return self.parameters

    def requirements(self):
        # Compute the observable for the given parameters.

        return self.requirements
