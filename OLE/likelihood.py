"""
This file contains a plain likelihood class that can be used to compute the loglikelihood.
"""

from OLE.utils.base import BaseClass


class Likelihood(BaseClass):

    def __init__(self, name=None, **kwargs):
        self.input_keys = []
        self.nuisance_sample_dict = {}
        self.hyperparameters = {}
        super().__init__(name, **kwargs)

    def update_theory_settings(self, theory_settings):
        # this function can be used to update the theory settings
        # check if it already has a sub direcory for requirements
        if 'requirements' not in theory_settings:
            theory_settings['requirements'] = {}

        return theory_settings

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.hyperparameters = kwargs
        pass

    def loglike(self, state):
        # Compute the loglikelihood for the given parameters.
        loglike = 0.0

        return loglike

    def loglike_state(self, state):
        # Compute the loglikelihood for the given parameters.
        state["loglike"][self._name] = self.loglike(state)

        return state

    def loglike_gradient(self, state):
        # Compute the gradient of the loglikelihood for the given parameters.

        return state
