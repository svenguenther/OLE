# This is the base class for BAO likelihoods. It is inherited by the BAO likelihoods for the different surveys.
from OLE.likelihood import Likelihood
import yaml
import os
import jax.numpy as jnp
import sys


class bao_base(Likelihood):
    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        # set flags
        self.differentiable = True
        self.jitable = True
        
        # load data from self._name.yaml
        with open(os.path.dirname(__file__) + '/' + self._name + '.yaml', 'r') as file:
            self.yaml = yaml.safe_load(file)

        if 'rs_fid' in self.yaml:
            self.rs_fid = self.yaml['rs_fid']
        else:
            self.rs_fid = 1.0

        self.covmat = jnp.array(self.yaml['covmat'])
        self.inv_covmat = jnp.linalg.inv(self.covmat)
        self.data = self.yaml['data']
        self.data_values = jnp.array([data_point[2] for data_point in self.data])


        # go through data and save teh demands
        self.requirements = {self._name: [(_[0], _[1], self.rs_fid) for _ in self.data]}


    # this function can be used to update the theory settings
    def update_theory_settings(self, theory_settings):
        super().update_theory_settings(theory_settings)

        if 'bao' not in theory_settings['requirements']:
            theory_settings['requirements']['bao'] = [self.requirements]
        else:
            theory_settings['requirements']['bao'].append(self.requirements)

        return theory_settings
    
    def loglike(self, state):
        # get the data from the state
        data = state['quantities'][self._name]

        # compute the chi2
        chi2 = jnp.dot(jnp.dot(data - self.data_values, self.inv_covmat), data - self.data_values) 

        return jnp.array([-0.5 * chi2])
    

    
