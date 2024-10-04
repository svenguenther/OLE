import numpy as np
import jax.numpy as jnp
import candl
import candl.data
from OLE.likelihood import Likelihood
import yaml
import os

class candl_likelihood(Likelihood):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        # Grab data set
        like_requested = kwargs["candl_dataset"]
        clear_priors = kwargs["clear_priors"] if "clear_priors" in kwargs else False
        if "candl.data." in like_requested:
            # Shortcut for data set
            if clear_priors:
                self.candl_like = candl.Like(eval(like_requested), priors=[])
            else:
                self.candl_like = candl.Like(eval(like_requested))
        else:
            # Path to data set
            if clear_priors:
                self.candl_like = candl.Like(like_requested, priors=[])
            else:
                self.candl_like = candl.Like(like_requested)
            
        
        # Grab required parameters (from data model and priors)
        self.input_keys = list(np.unique(self.candl_like.required_nuisance_parameters + self.candl_like.required_prior_parameters))

        # Grab spectrum conversion helper
        self.cl2dl = self.candl_like.ells * (self.candl_like.ells + 1) / (2.0 * jnp.pi) * (1e6)**2

        # Grab nuisance parameters.
        if kwargs["candl_dataset"] == 'candl.data.ACT_DR4_TTTEEE':
            # load yaml from 'ACT_DR4_TTTEEE.yaml' and convert to python dict
            with open(os.path.dirname(__file__) + '/ACT_DR4_TTTEEE.yaml', 'r') as file:
                self.nuisance_sample_dict = yaml.safe_load(file)['parameters']
        elif kwargs["candl_dataset"] == 'candl.data.SPT3G_2018_TTTEEE':
            # load yaml from 'SPT3G_2018_TTTEEE.yaml' and convert to python dict
            with open(os.path.dirname(__file__) + '/SPT3G_2018_TTTEEE.yaml', 'r') as file:
                self.nuisance_sample_dict = yaml.safe_load(file)['parameters']
        else:
            # no other dataset has been implemented so far
            self.nuisance_sample_dict = {}

        return
    
    # this function can be used to update the theory settings
    def update_theory_settings(self, theory_settings):
        super().update_theory_settings(theory_settings)

        # check if cosmo_settings are given in the input
        if 'cosmo_settings' not in theory_settings:
            theory_settings['cosmo_settings'] = {}

        # Update l_max_scalars to be the maximum of the current value and the value from the data set
        if 'l_max_scalars' not in theory_settings['cosmo_settings']:
            theory_settings['cosmo_settings']['l_max_scalars'] = self.candl_like.ell_max
        else:
            theory_settings['cosmo_settings']['l_max_scalars'] = max(self.candl_like.ell_max, theory_settings['cosmo_settings']['l_max_scalars'])

        # Add requirements for the theory
        theory_settings['requirements'].update({'tt': None, 'ee': None, 'te': None})
        
        return theory_settings

    # @partial(jax.jit, static_argnums=(0,))
    def loglike(self, state):
        # Compute the loglikelihood for the given parameters.

        # Grab calculated spectra, convert to Dl
        Dl = {'ell': self.candl_like.ells}
        for spec_type in self.candl_like.unique_spec_types:
            if spec_type == 'TT':
                print(state['quantities'][spec_type.lower()][2:])
            Dl[spec_type] = state['quantities'][spec_type.lower()][2:] * self.cl2dl 



        # Shuffle into parameters, spectra into dictionary, convert tau naming conventions
        candl_input = {}
        for key in self.input_keys:
            if key == 'tau':
                candl_input['tau'] = state['parameters']['tau_reio'][0]
            else:
                candl_input[key] = state['parameters'][key][0]
        candl_input['Dl'] = Dl

        # Hand off to candl
        loglike = self.candl_like.log_like(candl_input)

        return jnp.array([loglike])

