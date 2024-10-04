# this is a theory implementation of CAMB, a Boltzmann code for cosmological perturbations. The theory is initialized with a set of parameters, and the compute method is used to compute the observable for the given parameters. The likelihood is then used to compare the theoretical predictions with the observed data.

#
# NOTE: There is still some 10% offset at l>5000 between CAMB and CLASS...
# TODO: Someone should look into this.
#

from OLE.theory import Theory
from OLE.utils.base import c_km_s
import camb
import numpy as np
import jax.numpy as jnp

class CAMB(Theory):
    def initialize(self, **kwargs):
        super().initialize(**kwargs)   

        # Initialize the CLASS code
        self.cosmo = camb

        # look if cosmo_settings are given in the input
        if 'cosmo_settings' in kwargs:
            self.camb_settings = kwargs['cosmo_settings']
        else:
            raise ValueError('cosmo_settings not given in the theory input')

        self.camb_settings.update({"halofit_version":'mead',
                                   'nonlinear':3, 
                                   'lens_potential_accuracy':4,})

        return 
    
    def translate_CAMB_parameters(self, parameters):
        # Translate the parameters to the CLASS format
        camb_parameters = {}

        # go through the parameters
        for key, value in parameters.items():
            
            # only add the required parameters
            if key in self.required_parameters():
                
                # add here the translation
                if key == 'logA':
                    camb_parameters['As'] = 1e-10*np.exp(value)
                elif key == 'tau_reio':
                    camb_parameters['tau'] = value
                elif key == 'n_s':
                    camb_parameters['ns'] = value
                elif key == 'omega_b':
                    camb_parameters['ombh2'] = value
                elif key == 'omega_cdm':
                    camb_parameters['omch2'] = value
                elif key == 'h':
                    camb_parameters['H0'] = value*100
                else:
                    camb_parameters[key] = value

        return camb_parameters
    
    def translate_CAMB_settings(self, settings):
        # Translate the settings to the CLASS format
        camb_settings = {}

        # go through the settings
        for key, value in settings.items():
            if key == 'l_max_scalars':
                camb_settings['max_l'] = value+100
        return camb_settings

    def compute(self, state):
        # Compute the observable for the given parameters.
        camb_input = self.camb_settings.copy()
        camb_input = self.translate_CAMB_settings(camb_input)

        # set parameters
        camb_parameters = self.translate_CAMB_parameters(state['parameters'])
        for key in camb_parameters.keys():
            camb_input[key] = camb_parameters[key][0]

        # set CLASS parameters
        params = self.cosmo.set_params(**camb_input)

        # calculate results
        results = self.cosmo.get_results(params)

        # check for cls
        for key in ['tt', 'ee', 'te', 'bb','pp','tp']:
            if key in self.requirements.keys():
                powers = results.get_cmb_power_spectra(params, CMB_unit='K', raw_cl=True)
                fac =  2 * np.pi / 2.7255**2
                if key == 'tt':
                    state['quantities'][key] = powers['total'][:,0]*fac
                elif key == 'ee':
                    state['quantities'][key] = powers['total'][:,1]*fac
                elif key == 'te':
                    state['quantities'][key] = powers['total'][:,3]*fac
                elif key == 'bb':
                    state['quantities'][key] = powers['total'][:,2]*fac
                elif key == 'pp':
                    state['quantities'][key] = powers['lens_potential'][:,0]

        # check for bao and other quantities
        if 'bao' in self.requirements.keys():
            # go through all BAO likelihoods
            for bao in self.requirements['bao']:
                bao_name = list(bao.keys())[0]
                bao_data = []

                for key, z, rs_fid in bao[bao_name]:
                    if key == 'DM_over_rs':
                        bao_data.append((1+z)*self.cosmo.angular_distance(z)*rs_fid/self.cosmo.rs_drag())
                    elif key == 'bao_Hz_rs':
                        bao_data.append(self.cosmo.Hubble(z)*c_km_s*self.cosmo.rs_drag()/rs_fid)
                    else:
                        raise ValueError('BAO key not recognized:', key)

                state['quantities'][bao_name] = jnp.array(bao_data)

        return state

