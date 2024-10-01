# this is a theory implementation of CLASS, a Boltzmann code for cosmological perturbations. The theory is initialized with a set of parameters, and the compute method is used to compute the observable for the given parameters. The likelihood is then used to compare the theoretical predictions with the observed data.

from OLE.theory import Theory
from OLE.utils.base import c_km_s
import classy
import numpy as np
import jax.numpy as jnp

class CLASS(Theory):
    def initialize(self, **kwargs):
        super().initialize(**kwargs)   

        # Initialize the CLASS code
        self.cosmo = classy.Class()

        # look if class_settings are given in the input
        if 'class_settings' in kwargs:
            self.class_settings = kwargs['class_settings']
        else:
            raise ValueError('class_settings not given in the theory input')

        
        # update the class settings with the hyperparameters
        self.class_settings

        return 
    
    def translate_CLASS_parameters(self, parameters):
        # Translate the parameters to the CLASS format
        class_parameters = {}

        # go through the parameters
        for key, value in parameters.items():
            
            # only add the required parameters
            if key in self.required_parameters():
                
                # add here the translation
                if key == 'logA':
                    class_parameters['A_s'] = 1e-10*np.exp(value)
                else:
                    class_parameters[key] = value

        return class_parameters
    

    def compute(self, state):
        # Compute the observable for the given parameters.
        class_input = self.class_settings.copy()

        # set parameters
        class_parameters = self.translate_CLASS_parameters(state['parameters'])
        for key in class_parameters.keys():
            class_input[key] = class_parameters[key][0]

        # set CLASS parameters
        self.cosmo.set(class_input)

        # compute CLASS
        self.cosmo.compute()

        T_cmb = self.cosmo.T_cmb()

        # check for cls
        for key in ['tt', 'ee', 'te', 'bb']:
            if key in self.requirements.keys():
                cls = self.cosmo.lensed_cl(self.class_settings['l_max_scalars'])
                state['quantities'][key] = cls[key]

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

