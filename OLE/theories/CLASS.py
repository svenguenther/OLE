# this is a theory implementation of CLASS, a Boltzmann code for cosmological perturbations. The theory is initialized with a set of parameters, and the compute method is used to compute the observable for the given parameters. The likelihood is then used to compare the theoretical predictions with the observed data.

from OLE.theory import Theory
import classy
import numpy as np

class CLASS(Theory):
    def initialize(self, **kwargs):
        super().initialize(**kwargs)   

        # input parameters of the theory
        self.requirements = ['h', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'logA']

        self.cosmo = classy.Class()

        self.class_settings = {'output':'tCl,pCl,lCl,mPk',
                        'lensing':'yes',
                        #'l_max_scalars':3200, #  for SPT3G_2018_TTTEEE
                        # 'l_max_scalars':7925, #  for ACT

                        'N_ur':3.048,
                        'output_verbose':1,
                        }
        
        # update the class settings with the hyperparameters
        self.class_settings.update(self.hyperparameters)

        return 
    

    def compute(self, state):
        # Compute the observable for the given parameters.
        class_input = self.class_settings.copy()
        class_input['A_s'] = 1e-10*np.exp(state['parameters']['logA'][0])
        class_input['h'] = state['parameters']['h'][0]
        class_input['n_s'] = state['parameters']['n_s'][0]
        class_input['omega_b'] = state['parameters']['omega_b'][0]
        class_input['omega_cdm'] = state['parameters']['omega_cdm'][0]
        class_input['tau_reio'] = state['parameters']['tau_reio'][0]

        if state['parameters']['tau_reio'][0] < 0.01:
            class_input['tau_reio'] = 0.01

        self.cosmo.set(class_input)
        self.cosmo.compute()

        cls = self.cosmo.lensed_cl(self.class_settings['l_max_scalars'])
        
        T_cmb = 2.7255

        state['quantities']['tt'] = cls['tt'] * (T_cmb*1e6)**2
        state['quantities']['ee'] = cls['ee'] * (T_cmb*1e6)**2
        state['quantities']['te'] = cls['te'] * (T_cmb*1e6)**2
        state['quantities']['bb'] = cls['bb'] * (T_cmb*1e6)**2


        return state

