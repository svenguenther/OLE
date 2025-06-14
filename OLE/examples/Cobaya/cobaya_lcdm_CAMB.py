# This is an example of a CAMB cobaya runfile for a simple LCDM model.
#
import importlib
import importlib.util
import sys
import os

from OLE.interfaces.cobaya_interface import *
data_dir = os.getcwd()
# Additionally to the vanilla cobaya info dictionary, we have added the 'emulate' and 'emulator_settings' keys to the theory parameters.

# The 'emulate' key is a boolean which states if the theory should be emulated or not.

# The 'emulator_settings' key is a dictionary which contains the settings for the emulator. If not given the default settings are used which can be found in the corresponding OLE python files.
info = {
    
        "likelihood": {
            # planck 
            "planck_2018_highl_plik.TTTEEE":{},
            "planck_2018_lensing.clik":{},
            "planck_2018_lowl.TT_clik":{},
            "planck_2018_lowl.EE_clik":{},
            "bao.desi_2024_bao_all": {},
            # "bao.sdss_dr7_mgs": {},
            # "bao.sdss_dr12_consensus_bao": {},
            # "sn.pantheon": {},
        },



        "theory": {
               'camb': {
                   'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat',
                                    'halofit_version': 'mead',
                                    'lens_potential_accuracy': 1,
                                    'nnu': 3.044,
                                    'lmax': 3151,
                                    'theta_H0_range': [20, 100]
                                },
    
                'emulate' : True, 
                'ignore_obsolete': True,

                'emulator_settings': {
                    # accuracy parameters for loglike:
                    'quality_threshold_constant': 1.0,
                    'quality_threshold_linear': 0.1,

                    # uncommend for some nice debugging plots (makes OLE very slow)
                    # 'plotting_directory': data_dir+'/OLE_exercices/plots',
                    # 'testset_fraction': 0.1,

                    # veto to predict following quantities. 
                    # The emulator does not know intrinsicially which parameters are expected to be computed since it is build based upon a general cobaya state.
                    # This does include quantities which are in fact not used or not changing during the MCMC. We can manually veto them such that they are not predicted by the emulator.
                    'skip_emulation_quantities': ['TCMB']
                },            
            },
        },
        'params': {
            'logA': {'drop': True,
                     'latex': '\\log(10^{10} A_\\mathrm{s})',
                     'prior': {'max': 3.91, 'min': 1.61},
                     'proposal': 0.001,
                     'ref': {'dist': 'norm', 'loc': 3.05, 'scale': 0.001}},
            'ns': {'latex': 'n_\\mathrm{s}',
                   'prior': {'max': 1.2, 'min': 0.8},
                   'proposal': 0.002,
                   'ref': {'dist': 'norm', 'loc': 0.965, 'scale': 0.004}},
            'ombh2': {'latex': '\\Omega_\\mathrm{b} h^2',
                      'prior': {'max': 0.1, 'min': 0.005},
                      'proposal': 0.0001,
                      'ref': {'dist': 'norm',
                              'loc': 0.0224,
                              'scale': 0.0001},
                    },
            'omch2': {'latex': '\\Omega_\\mathrm{c} h^2',
                      'prior': {'max': 0.99, 'min': 0.001},
                      'proposal': 0.0005,
                      'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}},
            'tau': {'latex': '\\tau_\\mathrm{reio}',
                    'prior': {'max': 0.8, 'min': 0.01},
                    'proposal': 0.003,
                    'ref': {'dist': 'norm', 'loc': 0.055, 'scale': 0.006}},
            'H0': {'latex': 'H_0' , 
                   'prior':{'max': 100, 'min': 20},
                     'proposal': 0.5,
                        'ref': {'dist': 'norm', 'loc': 70, 'scale': 1}},
            'As': {'latex': 'A_\\mathrm{s}',
                   'value': 'lambda logA: 1e-10*np.exp(logA)'
                   },
            'A': {'derived': 'lambda As: 1e9*As',
                  'latex': '10^9 A_\\mathrm{s}'
                  },
        },

    "sampler": { 
        "mcmc": {
            # YOU NEED TO MANUALY SET BLOCKING WHEN USING CAMB. Otherwise the emulator will not work.
            'blocking': [ [1, ['ombh2', 'omch2', 'tau', 'H0', 'logA', 'ns', 'A_planck', 'calib_100T', 'calib_217T', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ksz_norm', 'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217', 'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217']]],
            },
        },
        'output': data_dir+'/OLE_exercices/chains_emulator/CAMB',
        'debug': False,
        'force': False,
        'resume': True,
        }

# run MCMC
updated_info, sampler = cobaya.run(info)







