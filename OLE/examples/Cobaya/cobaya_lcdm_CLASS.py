# This is an example of a cobaya runfile for a simple LCDM model.
#
import importlib
import importlib.util
import sys
import os

from OLE.interfaces.cobaya_interface import *


# Define your default info dictionary of cobaya.
# This dictionary contains the likelihoods, the theory, the parameters, the sampler, and the output settings.

# Additionally to the vanilla cobaya info dictionary, we have added the 'emulate' and 'emulator_settings' keys to the theory parameters.

# The 'emulate' key is a boolean which states if the theory should be emulated or not.

info = {
    
        "likelihood": {
            # planck lite
            "planck_2018_highl_plik.TTTEEE":{},
            "planck_2018_lensing.clik":{},
            "planck_2018_lowl.TT_clik":{},
            "planck_2018_lowl.EE_clik":{},
            "bao.sixdf_2011_bao": {},
            "bao.sdss_dr7_mgs": {},
            "bao.sdss_dr12_consensus_bao": {},
            "sn.pantheon": {},
        },

        "theory": {
            "classy": {
                "extra_args": {
                    # default CLASS settings:
                    "output": "tCl,pCl,lCl,mPk",
                    "non linear": "halofit",
                    "lensing":"yes",
                    "compute damping scale":"yes",
                    'N_ncdm' : 1,
                    'm_ncdm' : 0.06,
                    'T_ncdm' : 0.71611,
                    'N_ur': 2.0328,
                    # "N_ur": 3.044,
                },
                'emulate' : True, 
                'ignore_obsolete': True,

                'emulator_settings': {
                    # accuracy parameters for loglike:
                    'quality_threshold_constant': 0.1,
                    'quality_threshold_linear': 0.05,

                    # plotting directory. Uncomment for debugging and to create plots.
                    # 'plotting_directory': 'plots',
                    # 'testset_fraction': 0.1,

                    # veto to predict following quantities. 
                    # The emulator does not know intrinsicially which parameters are expected to be computed since it is build based upon a general cobaya state.
                    # This does include quantities which are in fact not used or not changing during the MCMC. We can manually veto them such that they are not predicted by the emulator.
                    'skip_emulation_quantities': ['T_cmb', 'bb', 'tp', 'ell']
                },            
            },
        },


        'params': {
            'A': {'derived': 'lambda A_s: 1e9*A_s',
                'latex': '10^9 A_\\mathrm{s}'},
            'A_s': {'latex': 'A_\\mathrm{s}',
                    'value': 'lambda logA: 1e-10*np.exp(logA)'},
            'h': {'latex': 'h',
                'prior': {'max': 1.0, 'min': 0.4},
                'proposal': 0.01,     
                'ref': {'dist': 'norm', 'loc': 0.68104431, 'scale': 0.001}
                },
            'clamp': {'derived': 'lambda A_s, tau_reio: '
                                '1e9*A_s*np.exp(-2*tau_reio)',
                    'latex': '10^9 A_\\mathrm{s} e^{-2\\tau}'},
            'logA': {'drop': True,
                    'latex': '\\log(10^{10} A_\\mathrm{s})',
                    'prior': {'max': 3.257, 'min': 2.837},
                    'proposal': 0.02,   
                    'ref': {'dist': 'norm', 'loc': 3.0486233, 'scale': 0.002}
                    },
            'n_s': {'latex': 'n_\\mathrm{s}',
                    'prior': {'max': 1.0235, 'min': 0.9095},
                    'proposal': 0.004,    
                    'ref': {'dist': 'norm', 'loc': 0.96854107, 'scale': 0.0004}
                    },
            'omega_b': {'latex': '\\Omega_\\mathrm{b} h^2',
                        'prior': {'max': 0.02452, 'min': 0.02032},
                        'proposal': 0.0002,
                        'ref': {'dist': 'norm','loc': 0.022191874,'scale': 0.00002}
                        },
            'omega_cdm': {'latex': '\\omega_\\mathrm{cdm} ',
                        'prior': {'max': 0.1329, 'min': 0.1057},
                        'proposal': 0.003,
                        'ref': {'dist': 'norm', 'loc': 0.11886091, 'scale': 0.0003}
                        },
            'sigma8': {'latex': '\\sigma_8'},
            'tau_reio': {'latex': '\\tau_\\mathrm{reio}',
                        'prior': {'max': 0.08449, 'min': 0.0276},
                        'proposal': 0.01,   #053075671
                        'ref': {'dist': 'norm', 'loc': 0.059658148, 'scale': 0.001}
                        },
        },
    "sampler": { 
        "mcmc": {
            "Rminus1_stop": 0.01,
            "covmat": os.path.expanduser("./lcdm.covmat"),

            # This function automaticially switches off oversampling once the emulator is trained.
            "callback_function": OLE_callback_function,
            "callback_every": 1,
            },
        },
        'output': 'chains_emulator/test_class',
        'debug': False,
        'force': False,
        'resume':True,
        }






# run MCMC
updated_info, sampler = cobaya.run(info)

getdist_available = False
# make some nice triangle plot
try:
    import getdist.plots as gdplt
    import getdist
    getdist_available = True
except:
    print('getdist not installed, no analysis done')

if getdist_available:
    gdplot = gdplt.get_subplot_plotter()

    # load samples from ./chains
    samples = getdist.loadMCSamples('chains/test_1', settings={'ignore_rows': 0.1})

    # make triangle plot
    gdplot.triangle_plot(samples, ['h', 'logA', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'A_planck'], filled=True)

    # store figure
    gdplot.export('triangle_plot.png')





