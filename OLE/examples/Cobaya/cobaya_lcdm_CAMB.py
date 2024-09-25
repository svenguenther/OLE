# This is an example of a cobaya runfile for a simple LCDM model.
#
import importlib
import importlib.util
import sys
import os
import warnings
# from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


#import tensorflow as tf
#
## Check the number of intra-op threads
#intra_op_threads = tf.config.threading.get_intra_op_parallelism_threads()
#print(f"TF_NUM_INTRAOP_THREADS: {intra_op_threads}")
#
## Check the number of inter-op threads
#inter_op_threads = tf.config.threading.get_inter_op_parallelism_threads()
#print(f"TF_NUM_INTEROP_THREADS: {inter_op_threads}")

data_dir = os.getcwd()
# Since we would like to use hte cobaya intrinsic 'boltzmannbase' theory, we need to adapt the source code of the cobaya intrinsic 'boltzmannbase' theory to use the OLE theory interface.
# 
# For all other cases, e.g. we define our own cobaya - Theory class, we can use the OLE theory interface directly.
# We do this by replacing "from cobaya.theory import Theory" by "from OLE.interfaces.cobaya_interface import Theory

spec = importlib.util.find_spec('cobaya.theories.cosmo.boltzmannbase', 'cobaya')
source = spec.loader.get_source('cobaya.theories.cosmo.boltzmannbase')

# write replace function to let boltzmannbase not import 'from cobaya.theory import Theory', but 'from OLE.theory import Theory'
def replace(source):
    source = source.replace("from cobaya.theory import Theory", "from OLE.interfaces.cobaya_interface import Theory")
    return source

source = replace(source)
module = importlib.util.module_from_spec(spec)
codeobj = compile(source, module.__spec__.origin, 'exec')
exec(codeobj, module.__dict__)
sys.modules['cobaya'].theories.cosmo.boltzmannbase = module

import cobaya 
cobaya.theories.cosmo.boltzmannbase = module
importlib.reload(cobaya)

# Additionally, due to some incompatibility of the logging systems, we do need to manually silence some of the logging output of the jax and matplotlib libraries.
import logging
logger = logging.getLogger('root')
logger.disabled = True
logger = logging.getLogger('jax._src.dispatch')
logger.disabled = True
logger = logging.getLogger('jax._src.compiler')
logger.disabled = True
logger = logging.getLogger('jax.experimental.host_callback')
logger.disabled = True
logger = logging.getLogger('jax._src.xla_bridge')
logger.disabled = True
logger = logging.getLogger('jax._src.interpreters.pxla')
logger.disabled = True
logger = logging.getLogger('matplotlib.font_manager')
logger.disabled = True
logger = logging.getLogger('jax._src.interpreters.pxla')
logger.disabled = True

# Define your default info dictionary of cobaya.
# This dictionary contains the likelihoods, the theory, the parameters, the sampler, and the output settings.

# Additionally to the vanilla cobaya info dictionary, we have added the 'emulate' and 'emulator_settings' keys to the theory parameters.

# The 'emulate' key is a boolean which states if the theory should be emulated or not.

# The 'emulator_settings' key is a dictionary which contains the settings for the emulator. If not given the default settings are used which can be found in the corresponding OLE python files.
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
               'camb': {
                   'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat',
                                    'halofit_version': 'mead',
                                    'lens_potential_accuracy': 1,
                                    'nnu': 3.044,
                                    'mnu': 0.06,
                                    'lmax': 6000,
                                    'num_massive_neutrinos': 1,
                                    'theta_H0_range': [20, 100]
                                },
    
                'emulate' : True, 
                'ignore_obsolete': True,

                'emulator_settings': {
                    'min_data_points': 80,

                    # the input parameters of the emulator
                    'output_directory': data_dir+'/OLE_exercices/chains_emulator/output',

                    # name of the cache file
                    'cache_file': data_dir+'/OLE_exercices/chains_emulator/cache.pkl',

                    # load the cache from the cache file
                    'load_cache': True,

                    'N_sigma': 4,
                    'dimensionality': 27,

                    # accuracy parameters for loglike:
                    'quality_threshold_constant': 0.5,
                    'quality_threshold_linear': 0.1,

                    # number of quality states to estimate from
                    'N_quality_samples': 5,


                    # the number of PCA components to use is determined by the explained variance. We require a minimum of 99.9% explained variance.
                    'max_output_dimensions': 10, 

                    # 'plotting_directory': data_dir+'/OLE_exercices/plots',
                    # 'testset_fraction': 0.1,
                    'logfile': data_dir+'/OLE_exercices/logfile',

                    # veto to predict following quantities. 
                    # The emulator does not know intrinsicially which parameters are expected to be computed since it is build based upon a general cobaya state.
                    # This does include quantities which are in fact not used or not changing during the MCMC. We can manually veto them such that they are not predicted by the emulator.
                    'skip_emulation_quantities': ['TCMB', 'dependency_params','bb']
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
            'Rminus1_cl_stop': 0.2,
                      'Rminus1_stop': 0.01,
                      "learn_proposal": True,
                    #   'covmat': '/home/guenther/software/projects/OLE/OLE/OLE/examples/Cobaya/lcdm.covmat',
                      'drag': False,
                      "measure_speeds": True,
                      #'oversample_power': 0.4,
                      'blocking': [ [1, ['ombh2', 'omch2', 'tau', 'H0', 'logA', 'ns']], 
                                    [6, ['A_planck', 'calib_100T', 'calib_217T', 'A_cib_217', 'xi_sz_cib', 'A_sz', 'ksz_norm', 'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217', 'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217', 'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217']]],
                      'proposal_scale': 1.9,
                      'output_every': 1
      
            },
        },
        'output': data_dir+'/OLE_exercices/Test/CAMB',
        'debug': False,
        'force': True,
        'resume': False,
        }






# run MCMC
updated_info, sampler = cobaya.run(info)







