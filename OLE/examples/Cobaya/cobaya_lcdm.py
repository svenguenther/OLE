# This is an example of a cobaya runfile for a simple LCDM model.
#




import importlib
import importlib.util
import sys

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
            "classy": {
                "extra_args": {
                    # default CLASS settings:
                    "output": "tCl,pCl,lCl,mPk",
                    "non linear": "halofit",
                    "lensing":"yes",
                    "compute damping scale":"yes",
                    #'N_ncdm' : 1,
                    #'m_ncdm' : 0.06,
                    #'T_ncdm' : 0.71611,
                    "N_ur": 3.044,
                },
                'emulate' : True, 
                'ignore_obsolete': True,

                'emulator_settings': {
                    'min_data_points': 40,
                    'kernel': 'RBF',
                    'output_directory': 'chains_emulator/output',
                    'force': False,

                    # name of the cache file
                    'cache_file': './chains_emulator/cache.pkl',

                    # load the cache from the cache file
                    'load_cache': True,

                    'debug': False,

                    'quality_points_radius': 0.5,

                    'delta_loglike': 300,

                    # accuracy parameters for loglike:
                    'quality_threshold_constant': 0.1,
                    'quality_threshold_linear': 0.1,
                    'quality_threshold_quadratic': 0.01,  

                    # number of quality states to estimate from
                    'N_quality_samples': 2,

                    # quality radius for the emulator
                    'quality_points_radius': 0.0,

                    # the number of PCA components to use is determined by the explained variance. We require a minimum of 99.9% explained variance.
                    'explained_variance_cutoff': 0.99,
                    'max_output_dimensions': 10, 

                    #'plotting_directory': './plots',

                    # related to the training
                    'num_iters': 60,

                    'logfile': 'chains_emulator/logfile.txt',

                    # veto to predict following quantities. 
                    # The emulator does not know intrinsicially which parameters are expected to be computed since it is build based upon a general cobaya state.
                    # This does include quantities which are in fact not used or not changing during the MCMC. We can manually veto them such that they are not predicted by the emulator.
                    'veto_list': ['T_cmb', 'bb', 'tp', 'ell']
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
                'ref': {'dist': 'norm', 'loc': 0.699, 'scale': 0.001}
                },
            'clamp': {'derived': 'lambda A_s, tau_reio: '
                                '1e9*A_s*np.exp(-2*tau_reio)',
                    'latex': '10^9 A_\\mathrm{s} e^{-2\\tau}'},
            'logA': {'drop': True,
                    'latex': '\\log(10^{10} A_\\mathrm{s})',
                    'prior': {'max': 3.257, 'min': 2.837},
                    'proposal': 0.02,   
                    'ref': {'dist': 'norm', 'loc': 3.046, 'scale': 0.002}
                    },
            'n_s': {'latex': 'n_\\mathrm{s}',
                    'prior': {'max': 1.0235, 'min': 0.9095},
                    'proposal': 0.004,    
                    'ref': {'dist': 'norm', 'loc': 0.967, 'scale': 0.0004}
                    },
            'omega_b': {'latex': '\\Omega_\\mathrm{b} h^2',
                        'prior': {'max': 0.02452, 'min': 0.02032},
                        'proposal': 0.0002,
                        'ref': {'dist': 'norm','loc': 0.0225,'scale': 0.00002}
                        },
            'omega_cdm': {'latex': '\\omega_\\mathrm{cdm} ',
                        'prior': {'max': 0.1329, 'min': 0.1057},
                        'proposal': 0.003,
                        'ref': {'dist': 'norm', 'loc': 0.123, 'scale': 0.003}
                        },
            'sigma8': {'latex': '\\sigma_8'},
            'tau_reio': {'latex': '\\tau_\\mathrm{reio}',
                        'prior': {'max': 0.08449, 'min': 0.0276},
                        'proposal': 0.01,   #053075671
                        'ref': {'dist': 'norm', 'loc': 0.058, 'scale': 0.001}
                        },
        },
    "sampler": { 
        "mcmc": {
            "drag":False,
            "learn_proposal": True,
            #"oversample_power": 0.0,
            "proposal_scale":2.1,
            "Rminus1_stop": 0.01,
            "max_tries": 24000,
            #"covmat": os.path.expanduser("./lcdm.covmat"),
            },
        },
        'output': 'chains_emulator/test_class',
        'debug': False,
        'force': True,
        'resume':False,
        }






# run MCMC
updated_info, sampler = cobaya.run(info)


# make some nice triangle plot
import getdist.plots as gdplt
import getdist

gdplot = gdplt.get_subplot_plotter()

# load samples from ./chains
samples = getdist.loadMCSamples('chains/test_1', settings={'ignore_rows': 0.1})

# make triangle plot
gdplot.triangle_plot(samples, ['h', 'logA', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'A_planck'], filled=True)

# store figure
gdplot.export('triangle_plot.png')





