
from typing import Sequence, Tuple, Union
import numpy as np
import jax.numpy as jnp

import time

import candl
import candl.data


# Since we would like to use the cobaya intrinsic 'boltzmannbase' theory, we need to adapt the source code of the cobaya intrinsic 'boltzmannbase' theory to use the OLE theory interface.
# 
# For all other cases, e.g. we define our own cobaya - Theory class, we can use the OLE theory interface directly.
# We do this by replacing "from cobaya.theory import Theory" by "from OLE.interfaces.cobaya_interface import Theory
import importlib
import importlib.util
import sys
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




# build fake planck lite
from cobaya.likelihood import Likelihood
from cobaya.typing import InfoDict



class my_likelihood(Likelihood):
    def initialize(self, **kwargs):

        self.use_likelihood = 'ACT_DR4_TTTEEE'
        
        # define your candl likelihood
        if self.use_likelihood == 'SPT3G_2018_TTTEEE':
            self.candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)
            self.use_cl = ["TT", "EE", "TE"]
        elif self.use_likelihood == 'ACT_DR4_TTTEEE':
            self.candl_like = candl.Like(candl.data.ACT_DR4_TTTEEE)
            self.use_cl = ["TT", "EE", "TE"]

    def logp(self, _derived, **params_values):
        # Compute the loglikelihood for the given parameters.

        Cls = self.provider.get_Cl(ell_factor=False, units='FIRASK2')



        if self.use_likelihood == 'SPT3G_2018_TTTEEE':
            input_keys = ['EE_Poisson_150x150', 'EE_Poisson_150x220', 'EE_Poisson_220x220', 'EE_Poisson_90x150', 'EE_Poisson_90x220', 'EE_Poisson_90x90', 'EE_PolGalDust_Alpha', 'EE_PolGalDust_Amp', 'EE_PolGalDust_Beta', 'Ecal150', 'Ecal220', 'Ecal90', 'Kappa', 'TE_PolGalDust_Alpha', 'TE_PolGalDust_Amp', 'TE_PolGalDust_Beta', 'TT_CIBClustering_Amp', 'TT_CIBClustering_Beta', 'TT_GalCirrus_Alpha', 'TT_GalCirrus_Amp', 'TT_GalCirrus_Beta', 'TT_Poisson_150x150', 'TT_Poisson_150x220', 'TT_Poisson_220x220', 'TT_Poisson_90x150', 'TT_Poisson_90x220', 'TT_Poisson_90x90', 'TT_kSZ_Amp', 'TT_tSZ_Amp', 'TT_tSZ_CIB_Corr_Amp', 'Tcal150', 'Tcal220', 'Tcal90']
            ell = jnp.float32(jnp.arange(3199)+2)
        elif self.use_likelihood == 'ACT_DR4_TTTEEE':
            input_keys = ['yp']
            ell = jnp.float32(jnp.arange(7924)+2)

        Dl = {'TT': Cls.get('tt')[2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi) /2.7255**2,
              'TE': Cls.get('te')[2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi) /2.7255**2,
              'EE': Cls.get('ee')[2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi) /2.7255**2,
            #   'BB': state['quantities']['bb'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
              'ell': ell}

        candl_input = {key: params_values[key] for key in params_values}
        candl_input['Dl'] = Dl
        candl_input['tau'] = params_values['tau_reio']

        loglike = self.candl_like.log_like(candl_input)

        return [loglike]
    
    def get_requirements(self):
        # State requisites to the theory code
        if self.use_likelihood == 'SPT3G_2018_TTTEEE':
            input_keys = ['tau_reio','EE_Poisson_150x150', 'EE_Poisson_150x220', 'EE_Poisson_220x220', 'EE_Poisson_90x150', 'EE_Poisson_90x220', 'EE_Poisson_90x90', 'EE_PolGalDust_Alpha', 'EE_PolGalDust_Amp', 'EE_PolGalDust_Beta', 'Ecal150', 'Ecal220', 'Ecal90', 'Kappa', 'TE_PolGalDust_Alpha', 'TE_PolGalDust_Amp', 'TE_PolGalDust_Beta', 'TT_CIBClustering_Amp', 'TT_CIBClustering_Beta', 'TT_GalCirrus_Alpha', 'TT_GalCirrus_Amp', 'TT_GalCirrus_Beta', 'TT_Poisson_150x150', 'TT_Poisson_150x220', 'TT_Poisson_220x220', 'TT_Poisson_90x150', 'TT_Poisson_90x220', 'TT_Poisson_90x90', 'TT_kSZ_Amp', 'TT_tSZ_Amp', 'TT_tSZ_CIB_Corr_Amp', 'Tcal150', 'Tcal220', 'Tcal90']
            ell = jnp.float32(jnp.arange(3199)+2)
            self.l_max = 3200
        elif self.use_likelihood == 'ACT_DR4_TTTEEE':
            input_keys = ['tau_reio','yp']
            ell = jnp.float32(jnp.arange(7924)+2)
            self.l_max = 7925

        req = {"Cl": {cl: self.l_max for cl in self.use_cl}}

        req.update({key: None for key in input_keys})

        return req    

my_like = my_likelihood()




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


info = {
        

        "likelihood": {
            # candl lite
            "my_like": {
                'external': my_like,
            },

            
            # "planck_2018_highl_plik.TTTEEE_lite":{},
            # "planck_2018_lowl.TT_clik":{},
            # "planck_2018_lowl.EE_clik":{},
            # "bao.sixdf_2011_bao": {},
            # "bao.sdss_dr7_mgs": {},
            # "bao.sdss_dr12_consensus_bao": {},
            # "sn.pantheon": {},
        },

        "theory": {
            "classy": {
                "extra_args": {
                    'output': 'tCl, pCl, lCl',
                    'l_max_scalars': 3200,
                    'lensing': 'yes',
                    'N_ur': 3.046,
                    'non linear': 'halofit',
                },

                'requires': ['tau_reio'],

                'emulate' : True, 
                'ignore_obsolete': True,

                'emulator_settings': {
                    'min_data_points': 80,

                    # dimensionality of mcmc
                    'dimensionality': 7,

                    # N sigma for cache
                    'N_sigma': 4.0,

                    # name of the cache file
                    'cache_file': './cobaya_act_out_OLE/cache.pkl',
                    'output_directory': './cobaya_act_out_OLE/output',

                    'debug': False,

                    # accuracy parameters for loglike:
                    'quality_threshold_constant': 0.5,
                    'quality_threshold_linear': 0.1,

                    # number of quality states to estimate from
                    'N_quality_samples': 5,
                    'logfile': './cobaya_act_out_OLE/logfile.txt',

                    # plotting dir
                    'plotting_directory': './cobaya_act_out_OLE/plots',
                    'testset_fraction': 0.1,


                    # veto to predict following quantities. 
                    # The emulator does not know intrinsicially which parameters are expected to be computed since it is build based upon a general cobaya state.
                    # This does include quantities which are in fact not used or not changing during the MCMC. We can manually veto them such that they are not predicted by the emulator.
                    'veto_list': ['T_cmb', 'pp', 'bb', 'tp', 'ell']
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
                        'prior': {'max': 0.1, 'min': 0.01},
                        'proposal': 0.01,   #053075671
                        'ref': {'dist': 'norm', 'loc': 0.058, 'scale': 0.001}
                        },


            # nuisance parameters of the ACT likelihood
            'yp': {'prior': {'min': 0.9, 'max': 1.1}, 'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.0001}, 'proposal': 0.0001},


        },
    "sampler": { 
        "mcmc": {
            "drag":False,
            "learn_proposal": True,
            "oversample_power": 0.4,
            "proposal_scale":2.1,
            "Rminus1_stop": 0.05,
            "max_tries": 24000,
            #"covmat": os.path.expanduser("./chains_emu/test_class.covmat"),
            "blocking":
                [[1, ["logA", "n_s", "omega_b", "omega_cdm", "tau_reio", "h",]],
                 [1, ["yp"]]],
            },
        },
        'output': './cobaya_act_out_OLE/spt',
        # 'debug': True,
        'force': True,
        'resume':False,
        'test': False,
        }


start = time.time()
updated_info, sampler = cobaya.run(info)
end = time.time()
print("Time elapsed: ", end - start)

# import getdist.plots as gdplt
# import getdist

# print(sampler.products())

# gdplot = gdplt.get_subplot_plotter()

# # load samples from ./chains
# samples = getdist.loadMCSamples('chains/test_1', settings={'ignore_rows': 0.1})

# # make triangle plot
# gdplot.triangle_plot(samples, ['x1', 'x2', 'x3'], filled=True)

# # store figure
# gdplot.export('triangle_plot.png')





