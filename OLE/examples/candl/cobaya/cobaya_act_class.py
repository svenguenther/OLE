
from typing import Sequence, Tuple, Union
from cobaya.typing import InfoDict
import numpy as np
import jax.numpy as jnp

import time

import candl
import candl.data
import cobaya


# build fake planck lite
from cobaya.likelihood import Likelihood

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

        # Do cross check hier :)  

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

# turn off all logging
# logging.disable(logging.INFO)

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
            # planck lite
            "my_like":my_like,

            
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
                    'l_max_scalars': 7250,
                    'lensing': 'yes',
                    'N_ur': 3.046,
                    'non linear': 'halofit',

                },
            'requires': ['tau_reio'],
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
                 [16, ["yp"]]],
            },
        },
        'output': './cobaya_act_out/spt',
        'debug': False,
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





