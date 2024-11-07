# This file contains a sampler class that interacts with the emulator, the theory code and the likelihood.
# The NUTS sampler is based upon jaxnuts: https://github.com/guillaume-plc/jaxnuts
from OLE.utils.base import BaseClass
from OLE.utils.mpi import *
from OLE.theory import Theory
from OLE.likelihood import Likelihood
from OLE.emulator import Emulator
from OLE.utils.mpi import *
from OLE.plotting import data_covmat_plot, covmat_diagonal_plot
from scipy.optimize import minimize

from functools import partial
from typing import Tuple
import jax.numpy as jnp
import jax
import fasteners
from jax.numpy import ndarray
import jax.lax as lax
import random
import copy
import os

import numpy as np
import jax.numpy as jnp

import jax.random as random

import time 

# The sampler class connects the theory, the likelihood, and the emulator.
# It can be also used to interact with the emulator also without giving a specific theory oder likelihood instance.

class Sampler(BaseClass):
    """
    Base sampler class. It connects the theory, the likelihood, and the emulator. It can be also used to interact with the emulator also without giving a specific theory oder likelihood instance.

    Key features are:
    - Initialize the sampler with the theory, the likelihood, and the emulator.
    - Transform/Retransform the parameters into the normalized eigenspace.
    - Generate the covariance matrix.
    - Implement priors.
    - Interacts with the emulator.



    Attributes
    --------------
    theory : Theory
        The theory instance.
    likelihood_collection : Dictionary of all used likelihoods. Has form of {name: likelihood_instance, name2: likelihood_instance2,...}
        The likelihood instance.
    emulator : Emulator
        The emulator instance.
    emulator_settings : dict
        The settings for the emulator. Will be passed to the emulator when intialized.
    likelihood_collection_settings : dict
        The settings for the likelihood. Will be passed to the likelihood when intialized. has the form of {name: likelihood_settings, name2: likelihood_settings2,...}
    theory_settings : dict
        The settings for the theory. Will be passed to the theory when intialized.
    sampling_settings : dict
        The settings for the sampler. Will be passed to the sampler.

    Methods
    --------------
    initialize :
        Initialises the sampler. It takes the parameters, the likelihood, the theory, the emulator, and the settings as input.
    update_covmat :
        Updates the covariance matrix.

    """

    theory: Theory
    likelihood: dict
    emulator_settings: dict
    likelihood_collection_settings: dict
    theory_settings: dict
    sampling_settings: dict

    emulator: Emulator

    
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
    
    def initialize(self, 
                   parameters, 
                   likelihood_collection=None, 
                   theory=None,
                   emulator=None, 
                   emulator_settings={},
                   likelihood_collection_settings={},
                    theory_settings={},
                    sampling_settings={},
                   **kwargs):

        """
        Initialises the sampler. It takes the parameters, the likelihood, the theory, the emulator, and the settings as input.

        Parameters
        --------------
        parameters : dict
            A dictionary containing the parameters and their properties such as priors, proposals, etc.
        likelihood_collection : dictionary of Likelihoods
            The likelihood instance.
        theory : Theory
            The theory instance.
        emulator : Emulator
            The emulator instance.
        emulator_settings : dict
            The settings for the emulator. Will be passed to the emulator when intialized.
        likelihood_settings : dict
            The settings for the likelihood. Will be passed to the likelihood when intialized.
        theory_settings : dict
            The settings for the theory. Will be passed to the theory when intialized.
        sampling_settings : dict
            The settings for the sampler. Will be passed to the sampler.
        **kwargs : dict
            Catch all rubbish.

        """

        super().initialize(**kwargs)

        # store kwargs
        self.kwargs = kwargs

        # Save all the settings
        self.emulator_settings = emulator_settings
        self.likelihood_collection_settings = likelihood_collection_settings
        self.theory_settings = theory_settings
        self.sampling_settings = sampling_settings

        # Store Likelihood and initialize if possible
        self.likelihood_collection = likelihood_collection
        if self.likelihood_collection is not None:
            for key in self.likelihood_collection.keys():
                if not self.likelihood_collection[key].initialized:
                    self.likelihood_collection[key].initialize(**self.likelihood_collection_settings[key])

        # update theory settings by likelihood
        if theory is not None:
            for key in self.likelihood_collection.keys():
                theory_settings = self.likelihood_collection[key].update_theory_settings(theory_settings)

        # Store Theory and initialize if possible
        self.theory = theory
        if self.theory is not None:
            self.theory_settings = theory_settings
            self.theory.initialize(**theory_settings)

        # Store Emulator and initialize if possible
        self.emulator = None
        if emulator is not None:
            self.emulator = emulator

        # load default parameter dictionary from likelihood collection
        if self.likelihood_collection is not None:
            self.parameter_dict = {}
            for key in self.likelihood_collection.keys():
                self.parameter_dict.update(self.likelihood_collection[key].nuisance_sample_dict)

        # Update on parameters if given
        if parameters is not None:
            self.parameter_dict.update(parameters) 

        # we gonna split all parameters into constant and varying parameters. Together they will be in the parameter_dict_full
        self.parameter_dict_full = copy.deepcopy(self.parameter_dict)

        # move the constant parameters that have the attribute 'value' to self.parameter_dict_constant
        self.parameter_dict_constant = {}
        for key in list(self.parameter_dict.keys()):
            if 'value' in list(self.parameter_dict[key].keys()):
                self.parameter_dict_constant[key] = self.parameter_dict[key]
                # self.parameter_dict.pop(key)

        # remove the parameters from self.parameter_dict
        for key in self.parameter_dict_constant.keys():
            self.parameter_dict.pop(key)

        defaulthyperparameters = {
            # output directory for the chain
            'output_directory': 'output',

            # force overwrite
            'force': False, #TODO: implement this

            # load (parameter) covmat from file from a path
            'covmat': None,

            # compute data covmat. This only works for differentiable likelihoods
            'compute_data_covmat': False,

            # plotting directory
            'plotting_directory': None,

            # status print frequency
            'status_print_frequency': 100,

            # logfile
            'logfile': None,

            # flag to use the emulator for the likelihood
            'use_emulator': True,

            # n_restart for the minimizer
            'n_restarts': 1,

        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in sampling_settings.items():
            self.hyperparameters[key] = value

        # create output directory
        import os
        if not os.path.exists(self.hyperparameters['output_directory']):
            os.makedirs(self.hyperparameters['output_directory'])

        # save proposal lengths. They are useful to normalize the parameters
        self.proposal_lengths = jnp.ones(len(self.parameter_dict))
        self.proposal_means = jnp.zeros(len(self.parameter_dict))
        for i, key in enumerate(self.parameter_dict.keys()):
            if type(self.parameter_dict[key]) is list:
                continue 
            elif 'proposal' in list(self.parameter_dict[key].keys()):
                self.proposal_lengths = self.proposal_lengths.at[i].set(self.parameter_dict[key]['proposal'])
            else:
                raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)
            
            if 'ref' in list(self.parameter_dict[key].keys()):
                self.proposal_means = self.proposal_means.at[i].set(self.parameter_dict[key]['ref']['mean'])
            else:
                self.proposal_means = self.proposal_means.at[i].set(0.5*(self.parameter_dict[key]['prior']['min']+self.parameter_dict[key]['prior']['max']))

        # hessian of the likelihood
        self.hessian_function = None

        # generate the covariance matrix
        self.covmat = self.generate_covmat()

        # go into eigenspace of covmat and build the inverse of the eigenvectors
        self.eigenvalues, self.eigenvectors = jnp.linalg.eigh(self.covmat)
        self.inv_eigenvectors = jnp.linalg.inv(self.eigenvectors)

        # we can now transform the parameters into the (normalized) eigenspace

        # remove the parameters from the test state which are not in self.theory.required_parameters

        # the likelihood will be initialized once again in the emulator. As a consequence we need to provide the emulator settings with the likelihood settings
        emulator_settings['likelihood_collection_settings'] = self.likelihood_collection_settings

        # if emulator was not loaded, we need to create one here:
        if self.emulator is None and self.hyperparameters['use_emulator']:
            # initialize the emulator
            self.emulator = Emulator(**emulator_settings)

            test_state = None

            if 'load_initial_state' not in emulator_settings:
                test_state = self.test_pipeline()
            else:
                if not emulator_settings['load_initial_state']:
                    test_state = self.test_pipeline()

            # here we check the required_parameters of the theory code. If they are specified, we initialize the emulator with the required_parameters
            # otherwise the emulator is inificalized with all parameters which could also include likelihood parameters!
            if len(self.theory.required_parameters()) > 0:
                self.emulator.initialize(self.likelihood_collection, test_state, input_parameters=self.theory.required_parameters(), **emulator_settings)
            else:
                self.emulator.initialize(self.likelihood_collection, test_state, **emulator_settings)


        self.nuisance_parameters = list(self.parameter_dict.keys())
        #remove keys which are in self.theory.required_parameters
        for key in self.theory.required_parameters():
            if key in self.nuisance_parameters:
                self.nuisance_parameters.remove(key)

        if len(self.nuisance_parameters) != 0:
            # get nuisance mean and std from the parameter_dict
            self.nuisance_means = jnp.zeros(len(self.nuisance_parameters))
            self.nuisance_stds = jnp.zeros(len(self.nuisance_parameters))

            for i, key in enumerate(self.nuisance_parameters):
                self.nuisance_means = self.nuisance_means.at[i].set(self.parameter_dict[key]['ref']['mean'])
                self.nuisance_stds = self.nuisance_stds.at[i].set(self.parameter_dict[key]['ref']['std'])

        self.parameter_names = list(self.parameter_dict_full.keys())

        pass

    def update_covmat(self, covmat):
        """
        Updates the parameter covariance matrix and computed eigenvalues and eigenvectors in order to transform the parameters into the normalized eigenspace. 
        It also checks for negative eigenvalues and nans or infs in the matrix.

        Parameters
        --------------
        covmat : ndarray
            The new covariance matrix.

        """
        # this function updates the covmat
        self.covmat = covmat

        # go into eigenspace of covmat and build the inverse of the eigenvectors
        self.eigenvalues, self.eigenvectors = jnp.linalg.eigh(self.covmat)

        # search for negative eigenvalues
        if jnp.any(self.eigenvalues < 0):
            self.error("Covmat contains negative eigenvalues")
            raise ValueError("Covmat contains negative eigenvalues")
        
        # search for nan or inf in the eigenvalues and eigenvectors\
        if jnp.isnan(self.eigenvalues).any() or jnp.isinf(self.eigenvalues).any():
            self.error("Covmat contains nans or infs")
            raise ValueError("Covmat contains nans or infs")
        
        if jnp.isnan(self.eigenvectors).any() or jnp.isinf(self.eigenvectors).any():
            self.error("Covmat contains nans or infs")
            raise ValueError("Covmat contains nans or infs")

        self.inv_eigenvectors = self.eigenvectors.T        

        return 0

    def print_status(self, i, chain):
        if i==0:
            return 0
        # this function computes the effective sample size and prints the status of the chain after the i-th interation. This is only done when i%100==0
        if ((i) % self.hyperparameters['status_print_frequency']) == 0:
            ess = self.estimate_effective_sample_size(chain[:i,:])
            mean_ess = np.mean(ess)
            self.info("Iteration %d: Mean ESS = %f", i, mean_ess)
            self.info("Iteration %d: Mean ESS/sek = %f", i, mean_ess/self.time_from_start())
            
        return 0

    def estimate_effective_sample_size(self, chain):
        """
        This function estimates the effective sample size of the chain. It is computed for each parameter.

        Parameters
        --------------
        chain : ndarray
            The current status of the chain.

        Returns
        --------------
        ndarray :
            The effective sample size for each parameter.
        """
        ESS = np.zeros(chain.shape[1])

        for i in range(chain.shape[1]):
            mean = np.mean(chain[:,i])
            autocorrelation_1 = np.sum((chain[1:,i]-mean)*(chain[:-1,i]-mean))/np.sum((chain[:,i]-mean)**2)

            ESS[i] = len(chain) / ( (1 + autocorrelation_1)/(1-autocorrelation_1) )

        return ESS

    def denormalize_inv_hessematrix(self, matrix):
        """ 
        This function denormalizes the inverse hessian matrix according to the eigenspace.
        """
        # this function denormalizes the matrix
        a = jnp.dot(self.eigenvectors,  
                    jnp.dot( 
                        jnp.dot(
                            jnp.dot( jnp.diag(jnp.sqrt(self.eigenvalues)),matrix), jnp.diag(jnp.sqrt(self.eigenvalues))), self.eigenvectors.T))

        return a

    def transform_parameters_into_normalized_eigenspace(self, parameters):
        """ 
        This function transforms the parameters into the normalized eigenspace.
        """
        # this function transforms the parameters into the normalized eigenspace
        return jnp.dot( self.inv_eigenvectors, ( parameters - self.proposal_means ))/jnp.sqrt(self.eigenvalues)
    
    def retranform_parameters_from_normalized_eigenspace(self, parameters):
        """
        This function transforms the parameters back from the normalized eigenspace.
        """
        # this function transforms the parameters back from the normalized eigenspace
        return jnp.dot(self.eigenvectors, parameters * jnp.sqrt(self.eigenvalues) ) + self.proposal_means

    def generate_covmat(self):
        """
        This function generates the covariance matrix either by loading it from a file or by using the proposal lengths. 
        It is called during the initialization of the sampler.
        Note that the covmat might miss some entries which are to be filled with the proposal lengths.

        Returns
        --------------
        ndarray :
            The covariance matrix.
        """

        
        # first: create diagonal covmat from proposal lengths
        covmat = np.zeros((len(self.parameter_dict), len(self.parameter_dict)))
        for i in range(len(self.parameter_dict)):
            covmat[i,i] = self.proposal_lengths[i]**2

        # second if the covmat is given (at least for some parameters), we need to fill the covmat with the given values
        if self.hyperparameters['covmat'] is not None:
            # load the covmat from a file. The first line is the strings of the parameters, the rest is the covmat
            with open(self.hyperparameters['covmat'], 'r') as f:
                lines = f.readlines()
                parameters = lines[0].split()
                covmat_loaded = np.zeros((len(parameters), len(parameters)))
                for i, line in enumerate(lines[1:]):
                    covmat_loaded[i] = np.array([float(_) for _ in line.split()])

            self.info("Loaded covmat from file: %s", self.hyperparameters['covmat'])
            self.info("Loaded parameters: %s", parameters)

            # fill the covmat with the given values
            for i, key in enumerate(self.parameter_dict.keys()):
                if key in parameters:
                    for j, key2 in enumerate(self.parameter_dict.keys()):
                        if key2 in parameters:
                            covmat[j,i] = covmat_loaded[parameters.index(key2), parameters.index(key)]
        
        # create the covmat from the proposal lengths
        return covmat

    def get_initial_position(self, N=1, normalized=False):
        """
        This function returns N initial positions for the sampler according to the parameter informations. 
        It samples from the 'ref' values of the parameters.

        Parameters
        --------------
        N : int
            The number of initial positions.
        normalized : bool
            If True, the parameters are returned in the normalized eigenspace.

        Returns
        --------------
        ndarray :
            The initial positions.
        """
        positions = []
        for i in range(N):
            position = []
            for key, value in self.parameter_dict.items():
                if 'value' in list(value.keys()):
                    position.append(value['value'])
                elif 'ref' in list(value.keys()):
                    # create candidate from normal distribution with mean and std from the 'ref' values but ensure that it is within the prior
                    while True:
                        candidate = value['ref']['mean'] + value['ref']['std']*np.random.normal()
                        if candidate > value['prior']['min'] and candidate < value['prior']['max']:
                            break
                    position.append(candidate)
                elif 'prior' in list(value.keys()):
                    position.append(value['prior']['min'] + (value['prior']['max']-value['prior']['min'])*np.random.uniform())
                else:
                    raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)
            positions.append(position)

        if normalized:
            for i in range(N):
                positions[i] = self.transform_parameters_into_normalized_eigenspace(np.array(positions[i]))
            return positions
        else:
            return np.array(positions)

    def get_bounds(self, normalized=False):
        """
        This function returns the bounds of the parameters.
        It is used for the minimizer.
        """

        bounds = []
        for key, value in self.parameter_dict.items():
            if 'prior' in list(value.keys()):
                bounds.append((value['prior']['min'], value['prior']['max']))
            else:
                raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)
        
        if normalized:
            lower_bound = np.array([_[0] for _ in bounds])
            upper_bound = np.array([_[1] for _ in bounds])

            normalized_bounds = np.vstack([self.transform_parameters_into_normalized_eigenspace(lower_bound),
                                           self.transform_parameters_into_normalized_eigenspace(upper_bound)])
            # put the bounds in the right order\
            bounds = []
            for i in range(len(normalized_bounds[0])):
                bounds.append((normalized_bounds[0,i], normalized_bounds[1,i]))

            return bounds
        else:
            return bounds

    def test_pipeline(self):
        """
        This function creates a test state from the given parameters.
        It is used to test the pipeline when initializing the sampler in order to ensure that the emulator, the theory, and the likelihood are working correctly.
        """
        # Create a test state from the given parameters.
        state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

        RNG = jax.random.PRNGKey(int(time.time())+get_mpi_rank())

        # translate the parameters to the state
        for key, value in self.parameter_dict_full.items():
            RNG, subkey = jax.random.split(RNG)
            if 'value' in list(value.keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict[key]['value']])
            elif 'ref' in list(value.keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict[key]['ref']['mean'] + self.parameter_dict[key]['ref']['std']*jax.random.normal(subkey)])
            elif 'prior' in list(value.keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict[key]['prior']['min'] + (self.parameter_dict[key]['prior']['max']-self.parameter_dict[key]['prior']['min'])*jax.random.uniform(subkey)]) # 
            else:
                raise ValueError("Parameter %s is not defined correctly. Please check the parameter_dict."%key)

        # compute the observables
        state = self.theory.compute(state)

        # compute the loglikelihood
        for key in self.likelihood_collection.keys():
            state = self.likelihood_collection[key].loglike_state(state)

        # compute the total loglike
        logprior = self.compute_logprior(state)
        state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior
            

        self.debug("Test pipeline:")
        self.debug(state)

        return state


    def calculate_data_covmat(self, parameters):
        """
        This function computes the hessian of the likelihood with respect to a certain point in parameter space. 
        This can be interpreded as the data covariance matrix.
        The hessian is calculated by jax.hessian.
        Note that this can only be done for differentiable likelihoods.
        Additionally, it tends to be numerically unstable for large parameter spaces.

        It is particularly useful for the emulator to normalize the data by the data covariance matrix.
        For example data points with large uncertainties are less important for the emulator training.

        Parameters
        --------------
        parameters : ndarray
            The parameters for which the hessian is calculated.

        Returns
        --------------
        dict :
            The data covariance matrix for each observable.
        """

        # Run the emulator to compute the observables.
        state = self.emulator.emulate(parameters)

        data_covmats = {}

        # for each observable we create a function that computes the hessian with respect to the observable.
        for observable in state['quantities'].keys():

            observable_values = jnp.array(state['quantities'][observable])

            # function to compute likelihood as a function of the observable
            def hessian_observable(x):
                local_state = copy.deepcopy(state)
                local_state['quantities'][observable] = x
                ll = 0.0
                for likelihood in self.likelihood_collection.keys():
                    ll += self.likelihood_collection[likelihood].loglike_state(local_state)['loglike'][observable]

                return -ll

            # compute hessian of observables
            observable_hessian = jnp.asarray(jax.hessian(hessian_observable)(observable_values))

            # find entries which do not constiute to the loglike
            diag_hessian = jnp.diag(observable_hessian)

            mask = np.zeros(len(observable_values))
            mask[diag_hessian!=0.0]=1

            # set index = indices where mask is 1
            index = np.where(mask==1)[0]
            indices = jnp.meshgrid(index, index)

            # observable_hessian is a n times n matrix. We need to remove the entries which do not contribute to the loglike
            original_shape = observable_hessian.shape

            observable_covmat = jnp.linalg.inv(observable_hessian[mask==1][:,mask==1])

            # create the data_covmat
            data_covmats[observable] = jnp.zeros(original_shape)
            data_covmats[observable] = data_covmats[observable].at[indices[0],indices[1]].set(observable_covmat)

            # check for nans or infs
            if jnp.isnan(data_covmats[observable]).any() or jnp.isinf(data_covmats[observable]).any():
                self.error("Data covariance matrix for observable %s contains nans or infs", observable)
                raise ValueError("Data covariance matrix for observable %s contains nans or infs"%observable)

            # store the diagonal of the data_covmat as txt file in the output directory
            if not os.path.exists(self.hyperparameters['output_directory']+'/data_covmats'):
                os.makedirs(self.hyperparameters['output_directory']+'/data_covmats')
            np.savetxt(self.hyperparameters['output_directory']+'/data_covmats/'+observable+'.txt', jnp.diag(data_covmats[observable]))

        if self.hyperparameters['plotting_directory'] is not None:
            for observable in state['quantities'].keys():
                # check if plotting directory exists, otherwise create it
                if not os.path.exists(self.hyperparameters['plotting_directory']+'/data_covmats'):
                    os.makedirs(self.hyperparameters['plotting_directory']+'/data_covmats')

                data_covmat_plot(data_covmats[observable], 'data covariance matrix '+ observable, self.hyperparameters['plotting_directory']+'/data_covmats/'+observable+'.png')

                covmat_diagonal_plot(data_covmats[observable], 'data diagonal covariance matrix '+ observable, self.hyperparameters['plotting_directory']+'/data_covmats/'+observable+'_diagonal.png')


        return data_covmats     

    # This function computes the logposteriors for given parameters.
    # If possible it uses the emulator to speed up the computation.
    # If the emulator is not trained or the emulator is not good enough, the theory code is used.
    def compute_logposterior_from_normalized_parameters(self, parameters):
        """
        This function computes the logposteriors for given normalized parameters.
        If possible it uses the emulator to speed up the computation, otherwise the theory code is used.

        It checks the emulator for its performance and decides whether to use the emulator or the theory code.
        Eventually, the logposterior is computed and returned.

        Parameters
        --------------
        parameters : ndarray
            The parameters for which the logposterior is computed.

        Returns
        --------------
        float :
            The logposterior.
        """
        self.start()

        RNGkey = jax.random.PRNGKey(int(time.clock_gettime_ns(0)))

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)
        
        # every 100 self.n calls, state the n
        if self.n % 100 == 0:
            self.info("Current sampler call: %d", self.n)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # add constant parameters to the state
        for key in self.parameter_dict_constant.keys():
            if 'value' in list(self.parameter_dict_constant[key].keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict_constant[key]['value']])

        self.debug("Calculating loglike for parameters: %s", state['parameters'])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])

        # compute the observables. First check whether emulator is already trained
        if (not self.hyperparameters['use_emulator']) or (not self.emulator.trained):

            self.debug("emulator not trained yet -> use theory code")
            state = self.theory.compute(state)
            self.debug("state after theory: %s for parameters: %s", state['quantities'], state['parameters'])
            
            for likelihood in self.likelihood_collection.keys():
                state = self.likelihood_collection[likelihood].loglike_state(state)
            self.debug("loglikes after theory: %s for parameters: %s", state['loglike'], state['parameters'])

            state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior
            if self.hyperparameters['use_emulator']:
                self.emulator.add_state(state)
        else:
            # here we need to test the emulator for its performance
            emulator_sample_states, RNGkey = self.emulator.emulate_samples(state['parameters'], RNGkey=RNGkey,noise=1)
            emulator_sample_loglikes = jnp.zeros(len(emulator_sample_states))
            for i, emulator_sample_state in enumerate(emulator_sample_states):
                for likelihood in self.likelihood_collection.keys():
                    emulator_sample_state = self.likelihood_collection[likelihood].loglike_state(emulator_sample_state)
                emulator_sample_loglikes.at[i].set(jnp.array(list(emulator_sample_state['loglike'].values())).sum())

            # check whether the emulator is good enough
            if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=state['parameters']):
                state = self.theory.compute(state)
                for likelihood in self.likelihood_collection.keys():
                    state = self.likelihood_collection[likelihood].loglike_state(state)
                state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior
                self.emulator.add_state(state)
            else:
                # Add the point to the quality points
                self.emulator.add_quality_point(state['parameters'])
            
                self.debug("emulator available - check emulation performance")
                state = self.emulator.emulate(state['parameters'])
                self.debug("state after emulator: %s for parameters: %s", state['quantities'], state['parameters'])

                for likelihood in self.likelihood_collection.keys():
                    state = self.likelihood_collection[likelihood].loglike_state(state)
                self.debug("loglike after theory: %s for parameters: %s", state['loglike'], state['parameters'])

                state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior
                self.debug("emulator prediction: %s", state['quantities'])


        # if we have a minimal number of states in the cache, we can train the emulator
        if self.hyperparameters['use_emulator']:
            if (len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']) and not self.emulator.trained:
                self.debug("Training emulator")
                self.emulator.train()
                self.debug("Emulator trained")

        self.increment(self.logger)

        if type(state['total_loglike']) != jax._src.interpreters.ad.JVPTracer:
            _ = "Logposterior: "+ str(state['total_loglike']) + ' at ' + " ".join([str(key) + ": " + str(value[0]) for key, value in state['parameters'].items()]) + "\n"
            self.write_to_log(_)

        return state['total_loglike']


    def compute_total_logposterior_from_normalized_parameters(self, parameters):
        """
        This function computes the total loglikelihood for given normalized parameters.
        It sums up the loglikelihoods for each observable.
        It calls the function compute_logposterior_from_normalized_parameters for each observable.

        Parameters
        --------------
        parameters : ndarray
            The parameters for which the loglikelihood is computed.

        Returns
        --------------
        float :
            The total loglikelihood.
        """
        res = self.compute_logposterior_from_normalized_parameters(parameters)
        return res.sum()

    def compute_theory_from_normalized_parameters(self, parameters):
        """
        This function computes only the theory (and thus the observational data) for given normalized parameters.
        It is used in the NUTS sampler to accelerate to initial burn-in phase by minimizing the nuisance parameters for the theory code.

        Parameters
        --------------
        parameters : ndarray
            The normalized parameters for which the theory is computed.

        Returns
        --------------
        dict :
            The state after the theory computation.
        """
        self.start()

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        self.debug("Calculating loglike for parameters: %s", state['parameters'])

        # compute the observables. First check whether emulator is already trained
        if not self.emulator.trained:

            self.debug("emulator not trained yet -> use theory code")
            state = self.theory.compute(state)
            self.debug("state after theory: %s for parameters: %s", state['quantities'], state['parameters'])
        else:
            # here we need to test the emulator for its performance
            emulator_sample_states, RNGkey = self.emulator.emulate_samples(state['parameters'],RNGkey=RNGkey , noise=1)

            emulator_sample_loglikes = jnp.zeros(len(emulator_sample_states))
            for i, emulator_sample_state in enumerate(emulator_sample_states):
                for likelihood in self.likelihood_collection.keys():
                    emulator_sample_state = self.likelihood_collection[likelihood].loglike_state(emulator_sample_state)
                emulator_sample_loglikes.at[i].set(jnp.array(list(emulator_sample_state['loglike'].values())).sum())

            # check whether the emulator is good enough
            if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=state['parameters']):
                state = self.theory.compute(state)
            else:
                # Add the point to the quality points
                self.emulator.add_quality_point(state['parameters'])
            
                self.debug("emulator available - check emulation performance")
                state = self.emulator.emulate(state['parameters'])
                self.debug("state after emulator: %s for parameters: %s", state['quantities'], state['parameters'])

        # if we have a minimal number of states in the cache, we can train the emulator
        if (len(self.emulator.data_cache.states) >= self.emulator.hyperparameters['min_data_points']) and not self.emulator.trained:
            self.debug("Training emulator")
            self.emulator.train()
            self.debug("Emulator trained")

        self.increment(self.logger)        
        return state
    
    def logposterior_function(self, parameter_values, local_state):
        """ 
        This function computes the loglikelihoods for given parameters and a given theory state.
        It is used in the minimization of the nuisance parameters for the NUTS sampler.

        Parameters
        --------------
        parameter_values : ndarray
            The parameters for which the loglikelihood is computed.
        local_state : dict
            The state of the theory code.

        Returns
        --------------
        float :
            The loglikelihood.
        """
        for i, key in enumerate(self.nuisance_parameters):
            local_state['parameters'][key] = jnp.array([parameter_values[i]])*self.nuisance_stds[i] + self.nuisance_means[i]
        
        for key in self.likelihood_collection.keys():
            local_state = self.likelihood_collection[key].loglike_state(local_state)

        logprior = self.compute_logprior(local_state)
        local_state['total_loglike'] = jnp.array(list(local_state['loglike'].values())).sum() + logprior

        if type(local_state['total_loglike']) != jax._src.interpreters.ad.JVPTracer:
            _ = "Logposterior: "+ str(local_state['loglike']) + ' at ' + " ".join([str(key) + ": " + str(value[0]) for key, value in local_state['parameters'].items()]) + "\n"
            self.write_to_log(_)
        
        return -local_state['total_loglike'].sum()
    
    
    def emulate_logposterior_from_normalized_parameters_differentiable(self, parameters_norm):
        """ 
        This function emulates the logposteriors for given parameters.
        It does not automatically add the state to the emulator. Thus it is differentiable.

        Parameters
        --------------
        parameters_norm : ndarray
            The normalized parameters for which the logposterior is computed.

        Returns
        --------------
        float :
            The logposterior.
        """

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters_norm)

        return self.emulate_logposterior_from_parameters_differentiable(parameters)

    def emulate_logposterior_from_parameters_differentiable(self, parameters):
        """ 
        This function emulates the logposteriors for given parameters.
        It does not automatically add the state to the emulator. Thus it is differentiable.

        Parameters
        --------------
        parameters : ndarray
            The parameters for which the logposterior is computed.

        Returns
        --------------
        float :
            The logposterior.
        """
        self.start()

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # add constant parameters to the state
        for key in self.parameter_dict_constant.keys():
            if 'value' in list(self.parameter_dict_constant[key].keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict_constant[key]['value']])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])

        # make jax lax cond function that if the logprior is < -1e20, the loglike is -1e20, else the loglike is computed
        def full_loglike(state):
            self.debug("emulator available - check emulation performance")
            state = self.emulator.emulate_jit(state['parameters'])
            self.debug("state after emulator: %s for parameters: %s", state['quantities'], state['parameters'])

            for likelihood in self.likelihood_collection.keys():
                state = self.likelihood_collection[likelihood].loglike_state(state)
            self.debug("loglike after theory: %s for parameters: %s", state['loglike'], state['parameters'])

            state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior
            self.debug("emulator prediction: %s", state['quantities'])
            return state['total_loglike']
        
        def full_logprior(state):
            logprior = self.compute_logprior_with_border(state)

            # stitch parameters to the allowed parameter space
            stitched_parameters = self.stitch_parameters(state['parameters'])

            # compute the observables
            state = self.emulator.emulate_jit(stitched_parameters)
            for likelihood in self.likelihood_collection.keys():
                state = self.likelihood_collection[likelihood].loglike_state(state)

            state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior

            return state['total_loglike']
        
        state['total_loglike'] = lax.cond(logprior < -1e20, full_logprior, full_loglike, state)

        parameters = self.transform_parameters_into_normalized_eigenspace(parameters)

        self.increment(self.logger)
        
        if type(state['total_loglike']) != jax._src.interpreters.ad.JVPTracer:
            _ = "Logposterior: "+ str(state['total_loglike']) + ' at ' + " ".join([str(key) + ": " + str(value[0]) for key, value in state['parameters'].items()]) + "\n"
            self.write_to_log(_)

        return state['total_loglike']

    # This function emulates the loglikelihoods for given parameters.
    def emulate_loglikelihood_from_parameters_differentiable(self, parameters):
        """
        This function emulates the loglikelihoods for given parameters. Note that it does not consider the posterior.
        It does not automatically add the state to the emulator. Thus it is differentiable.

        Parameters
        --------------
        parameters : ndarray
            The parameters for which the loglikelihood is computed.

        Returns
        --------------
        float :
            The loglikelihood.
        """
        self.start()

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # add constant parameters to the state
        for key in self.parameter_dict_constant.keys():
            if 'value' in list(self.parameter_dict_constant[key].keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict_constant[key]['value']])

        state = self.emulator.emulate_jit(state['parameters'])
        for likelihood in self.likelihood_collection.keys():
            state = self.likelihood_collection[likelihood].loglike_state(state)
        state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum()
        self.debug("loglike after theory: %f for parameters: %s", state['total_loglike'], state['parameters'])

        self.increment(self.logger)

        if type(state['total_loglike']) != jax._src.interpreters.ad.JVPTracer:
            _ = "Logposterior: "+ str(state['total_loglike']) + ' at ' + " ".join([str(key) + ": " + str(value[0]) for key, value in state['parameters'].items()]) + "\n"
            self.write_to_log(_)
        
        return state['total_loglike']


    def sample_emulate_logposterior_from_normalized_parameters_differentiable(self, parameters, N=1, RNGkey=jax.random.PRNGKey(int(time.time())), noise = 0.):
        """
        This function samples the logposteriors for given normalized parameters from the emulator in order to test its performance.

        Parameters
        --------------
        parameters : ndarray
            The normalized parameters for which the logposterior is computed.
        N : int
            The number of samples.
        RNGkey : jax.random.PRNGKey
            The random key for the sampling.
        noise : float
            The noise level for the sampling.

        Returns
        --------------
        ndarray :
            The logposteriors for the states.
        """

        # rescale the parameters
        parameters = self.retranform_parameters_from_normalized_eigenspace(parameters)

        loglikes = self.sample_emulate_logposterior_from_parameters_differentiable(parameters, N=N, RNGkey=RNGkey, noise = noise)
        
        return loglikes



    def sample_emulate_logposterior_from_parameters_differentiable(self, parameters, N=1, RNGkey=jax.random.PRNGKey(int(time.time())), noise = 0.):
        """
        This function samples the logposteriors for given parameters from the emulator in order to test its performance.

        Parameters
        --------------
        parameters : ndarray
            The normalized parameters for which the logposterior is computed.
        N : int
            The number of samples.
        RNGkey : jax.random.PRNGKey
            The random key for the sampling.
        noise : float
            The noise level for the sampling.

        Returns
        --------------
        ndarray :
            The logposteriors for the states.
        """

        self.start()

        # Run the sampler.
        state = {'parameters': {}, 'quantities': {}, 'loglike': {}, 'total_loglike': None}

        # translate the parameters to the state
        for i, key in enumerate(self.parameter_dict.keys()):
            state['parameters'][key] = jnp.array([parameters[i]])

        # add constant parameters to the state
        for key in self.parameter_dict_constant.keys():
            if 'value' in list(self.parameter_dict_constant[key].keys()):
                state['parameters'][key] = jnp.array([self.parameter_dict_constant[key]['value']])

        # compute logprior
        logprior = self.compute_logprior(state)
        self.debug("logprior: %f for parameters: %s", logprior, state['parameters'])

        self.debug("emulator available - check emulation performance")

        states, RNGkey = self.emulator.emulate_samples_jit(state['parameters'], RNGkey, noise = noise)

        loglikes = jnp.zeros(N)
        for i in range(N):
            state = states[i]
            for likelihood in self.likelihood_collection.keys():
                state = self.likelihood_collection[likelihood].loglike_state(state)
            state['total_loglike'] = jnp.array(list(state['loglike'].values())).sum() + logprior
            loglikes = loglikes.at[i].set(state['total_loglike'])

        self.increment(self.logger)

        # Currently not working. TODO: SG: Fix that
        # if type(states[0]['loglike'][0]) != jax._src.interpreters.ad.JVPTracer:
        #     _ = "Sampled Logposterior: " + " ".join(str([_[0] for _ in loglikes])) + ' at ' + " ".join([str(key) + ": " + str(value[0]) for key, value in states[0]['parameters'].items()]) + "\n"
        #     self.write_to_log(_)
        
        return loglikes

    
    def emulate_total_logposterior_from_normalized_parameters_differentiable(self, parameters):
        """
        This function emulates the total loglikelihood for given normalized parameters.
        """
        res = self.emulate_logposterior_from_normalized_parameters_differentiable(parameters)
        return res
    
    def sample_emulate_total_logposterior_from_normalized_parameters_differentiable(self, parameters, noise = 0):
        """
        This function samples the total loglikelihood for given normalized parameters from the emulator in order to test its performance.
        """
        N = self.emulator.hyperparameters['N_quality_samples']
        res = [_.sum() for _ in self.sample_emulate_logposterior_from_normalized_parameters_differentiable(parameters, N=N , noise = noise)]
        return res
    
    def sample_emulate_total_logposterior_from_parameters_differentiable(self, parameters, noise = 0):
        """
        This function samples the total loglikelihood for given parameters from the emulator in order to test its performance.
        """
        N = self.emulator.hyperparameters['N_quality_samples']
        res = [_.sum() for _ in self.sample_emulate_logposterior_from_parameters_differentiable(parameters, N=N , noise = noise)]
        return res

    def emulate_total_minuslogposterior_from_normalized_parameters_differentiable(self, parameters):
        """ 
        This function emulates the total minus logposterior for given normalized parameters.
        """
        res = -self.emulate_logposterior_from_normalized_parameters_differentiable(parameters)
        return res.sum()
    
    def emulate_total_minusloglikelihood_from_parameters_differentiable(self, parameters):
        """
        This function emulates the total minus loglikelihood for given parameters.
        """
        res = -self.emulate_loglikelihood_from_parameters_differentiable(parameters)
        return res.sum()
    
    def compute_total_minuslogposterior_from_normalized_parameters(self, parameters):
        """
        This function computes the total minus logposterior for given normalized parameters.
        """
        res = -self.compute_logposterior_from_normalized_parameters(parameters)
        return res.sum()
    
    def compute_logprior(self, state):
        """
        This function computes the logprior for a given state.
        It uses the parameter_dict to compute the logprior.
        If the prior is not defined, it uses a flat prior.
        Possible priors are: 'uniform', 'gaussian', 'jeffreys', log-normal, etc.

        Parameters
        --------------
        state : dict
            The state for which the logprior is computed.

        Returns
        --------------
        float :
            The logprior.
        """
        log_prior = 0.0
        for key, value in self.parameter_dict.items():

            # check if 'type' is in prior. TODO: SG: remove that. Make type mandatory
            if 'type' not in value['prior'].keys():
                log_prior+= jnp.log(1.0/(value['prior']['max']-value['prior']['min']))
                log_prior -= jnp.heaviside(value['prior']['min'] - state['parameters'][key][0],  1.0) * 99999999999999999999999.  + jnp.heaviside(state['parameters'][key][0] - value['prior']['max'], 1.0) * 99999999999999999999999.
            else:
                if value['prior']['type'] == 'uniform':
                    # if mean and std are given, we use a gaussian prior
                    log_prior+= jnp.log(1.0/(value['prior']['max']-value['prior']['min']))
                    log_prior -= jnp.heaviside(value['prior']['min'] - state['parameters'][key][0],  1.0) * 99999999999999999999999.  + jnp.heaviside(state['parameters'][key][0] - value['prior']['max'], 1.0) * 99999999999999999999999.
            
                if value['prior']['type'] == 'gaussian':
                    log_prior += -0.5*(state['parameters'][key][0]-value['prior']['mean'])**2/value['prior']['std']**2 - 0.5*jnp.log(2*jnp.pi*value['prior']['std']**2)
                    log_prior -= jnp.heaviside(value['prior']['min'] - state['parameters'][key][0],  1.0) * 99999999999999999999999.  + jnp.heaviside(state['parameters'][key][0] - value['prior']['max'], 1.0) * 99999999999999999999999.
        
                if value['prior']['type'] == 'jeffreys':
                # first we need to compute the fisher information matrix. 
                # If the emulator is not trained yet, we use the estimate from the covmat.
                    if not self.emulator.trained:
                        fisher_information = self.covmat
                    else:
                        if self.hessian_function is None:
                            params = jnp.array([state['parameters'][_][0] for _ in state['parameters'].keys()])
                            self.hessian_function = jax.jit(jax.hessian(self.emulate_total_minusloglikelihood_from_parameters_differentiable))
                            fisher_information = self.hessian_function(params)
                        else:
                            params = jnp.array([state['parameters'][_][0] for _ in state['parameters'].keys()])
                            fisher_information = self.hessian_function(params)

                    determinant = jnp.linalg.det(fisher_information)
                    det_min = 0.01

                    log_prior = jnp.log(jnp.sqrt(jnp.abs(determinant)))

                if value['prior']['type'] == 'log-normal':
                    # Note tested yet! TODO: test this!
                    self.warning("Log-normal prior not tested yet!")
                    log_prior += -0.5*(jnp.log(state['parameters'][key][0])-value['prior']['mean'])**2/value['prior']['std']**2 - 0.5*jnp.log(2*jnp.pi*value['prior']['std']**2)
                    log_prior -= jnp.heaviside(value['prior']['min'] - state['parameters'][key][0],  1.0) * 99999999999999999999999.  + jnp.heaviside(state['parameters'][key][0] - value['prior']['max'], 1.0) * 99999999999999999999999.


        return log_prior
    
    def compute_logprior_with_border(self, state):
        """ 
        This function computes the logprior for a given state.
        It uses the parameter_dict to compute the logprior.
        If the prior is not defined, it uses a flat prior.
        Possible priors are: 'uniform', 'gaussian', 'jeffreys', log-normal, etc.
        If we exceed the borders of the prior, we project the parameters to the border and add a penalty term to the logprior.

        Parameters
        --------------
        state : dict
            The state for which the logprior is computed.

        Returns
        --------------
        float :
            The logprior.
        """
        log_prior = 0.0

        for key, value in self.parameter_dict.items():

            # if mean and std are given, we use a gaussian prior
            if 'mean' in value['prior'].keys():
                log_prior += -0.5*(state['parameters'][key][0]-value['prior']['mean'])**2/value['prior']['std']**2 - 0.5*jnp.log(2*jnp.pi*value['prior']['std']**2)
            # else we sticke with the flat prior
            else:
                log_prior+= jnp.log(1.0/(value['prior']['max']-value['prior']['min']))

            # add penalty term if we exceed the borders, which is linear in the distance to the border
            distance_to_border = jnp.minimum(state['parameters'][key][0]-value['prior']['min'], value['prior']['max']-state['parameters'][key][0]) / value['proposal']
            
            # with jax lax cond we can add the penalty term only if we exceed the borders
            def add_penalty_term():
                c_lin = 1000.0 # penalty term
                return distance_to_border * c_lin
            
            log_prior += lax.cond(jnp.logical_or(state['parameters'][key][0] < value['prior']['min'], state['parameters'][key][0] > value['prior']['max']), add_penalty_term, lambda: 0.0)

        return log_prior

    def stitch_parameters(self, parameters):
        """
        This function stitches the parameters to the allowed parameter space.
        If the parameters exceed the borders, they are projected to the border.

        Parameters
        --------------
        parameters : dict
            The parameters for which the logprior is computed.

        Returns
        --------------
        dict :
            The stitched parameters.
        """
        stitched_parameters = copy.deepcopy(parameters)
        for key, value in self.parameter_dict.items():
            if 'prior' in list(value.keys()):
                stitched_parameters[key] = jnp.minimum(jnp.maximum(parameters[key], value['prior']['min']), value['prior']['max'])
        return stitched_parameters
    

    def write_to_log(self, message):
        # write the message to the logfile
        if self.hyperparameters['logfile'] is None:
            return
        
        file_name = self.hyperparameters['logfile']+'_'+str(get_mpi_rank())+'.log'
        # check if file/directory exists, otherwise create it
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'a') as logfile:
            # add timestamp to message
            message = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + message
            logfile.write(message)

        pass

    
    def sample(self):
        # Run the sampler.
        
        return 
