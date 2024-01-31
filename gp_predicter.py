# Here we will program the gp_predicter class. One instance of this class will be created for each quantity that is to be emulated.
# Each quantity may have different number of dimensions. Thus, each gp_predicter requires a different preprocessing and potentially PCA data compression.
# 

# use GPJax to fit the data
from jax.config import config

config.update("jax_enable_x64", True)

from jax import jit, random
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox

global RANDOM_KEY 
RANDOM_KEY = random.PRNGKey(42)
global SUB_KEY
SUB_KEY = 0

from jax import grad

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

import time

# Path: OLE/gp_predicter.py

from .utils.base import BaseClass
from .data_processing import data_processor

from .interfaces import gpjax_interface

class GP_predictor(BaseClass):

    quantity_name: str

    hyperparameters: dict

    GPs: list

    def __init__(self, quantity_name=None, **kwargs):
        super().__init__('GP '+quantity_name, **kwargs)
        self.quantity_name = quantity_name

    def initialize(self, ini_state, **kwargs):
        # in the ini_state and example state is given which contains the parameters and the quantities which are to be emulated. The example state also contains an example value for each quantity which is used to determine the dimensionality of the quantity.


        # default hyperparameters
        defaulthyperparameters = {
            'kernel': 'RBF',

            # plotting directory
            'plotting_directory': None,

        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        # We need to determine the dimensionality of the quantity to be emulated. This is done by looking at the example state.
        self.output_size = len(ini_state['quantities'][self.quantity_name])
        self.info('Output size: %d', self.output_size)

        # We need to determine the dimensionality of the parameters. This is done by looking at the example state.
        self.input_size = len(ini_state['parameters'])
        self.info('Input size: %d', self.input_size)

        # We can now initialize the data processor for this quantity.
        self.data_processor = data_processor('Data processor ' + self.quantity_name, debug=self.debug_mode)
        self.data_processor.initialize(self.input_size, self.output_size, self.quantity_name,**kwargs)

        pass

    def predict(self, parameters):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros(len(self.GPs))
        for i in range(len(self.GPs)):
            output_data_compressed = output_data_compressed.at[i].set(self.GPs[i].predict(parameters_normalized))    # this is the time consuming part

        # Untransform the output data.
        output_data_normalized = self.data_processor.decompress_data(output_data_compressed)


        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)
        
        return output_data
    
    def sample_prediction(self, parameters, N):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros((N, len(self.GPs)))
        for i in range(len(self.GPs)):
            output_data_compressed = output_data_compressed.at[:,[i]].set(self.GPs[i].sample(parameters_normalized, N).transpose())    # this is the time consuming part

        # Untransform the output data.
        output_data_normalized = jnp.array([self.data_processor.decompress_data(output_data_compressed[i,:]) for i in range(N)])

        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)
        return output_data
    
    def predict_gradients(self, parameters):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = np.zeros((len(self.GPs), self.input_size))

        for i in range(len(self.GPs)):
            output_data_compressed[i] = self.GPs[i].predict_gradient(parameters_normalized.copy())

        output_data_compressed = jnp.array(output_data_compressed)

        # data out
        data_out = np.zeros((self.output_size, self.input_size))

        # note that in order to get the gradients we have to scale it twice with input and output normalization
        
        for i in range(self.input_size):
            data_out[:,i] = self.data_processor.decompress_data(output_data_compressed[:,i]) 
            data_out[:,i] /= self.data_processor.input_stds[i]

        for j in range(self.output_size):
            data_out[j,:] *= self.data_processor.output_stds[j]
        
        return data_out.T
    
    def train(self):
        # Train the GP emulator.
        input_data = self.data_processor.input_data_normalized
        output_data = self.data_processor.output_data_emulator

        # For each dimension of the output_data we create a GP.
        self.GPs = []
        for i in range(output_data.shape[1]):
            # Create a GP for each dimension of the output_data.
            self.GPs.append(GP('GP '+self.quantity_name+' dim '+str(i), **self.hyperparameters))

            # Load the data into the GP.
            self.GPs[i].load_data(input_data, output_data[:,i])

            # Train the GP.
            self.GPs[i].train()

        pass

    def train_single_GP(self, input_data, output_data):
        # Train the GP emulator.
        D = gpx.Dataset(input_data, output_data)

        pass

    def load_data(self, input_data_raw, output_data_raw):
        # Load the raw data from the data cache.
        self.data_processor.load_data(input_data_raw, output_data_raw)
        pass

    def set_parameters(self, parameters):
        # Set the parameters of the emulator.
        self.data_processor.set_parameters(parameters)
        pass




class GP(BaseClass):

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)

        # default hyperparameters
        defaulthyperparameters = {
            # Kernel type
            'kernel': 'RBF',
            # Exponential decay learning rate
            'learning_rate': 0.02,
            # Number of iterations
            'num_iters': 100,
        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        # Flag that indicates whether it is required to recompute the kernel matrix (Kxx) of the training data set.
        self.recompute_kernel_matrix = False
        self.Kxx = None

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        pass

    def load_data(self, input_data, output_data):
        # Load the data from the data processor.
        self.recompute_kernel_matrix = True
        self.input_data = input_data
        self.output_data = output_data[:,None]
        self.D = gpx.Dataset(self.input_data, self.output_data)
        pass

    def train(self):
        # Train the GP emulator.
        
        # Create the kernel
        if self.hyperparameters['kernel'] == 'RBF':
            kernel = gpx.kernels.RBF() #+ gpx.kernels.White()
        else:
            raise ValueError('Kernel not implemented')
        
        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

        # Create the likelihood
        likelihood = gpx.gps.Gaussian(num_datapoints=self.D.n)
        

        posterior = prior * likelihood

        negative_mll = gpx.objectives.ConjugateMLL(negative=True)
        negative_mll(posterior, train_data=self.D)

        negative_mll = jit(negative_mll)

        # have exponential decay learning rate
        lr = lambda t: jnp.exp(-self.hyperparameters['learning_rate']*t)

        # fit
        self.opt_posterior, history = gpx.fit(
            model=posterior,
            objective=negative_mll,
            train_data=self.D,
            optim=ox.adam(learning_rate=lr),
            num_iters=self.hyperparameters['num_iters'],
            safe=True,
            key=jr.PRNGKey(0),
        )


        # DEEEEEBUG
        # predict on the training data
        # latent_dist = self.opt_posterior.predict(self.D.X, train_data=self.D)  
        # predictive_dist = self.opt_posterior.likelihood(latent_dist)

        # predictive_std = predictive_dist.stddev()
        # predictive_mean = predictive_dist.mean()

        # Kxx = self.opt_posterior.compute_Kxx(self.D)
        # ac = self.opt_posterior.calculate_mean_single_from_Kxx(self.input_data[0,None], self.D, Kxx)


        # DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo
        # print('predictive_std')
        # print(predictive_std)
        # print('ac')
        # print(ac)
        # print('predictive_mean')
        # print(predictive_mean)
        # print('output_data')
        # print(self.output_data)

        # dim = 5

        # fig,ax = plt.subplots()
        # ax.plot(self.input_data[:,dim], self.output_data, 'o')
        # ax.errorbar(self.input_data[:,dim], predictive_mean, yerr=predictive_std, fmt='o')

        # plt.savefig('test.png')


        # import sys 
        # sys.exit()

        pass


    def predict(self, input_data, return_std=False):
        # Predict the output data for the given input data.


        # OLD CODE
        # latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
        # predictive_dist = self.opt_posterior.likelihood(latent_dist)
        # predictive_mean = predictive_dist.mean()
        # ab = self.opt_posterior.predict_mean_single(input_data, self.D)

        if self.recompute_kernel_matrix:
            Kxx = self.opt_posterior.compute_Kxx(self.D)
            self.recompute_kernel_matrix = False
            self.Kxx = Kxx
        else:
            Kxx = self.Kxx

        ac = self.opt_posterior.calculate_mean_single_from_Kxx(input_data, self.D, Kxx)

        if not return_std:
            return ac
        else:
            latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.likelihood(latent_dist)
            predictive_std = predictive_dist.stddev()
            return ac, predictive_std[0]
        
    def sample(self, input_data, N):
        global RANDOM_KEY
        global SUB_KEY
        # Predict the output data for the given input data.

        if self.recompute_kernel_matrix:
            Kxx = self.opt_posterior.compute_Kxx(self.D)
            self.Kxx = Kxx
        else:
            Kxx = self.Kxx

        ac = self.opt_posterior.calculate_mean_single_from_Kxx(input_data, self.D, Kxx)

        # DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo
        latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        predictive_std = predictive_dist.stddev()
        predictive_mean = predictive_dist.mean()

        RANDOM_KEY, SUB_KEY = random.split(RANDOM_KEY)
        return random.normal(key= SUB_KEY, shape=(1,N)) * jnp.sqrt(predictive_std[0]) + ac
        
    def predict_gradient(self, input_data):
        # Predict the gradient of the output data for the given input data.
        gradient = grad(self.opt_posterior.predict_mean_single)

        return gradient(input_data, self.D)
