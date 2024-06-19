# Here we will program the data processing class. One instance of this class will be created for each quantity that is to be emulated.
# Each quantity may have different number of dimensions. Thus, each data processing requires a different preprocessing and potentially PCA data compression.
# This class additionally stores all relevant data

from OLE.utils.base import BaseClass
import numpy as np
import jax.numpy as jnp
import os
import gc
import copy

import scipy as sp

from OLE.plotting import data_plot_raw, data_plot_normalized, pca_parameter_plot, variance_plots, eigenvector_plots

class data_processor(BaseClass):

    input_size: int
    output_size: int

    hyperparameters: dict

    # raw data
    input_data_raw: jnp.ndarray
    output_data_raw: jnp.ndarray

    # normalized data
    input_data_normalized: jnp.ndarray
    output_data_normalized: jnp.ndarray

    # (potentially) compressed data which are handed to the emulator
    output_data_emulator: jnp.ndarray
    output_data_emulator_dim: int

    # input means and stds
    input_means: jnp.ndarray
    input_stds: jnp.ndarray

    # output means and stds
    output_means: jnp.ndarray
    output_stds: jnp.ndarray

    # output PCA compression means and stds
    output_pca_means: jnp.ndarray
    output_pca_stds: jnp.ndarray


    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)


    def initialize(self, input_size, output_size, quantity_name, **kwargs):
        # Initialize the data processor with the input and output size of the quantity to be emulated.
        self.input_size = input_size
        self.output_size = output_size
        self.quantity_name = quantity_name

        defaulthyperparameters = {
            # explained variance cutoff is the minimum explained variance which is required for the PCA compression. Once this value is reached, the PCA compression is stopped.
            'explained_variance_cutoff': 0.999,
            # this should also inform the error of the GPs to remain consistent. Or alternatively since we specify error params,
            # those might also set this parameter

            # maximal number of dimensions of the compressed data
            'max_output_dimensions': 30,

            # plotting directory
            'plotting_directory': None,

            # testset fraction
            'testset_fraction': None,

            # Load of (observable) covmats
            # If obersavables are not provided, the data is normalized using the means and stds of the data.

            # The observable covmats can be either 1 dimensional and represent the diagonal of the covariance matrix or 2 dimensional and represent the full covariance matrix.
            # If None is given we compute the covariance matrix from the data.
            'observable_covmat': None,

            # Normalize by full covariance matrix? If False, we normalize by the diagonal of the covariance matrix. 
            # Note that a full covmat normalization is computationally more expensive.
            'normalize_by_full_covmat': False,
            
        }

        self.data_covmat = kwargs['data_covmat']

        if self.data_covmat is not None:
            # check if the data covmat is 1 or 2 dimensional
            if len(self.data_covmat.shape) == 1:
                # if the data covmat is 1 dimensional, we assume that it is the diagonal of the covariance matrix
                self.data_covmat = jnp.diag(self.data_covmat)
            elif len(self.data_covmat.shape) == 2:
                pass 
            else:
                # for scalar quantities, the data covmat is a 1x1 matrix
                self.data_covmat = jnp.array([[self.data_covmat]])

            # check that size of covmat fits the output size
            if (self.data_covmat.shape[0] != self.output_size) or (self.data_covmat.shape[1] != self.output_size):
                raise ValueError("Size of data covmat does not fit the output size. Expected size: %d, given size: %d", self.output_size, self.data_covmat.shape[0])
            
            self.data_covmat = jnp.array(self.data_covmat)


        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        self.input_data_raw = None
        self.output_data_raw = None

        self.input_data_normalized = None
        self.output_data_normalized = None
        
        pass

    def clean_data(self):
        del self.input_data_raw
        del self.output_data_raw

        del self.input_data_normalized
        del self.output_data_normalized

        self.input_data_raw = None
        self.output_data_raw = None

        self.input_data_normalized = None
        self.output_data_normalized = None
        

        gc.collect()

        return

    def load_data(self, input_data_raw, output_data_raw):
        # Load the raw data from the data cache.
        self.input_data_raw = input_data_raw
        self.output_data_raw = output_data_raw

        pass

    def compute_normalization(self):
        # Normalize the raw data.
        # calculate the means and stds of the input data
        self.input_means = jnp.mean(self.input_data_raw, axis=0)
        self.input_stds = jnp.std(self.input_data_raw, axis=0)

        # calculate the means and stds of the output data
        self.output_means = jnp.mean(self.output_data_raw, axis=0)

        if self.data_covmat is not None: 
            self.output_stds = jnp.sqrt(jnp.abs(jnp.diag(self.data_covmat)))
        else:
            self.output_stds = jnp.std(self.output_data_raw, axis=0)

        # set all stds which are 0 to 1
        
        self.input_stds = jnp.where(self.input_stds == 0, 1, self.input_stds)
        self.output_stds = jnp.where(self.output_stds == 0, 10**8, self.output_stds)

        # 

    def compute_compression(self):
        # Compress the normalized data. This is done by applying a PCA to the normalized data.
        # The PCA is performed on the output data. The output data is compressed to the dimensionality of the input data.
        # The compressed data is handed to the emulator.

        # calculate the PCA of the output data
        data_cov = jnp.dot(self.output_data_normalized.T, self.output_data_normalized)

        n_eigenvalues = min(self.hyperparameters['max_output_dimensions'], self.output_size)

        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(np.array(data_cov), n_eigenvalues)

        # sort the eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:,idx].real

        # calculate the explained variance
        explained_variance = eigenvalues / jnp.sum(eigenvalues)

        # calculate the cumulative explained variance
        cumulative_explained_variance = jnp.cumsum(explained_variance)

        # find the number of components which explain the variance
        n_components = jnp.argmax(cumulative_explained_variance > self.hyperparameters['explained_variance_cutoff']) + 1

        # if the number of components is larger than the maximal number of dimensions, set the number of components to the maximal number of dimensions
        if n_components == self.hyperparameters['max_output_dimensions']:
            self.warning("Number of components is larger than the maximal number of dimensions. Setting number of components to %d", n_components)
            
        self.output_data_emulator_dim = n_components

        self.info("Compressing data to %d dimensions", n_components)

        # calculate the projection matrix
        self.projection_matrix = copy.deepcopy(eigenvectors[:, :n_components])

        # plot the explained variance, the cumulative explained variance and the eigenvectors
        if self.hyperparameters['plotting_directory'] is not None:
            # check that the directory exists
            if not os.path.exists(self.hyperparameters['plotting_directory']+ "/PCA_plots"):
                os.makedirs(self.hyperparameters['plotting_directory']+ "/PCA_plots")
            variance_plots(explained_variance, 'explained variance '+self.quantity_name,'explained variance ', self.hyperparameters['plotting_directory']+ "/PCA_plots/explained_variance_"+self.quantity_name+'.png')
            variance_plots(1.0-cumulative_explained_variance, '1 - cumulative variance '+self.quantity_name,'1 - cumulative variance ', self.hyperparameters['plotting_directory']+ "/PCA_plots/cumulative_variance_"+self.quantity_name+'.png')
            eigenvector_plots(eigenvectors[:, :n_components].T, 'Eigenvectors '+self.quantity_name ,self.hyperparameters['plotting_directory']+ "/PCA_plots/eigenvectors_"+self.quantity_name+'.png')
        
        del eigenvectors
        del eigenvalues

        gc.collect()


    # The following functions are used to process the input and output data of the emulator during the training process
    def normalize_training_data(self):
        del self.input_data_normalized
        del self.output_data_normalized

        gc.collect()

        # normalize the input data
        self.input_data_normalized = (self.input_data_raw - self.input_means) / self.input_stds

        # normalize the output data
        if self.hyperparameters['normalize_by_full_covmat'] and self.data_covmat is not None:
            _ = self.output_data_raw - self.output_means
            self.output_data_normalized = jnp.dot(jnp.linalg.inv(self.data_covmat), _.T).T
        else:
            self.output_data_normalized = (self.output_data_raw - self.output_means) / self.output_stds

        # if there is a plotting directory, plot the raw output data and the normalized output data
        if self.hyperparameters['plotting_directory'] is not None:
            # check that the directory exists
            if not os.path.exists(self.hyperparameters['plotting_directory']):
                os.makedirs(self.hyperparameters['plotting_directory'])
            data_plot_raw(self.output_data_raw, self.quantity_name, self.hyperparameters['plotting_directory']+ "/raw_data_"+self.quantity_name)
            data_plot_normalized(self.output_data_normalized, self.quantity_name, self.hyperparameters['plotting_directory']+ "/normalized_data_"+self.quantity_name)

        pass

    def denormalize_data(self, output_data_normalized):
        # denormalize the data
        if self.hyperparameters['normalize_by_full_covmat'] and self.data_covmat is not None:
            output_data_raw = jnp.dot(self.data_covmat, output_data_normalized.T).T + self.output_means
        else:
            output_data_raw = output_data_normalized * self.output_stds + self.output_means

        return output_data_raw
    
    def denormalize_std(self, output_std_normalized):
        # denormalize the data
        if self.hyperparameters['normalize_by_full_covmat'] and self.data_covmat is not None:
            output_std_raw = jnp.dot(self.data_covmat, output_std_normalized.T).T
        else:
            output_std_raw = output_std_normalized * self.output_stds

        return output_std_raw

    # data compression
    def compress_training_data(self):
        # project the output data onto the projection matrix
        self.output_data_emulator = jnp.dot(self.output_data_normalized, self.projection_matrix)

        # perform second normalization of the compressed data
        self.output_pca_means = jnp.mean(self.output_data_emulator, axis=0)
        self.output_pca_stds = jnp.std(self.output_data_emulator, axis=0)



        # set all stds which are 0 to 1
        self.output_pca_stds = jnp.where(self.output_pca_stds == 0, 1, self.output_pca_stds)


        # normalize the compressed data
        self.output_data_emulator = (self.output_data_emulator - self.output_pca_means) / self.output_pca_stds

        # make plots of the compressed data
        if self.hyperparameters['plotting_directory'] is not None:
            # check that the directory exists
            if not os.path.exists(self.hyperparameters['plotting_directory']+ "/compressed_data"):
                os.makedirs(self.hyperparameters['plotting_directory']+ "/compressed_data")
            for i in range(self.output_data_emulator_dim):
                for j in range(self.input_size):
                    pca_parameter_plot(self.input_data_raw[:,j], self.output_data_emulator[:,i], 'Parameter '+ str(j), 'PCA - component ' + str(i) ,self.quantity_name, self.hyperparameters['plotting_directory']+ "/compressed_data/"+self.quantity_name+"_dim_"+str(i)+"_input_"+str(j))
            
        pass

    # The following functions are used to process the output data of the emulator outside the training process
    def decompress_data(self, output_data_emulator):
        # undo the second normalization
        output_data_emulator = output_data_emulator * self.output_pca_stds + self.output_pca_means

        # decompress the data
        output_data_normalized = jnp.dot(output_data_emulator, self.projection_matrix.T)

        return output_data_normalized
    
    def decompress_std(self, output_std_emulator):
        # undo the second normalization
        output_std_emulator = output_std_emulator * self.output_pca_stds

        # decompress the data
        output_std_normalized = jnp.dot(output_std_emulator, jnp.abs(self.projection_matrix.T))

        return output_std_normalized

    # The following functions are used to process the input data of the emulator outside the training process
    def normalize_input_data(self, input_data_raw):
        # normalize the data
        input_data_normalized = (input_data_raw - self.input_means) / self.input_stds

        return input_data_normalized
        
    def denormalize_input_data(self, input_data_normalized):
        # normalize the data
        input_data_raw = input_data_normalized * self.input_stds + self.input_means

        return input_data_raw

