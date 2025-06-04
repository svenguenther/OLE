"""
Data Processing
"""

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
import sklearn.decomposition as skd
from OLE.utils.mpi import get_mpi_rank

from OLE.plotting import (
    data_plot_raw,
    data_plot_normalized,
    pca_parameter_plot,
    variance_plots,
    eigenvector_plots,
)


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
            "min_variance_per_bin": 5e-6,
            # this should also inform the error of the GPs to remain consistent. Or alternatively since we specify error params,
            # those might also set this parameter
            # maximal number of dimensions of the compressed data
            "max_output_dimensions": 40,
            # however, this number has to be smaller than the min_data_points
            "min_data_points": 80,
            # working directory
            "working_directory": './',
            # plotting directory
            "plotting_directory": None,
            # testset fraction
            "testset_fraction": None,
            # Load of (observable) covmats
            # If obersavables are not provided, the data is normalized using the means and stds of the data.
            # The observable covmats can be either 1 dimensional and represent the diagonal of the covariance matrix or 2 dimensional and represent the full covariance matrix.
            # Normalize by full covariance matrix? If False, we normalize by the diagonal of the covariance matrix.
            # Note that a full covmat normalization is computationally more expensive.
            "normalize_by_full_covmat": False,
        }

        self.data_covmat = kwargs["data_covmat"]

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
            if (self.data_covmat.shape[0] != self.output_size) or (
                self.data_covmat.shape[1] != self.output_size
            ):
                raise ValueError(
                    "Size of data covmat does not fit the output size. Expected size: %d, given size: %d",
                    self.output_size,
                    self.data_covmat.shape[0],
                )

            self.data_covmat = jnp.array(self.data_covmat)

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        self.input_data_raw = None
        self.output_data_raw = None

        self.input_data_normalized = None
        self.output_data_normalized = None

        # check if 'max_output_dimensions' is smaller than 'min_data_points'
        if self.hyperparameters["max_output_dimensions"] > self.hyperparameters["min_data_points"]:
            self.hyperparameters["max_output_dimensions"] = self.hyperparameters["min_data_points"]

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

        # there are some cases in which the stds are 0, e.g. if the data is constant.
        # Those positions with the corresponding data values are stored and later applied whenever the data is denormalized
        self.const_data_positions = jnp.where(self.output_stds == 0)[0]
        self.const_data_values = self.output_means[self.const_data_positions]

        # set all stds which are 0 to 1
        self.input_stds = jnp.where(self.input_stds == 0, 1, self.input_stds)
        self.output_stds = jnp.where(self.output_stds == 0, 1, self.output_stds)

    def compute_compression(self):
        # Compress the normalized data. This is done by applying a PCA to the normalized data.
        # The PCA is performed on the output data. The output data is compressed to the dimensionality of the input data.
        # The compressed data is handed to the emulator.

        n_eigenvalues = min(
            self.hyperparameters["max_output_dimensions"], self.output_size
        )

        # ensure that we got enough data points to perform the PCA
        n_eigenvalues = min(n_eigenvalues, self.output_data_normalized.shape[0])

        # use PCA of Scipy to calculate the eigenvectors and eigenvalues
        pca = skd.PCA(n_components=n_eigenvalues)
        pca.fit(self.output_data_normalized)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        eigenvalues = eigenvalues * (self.output_data_normalized.shape[0]-1)

        eigenvectors = jnp.array(eigenvectors.T)
        eigenvalues = jnp.array(eigenvalues)


        # calculate the explained variance
        self.explained_variance = eigenvalues / jnp.sum(eigenvalues)

        # calculate the cumulative explained variance
        self.cumulative_explained_variance = jnp.cumsum(self.explained_variance)

        # find the number of components which explain the variance.
        min_variance_per_bin = self.hyperparameters["min_variance_per_bin"]
        if self.data_covmat is not None:
            min_variance = min_variance_per_bin * len(
                self.data_covmat[self.data_covmat > 0.0]
            )
        else:
            min_variance = min_variance_per_bin * self.output_size

        # additionally we need to scale the variance by the number of data points.
        min_variance = min_variance * len(self.output_data_normalized)

        n_components = max(len(eigenvalues[eigenvalues > min_variance]), 1)

        if min_variance == 0.0:
            n_components = 1

        # CF: those 5 lineas above are ptoetnially wronmg and harmfull
        # 
        #     
        self.relative_importance = eigenvalues / min_variance

        # if the number of components is larger than the maximal number of dimensions, set the number of components to the maximal number of dimensions
        if n_components == self.hyperparameters["max_output_dimensions"]:
            self.warning(
                "Number of components is larger than the maximal number of dimensions. Setting number of components to %d",
                n_components,
            )

        self.output_data_emulator_dim = n_components

        self.debug("Compressing data to %d dimensions", n_components)

        # calculate the projection matrix
        self.projection_matrix = copy.deepcopy(eigenvectors[:, :n_components])

        # plot the explained variance, the cumulative explained variance and the eigenvectors
        if (self.hyperparameters["plotting_directory"] is not None) and (get_mpi_rank() == 0):
            # check that the directory exists
            if not os.path.exists(
                os.path.join(self.hyperparameters["working_directory"], self.hyperparameters["plotting_directory"], "PCA_plots")
            ):
                os.makedirs(os.path.join(self.hyperparameters["working_directory"], self.hyperparameters["plotting_directory"], "PCA_plots"))
            variance_plots(
                self.explained_variance,
                "explained variance " + self.quantity_name,
                "explained variance ",
                os.path.join(
                self.hyperparameters["working_directory"] 
                , self.hyperparameters["plotting_directory"]
                , "PCA_plots/explained_variance_"
                + self.quantity_name
                + ".png"),
            )
            variance_plots(
                1.0 - self.cumulative_explained_variance,
                "1 - cumulative variance " + self.quantity_name,
                "1 - cumulative variance ",
                os.path.join(
                self.hyperparameters["working_directory"] 
                , self.hyperparameters["plotting_directory"]
                , "PCA_plots/cumulative_variance_"
                + self.quantity_name
                + ".png"),
            )
            eigenvector_plots(
                eigenvectors[:, :n_components].T,
                "Eigenvectors " + self.quantity_name,
                os.path.join(
                self.hyperparameters["working_directory"] 
                , self.hyperparameters["plotting_directory"]
                , "PCA_plots/eigenvectors_"
                + self.quantity_name
                + ".png"),
            )

        del eigenvectors
        del eigenvalues

        gc.collect()

    # The following functions are used to process the input and output data of the emulator during the training process
    def normalize_training_data(self):
        del self.input_data_normalized
        del self.output_data_normalized

        gc.collect()

        # normalize the input data
        self.input_data_normalized = (
            self.input_data_raw - self.input_means
        ) / self.input_stds

        # normalize the output data
        if (
            self.hyperparameters["normalize_by_full_covmat"]
            and self.data_covmat is not None
        ):
            _ = self.output_data_raw - self.output_means
            self.output_data_normalized = jnp.dot(
                jnp.linalg.inv(self.data_covmat), _.T
            ).T
        else:
            safe_output_stds = jnp.where(self.output_stds == 0.0, 1.0, self.output_stds)
            self.output_data_normalized = (
                self.output_data_raw - self.output_means
            ) / safe_output_stds

        # if there is a plotting directory, plot the raw output data and the normalized output data
        if (self.hyperparameters["plotting_directory"] is not None) and (get_mpi_rank() == 0):
            # check that the directory exists
            if not os.path.exists(os.path.join(self.hyperparameters["working_directory"], self.hyperparameters["plotting_directory"])):
                os.makedirs(os.path.join(self.hyperparameters["working_directory"], self.hyperparameters["plotting_directory"]))
            data_plot_raw(
                self.output_data_raw,
                self.quantity_name,
                os.path.join(
                self.hyperparameters["working_directory"] 
                , self.hyperparameters["plotting_directory"]
                , "raw_data_"
                + self.quantity_name),
            )
            data_plot_normalized(
                self.output_data_normalized,
                self.quantity_name,
                os.path.join(
                self.hyperparameters["working_directory"] 
                , self.hyperparameters["plotting_directory"]
                , "normalized_data_"
                + self.quantity_name),
            )

        pass

    def denormalize_data(self, output_data_normalized):
        # denormalize the data
        if (
            self.hyperparameters["normalize_by_full_covmat"]
            and self.data_covmat is not None
        ):
            output_data_raw = (
                jnp.dot(self.data_covmat, output_data_normalized.T).T
                + self.output_means
            )
        else:
            output_data_raw = (
                output_data_normalized * self.output_stds + self.output_means
            )

        const_data_values_broadcasted = jnp.tile(self.const_data_values, (output_data_raw.shape[0], 1))

        output_data_raw = output_data_raw.at[:, self.const_data_positions].set(
            const_data_values_broadcasted
        )

        return output_data_raw
    
    def normalize_data(self, output_data_raw):
        # normalize the data
        if (
            self.hyperparameters["normalize_by_full_covmat"]
            and self.data_covmat is not None
        ):
            output_data_normalized = jnp.dot(
                jnp.linalg.inv(self.data_covmat), (output_data_raw - self.output_means).T
            ).T
        else:
            safe_output_stds = jnp.where(self.output_stds == 0.0, 1.0, self.output_stds)
            output_data_normalized = (
                output_data_raw - self.output_means
            ) / safe_output_stds

        return output_data_normalized

    def denormalize_std(self, output_std_normalized):
        # denormalize the data
        if (
            self.hyperparameters["normalize_by_full_covmat"]
            and self.data_covmat is not None
        ):
            output_std_raw = jnp.dot(self.data_covmat, output_std_normalized.T).T
        else:
            output_std_raw = output_std_normalized * self.output_stds

        return output_std_raw
    
    def normalize_std(self, output_std_raw):
        # normalize the data
        if (
            self.hyperparameters["normalize_by_full_covmat"]
            and self.data_covmat is not None
        ):
            output_std_normalized = jnp.dot(
                jnp.linalg.inv(self.data_covmat), output_std_raw.T
            ).T
        else:
            output_std_normalized = output_std_raw / self.output_stds

        return output_std_normalized

    # data compression
    def compress_training_data(self):
        # project the output data onto the projection matrix
        self.output_data_emulator = jnp.dot(
            self.output_data_normalized, self.projection_matrix
        )

        # perform second normalization of the compressed data
        self.output_pca_means = jnp.mean(self.output_data_emulator, axis=0)
        self.output_pca_stds = jnp.std(self.output_data_emulator, axis=0)

        # set all stds which are 0 to 1
        self.output_pca_stds = jnp.where(
            self.output_pca_stds == 0, 1, self.output_pca_stds
        )

        # normalize the compressed data
        self.output_data_emulator = (
            self.output_data_emulator - self.output_pca_means
        ) / self.output_pca_stds

        # make plots of the compressed data
        if (self.hyperparameters["plotting_directory"] is not None) and (get_mpi_rank() == 0):
            # check that the directory exists
            if not os.path.exists(
                os.path.join(self.hyperparameters["working_directory"], self.hyperparameters["plotting_directory"], "compressed_data")
            ):
                os.makedirs(
                    os.path.join(self.hyperparameters["working_directory"], self.hyperparameters["plotting_directory"], "compressed_data")
                )
            for i in range(self.output_data_emulator_dim):
                for j in range(self.input_size):
                    pca_parameter_plot(
                        self.input_data_raw[:, j],
                        self.output_data_emulator[:, i],
                        "Parameter " + str(j),
                        "PCA - component " + str(i),
                        self.quantity_name,
                        os.path.join(
                        self.hyperparameters["working_directory"] 
                        , self.hyperparameters["plotting_directory"]
                        , "compressed_data"
                        , self.quantity_name
                        + "_dim_"
                        + str(i)
                        + "_input_"
                        + str(j)),
                    )

        pass

    def compress_data(self, output_data_normalized):
        # project the output data onto the projection matrix
        output_data_emulator = jnp.dot(
                    output_data_normalized, self.projection_matrix
                )

        output_data_emulator = (
            output_data_emulator - self.output_pca_means
        ) / self.output_pca_stds

        return output_data_emulator

    # The following functions are used to process the output data of the emulator outside the training process
    def decompress_data(self, output_data_emulator):
        # undo the second normalization
        output_data_emulator = (
            output_data_emulator * self.output_pca_stds + self.output_pca_means
        )

        # decompress the data
        output_data_normalized = jnp.dot(output_data_emulator, self.projection_matrix.T)

        return output_data_normalized

    def decompress_std(self, output_std_emulator):
        # undo the second normalization
        output_std_emulator = output_std_emulator * self.output_pca_stds

        # decompress the data
        output_std_normalized = jnp.dot(
            output_std_emulator, jnp.abs(self.projection_matrix.T)
        )

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
