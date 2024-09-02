"""
GP Predicter
"""

# Here we will program the gp_predicter class. One instance of this class will be created for each quantity that is to be emulated.
# Each quantity may have different number of dimensions. Thus, each gp_predicter requires a different preprocessing and potentially PCA data compression.
#
import jax

# use GPJax to fit the data
# from jax.config import config
import os

jax.config.update("jax_enable_x64", True)

from jax import jit, random
import jax.numpy as jnp
import numpy as np
from jaxtyping import install_import_hook
import optax as ox

from functools import partial
import gc
import copy

from jax import grad

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

import time

# Path: OLE/gp_predicter.py

from OLE.utils.base import BaseClass
from OLE.data_processing import data_processor
from OLE.plotting import (
    plain_plot,
    error_plot,
    loss_plot,
    plot_pca_components_test_set,
    plot_prediction_test,
)

from OLE.interfaces import gpjax_interface


class GP_predictor(BaseClass):

    quantity_name: str

    hyperparameters: dict

    GPs: list

    def __init__(self, quantity_name=None, **kwargs):
        super().__init__("GP " + quantity_name, **kwargs)
        self.quantity_name = quantity_name

    def initialize(self, ini_state, **kwargs):
        # in the ini_state and example state is given which contains the parameters and the quantities which are to be emulated. The example state also contains an example value for each quantity which is used to determine the dimensionality of the quantity.

        self.GPs = []

        # default hyperparameters
        defaulthyperparameters = {
            # plotting directory
            "plotting_directory": None,
            # testset fraction. If we have a testset, which is not None, then we will use this fraction of the data as a testset
            "testset_fraction": None,
            # error
            "error_tolerance": 1.0,
            "excess_fraction": 0.1,
            "error_boost": 0.1,
            "kernel_fitting_frequency": 4,
        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        # We need to determine the dimensionality of the quantity to be emulated. This is done by looking at the example state.
        self.output_size = len(ini_state["quantities"][self.quantity_name])
        self.debug("Output size: %d", self.output_size)

        # We need to determine the dimensionality of the parameters. This is done by looking at the example state.
        self.input_size = len(ini_state["parameters"])
        self.debug("Input size: %d", self.input_size)

        # We can now initialize the data processor for this quantity.
        self.data_processor = data_processor(
            "Data processor " + self.quantity_name, debug=self.debug_mode
        )
        self.data_processor.initialize(
            self.input_size, self.output_size, self.quantity_name, **kwargs
        )

        # For each dimension of the output_data we create a GP.
        self.GPs = []
        self.num_GPs = 0  # the number of GPs we have

        pass

    # @partial(jit, static_argnums=0)
    def predict(self, parameters):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros(self.num_GPs)
        for i in range(self.num_GPs):
            output_data_compressed = output_data_compressed.at[i].set(
                self.GPs[i].predict(parameters_normalized)
            )  # this is the time consuming part

        # Untransform the output data.
        output_data_normalized = self.data_processor.decompress_data(
            output_data_compressed
        )

        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)

        return output_data

    # Predict the quantity for the given GPOutput. This is just to split the precit function so we can autodiff it in parts
    def predict_fromGP(self, GPOut):

        output_data_compressed = GPOut

        # Untransform the output data.
        output_data_normalized = self.data_processor.decompress_data(
            output_data_compressed
        )

        output_data = self.data_processor.denormalize_data(output_data_normalized)

        return output_data

    # same as above, just wrapped the input for autodiff
    def predict_fromGP_scalar(self, GPOut, index):

        output_data_compressed = GPOut

        # Untransform the output data.
        output_data_normalized = self.data_processor.decompress_data(
            output_data_compressed
        )

        output_data = self.data_processor.denormalize_data(output_data_normalized)

        return output_data[index]

    # sexond part such that predict(parameters) = predict_fromGP(predict_GPout(self,parameters))
    def predict_GPOut(self, parameters):

        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        output_data_compressed = jnp.zeros(self.num_GPs)
        for i in range(self.num_GPs):
            output_data_compressed = output_data_compressed.at[i].set(
                self.GPs[i].predict(parameters_normalized)
            )  # this is the time consuming part

        return jnp.array(output_data_compressed)

    # @partial(jit, static_argnums=0)
    def predict_value_and_std(self, parameters):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros(self.num_GPs)
        output_std_compressed = jnp.zeros(self.num_GPs)

        for i in range(self.num_GPs):
            v, s = self.GPs[i].predict_value_and_std(parameters_normalized)
            output_data_compressed = output_data_compressed.at[i].set(v)
            output_std_compressed = output_std_compressed.at[i].set(s)

        # Untransform the output data.
        output_data_normalized = self.data_processor.decompress_data(
            output_data_compressed
        )
        output_std_normalized = self.data_processor.decompress_std(
            output_std_compressed
        )

        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)
        output_std = self.data_processor.denormalize_std(output_std_normalized)

        return output_data, output_std

    # @partial(jit, static_argnums=0)
    def sample_prediction(self, parameters, N=1, RNGkey=random.PRNGKey(time.time_ns())):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros((N, self.num_GPs))
        for i in range(self.num_GPs):
            _, RNGkey = self.GPs[i].sample(parameters_normalized, RNGkey=RNGkey)
            output_data_compressed = output_data_compressed.at[:, [i]].set(
                _
            )  # this is the time consuming part

        # Untransform the output data.
        output_data_normalized = jnp.array(
            [
                self.data_processor.decompress_data(output_data_compressed[i, :])
                for i in range(N)
            ]
        )

        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)
        return output_data, RNGkey

        # @partial(jit, static_argnums=0)

    def sample_prediction_noiseFree(
        self, parameters, N=1, RNGkey=random.PRNGKey(time.time_ns())
    ):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros((N, self.num_GPs))
        for i in range(self.num_GPs):
            _, RNGkey = self.GPs[i].sample_noiseFree(
                parameters_normalized, RNGkey=RNGkey
            )
            output_data_compressed = output_data_compressed.at[:, [i]].set(
                _
            )  # this is the time consuming part

        # Untransform the output data.
        output_data_normalized = jnp.array(
            [
                self.data_processor.decompress_data(output_data_compressed[i, :])
                for i in range(N)
            ]
        )

        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)
        return output_data, RNGkey

    # @partial(jit, static_argnums=0)
    def predict_gradients(self, parameters):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = np.zeros((self.num_GPs, self.input_size))

        for i in range(self.num_GPs):
            output_data_compressed[i] = self.GPs[i].predict_gradient(
                parameters_normalized.copy()
            )

        output_data_compressed = jnp.array(output_data_compressed)

        # data out
        data_out = np.zeros((self.output_size, self.input_size))

        # note that in order to get the gradients we have to scale it twice with input and output normalization

        for i in range(self.input_size):
            data_out[:, i] = self.data_processor.decompress_data(
                output_data_compressed[:, i]
            )
            data_out[:, i] /= self.data_processor.input_stds[i]

        for j in range(self.output_size):
            data_out[j, :] *= self.data_processor.output_stds[j]

        return data_out.T

    def reset_error(self):

        for i in range(self.num_GPs):
            self.GPs[i].hyperparameters["error_tolerance"] = self.hyperparameters[
                "error_tolerance"
            ]

    def disable_error(self):

        self.hyperparameters["error_tolerance"] = 0.0

        for i in range(self.num_GPs):
            self.GPs[i].hyperparameters["error_tolerance"] = self.hyperparameters[
                "error_tolerance"
            ]

    # def set_error(self,index,quantity_derivs,acceptable_error):

    #    GPOut = jnp.array([self.GPs[i].output_data[index][0] for i in range(self.num_GPs)])

    # get derivatives of output wrt to GP's
    #    GP_derivs = grad(self.predict_fromGP_scalar,0)

    #    derivs = []
    #    for i in range(self.output_size):
    #        derivs.append(GP_derivs(GPOut,i))
    # derivs is now a matrix with first index refering to the output and second index to the GP

    # dloglike/dGP = dloglike/dQuant * dQuant/dGP

    #    dlogdGP = []
    #    for i in range(self.num_GPs):
    #        dlog = 0.
    #        for j in range(self.output_size):
    #            dlog += quantity_derivs[j]*derivs[j][i]
    #        dlogdGP.append(dlog)

    #    for i in range(self.num_GPs):
    #        max_error = acceptable_error / dlogdGP[i] # check that this is correct ...

    #        if self.GPs[i].hyperparameters['error_tolerance'] > max_error**2:
    #            self.GPs[i].hyperparameters['error_tolerance'] = max_error[0]**2

    def train(self):
        # Train the GP emulator.

        input_data = self.data_processor.input_data_normalized
        output_data = self.data_processor.output_data_emulator

        self.num_GPs = output_data.shape[1]

        # if there are not Gps yet we need to create them
        if self.num_GPs > len(self.GPs):
            for i in range(len(self.GPs), output_data.shape[1]):
                # Create a GP for each dimension of the output_data.
                self.GPs.append(
                    copy.deepcopy(
                        GP(
                            "GP " + self.quantity_name + " dim " + str(i),
                            **self.hyperparameters,
                        )
                    )
                )

        for i in range(self.num_GPs):

            # Load the data into the GP.
            self.GPs[i].load_data(input_data, output_data[:, i])

            # here is the break in training for setting the error. If we implement it here it would be much nicer!!

            # Train the GP.
            self.GPs[i].train()

        # if we have a plotting directory, we will run some tests
        if (
            self.hyperparameters["plotting_directory"] is not None
            and self.hyperparameters["testset_fraction"] is not None
        ):
            self.run_tests()

        pass

    def initialize_training(self):
        # Set up for training.

        input_data = self.data_processor.input_data_normalized
        output_data = self.data_processor.output_data_emulator

        self.num_GPs = output_data.shape[1]

        # if there are not Gps yet we need to create them
        if self.num_GPs > len(self.GPs):
            for i in range(len(self.GPs), output_data.shape[1]):
                # Create a GP for each dimension of the output_data.
                self.GPs.append(
                    copy.deepcopy(
                        GP(
                            "GP " + self.quantity_name + " dim " + str(i),
                            **self.hyperparameters,
                        )
                    )
                )

        for i in range(self.num_GPs):

            # Load the data into the GP.
            self.GPs[i].load_data(input_data, output_data[:, i])

        pass

    def finalize_training(self):
        # Train the GP emulator.

        for i in range(self.num_GPs):

            # Train the GP.
            self.GPs[i].train()

        # if we have a plotting directory, we will run some tests
        if (
            self.hyperparameters["plotting_directory"] is not None
            and self.hyperparameters["testset_fraction"] is not None
        ):
            self.run_tests()

        pass

    def train_single_GP(self, input_data, output_data):
        # Train the GP emulator.
        D = gpx.Dataset(input_data, output_data)

        pass

    def load_data(self, input_data_raw, output_data_raw):
        # Load the raw data from the data cache.
        self.data_processor.clean_data()
        self.data_processor.load_data(input_data_raw, output_data_raw)
        pass

    def set_parameters(self, parameters):
        # Set the parameters of the emulator.
        self.data_processor.set_parameters(parameters)
        pass

    def run_tests(self):
        # Run tests for the emulator.
        np.random.seed(0)
        train_indices, test_indices = np.split(
            np.random.permutation(self.GPs[0].D.n),
            [int((1 - self.hyperparameters["testset_fraction"]) * self.GPs[0].D.n)],
        )

        # predict the test set
        for i in range(len(test_indices)):
            self.debug("Predicting test set point: %d" % i)
            prediction, std = self.predict_value_and_std(
                self.data_processor.input_data_raw[jnp.array([test_indices[i]])]
            )

            true = self.data_processor.output_data_raw[jnp.array([test_indices[i]])]

            # check that plotting_dir/preictions/ exists
            if not os.path.exists(
                self.hyperparameters["plotting_directory"] + "/predictions/"
            ):
                os.makedirs(
                    self.hyperparameters["plotting_directory"] + "/predictions/"
                )

            # plot the prediction
            plot_prediction_test(
                prediction,
                true,
                std,
                self.quantity_name,
                self.data_processor.input_data_raw[jnp.array(test_indices[i])],
                self.hyperparameters["plotting_directory"]
                + "/predictions/"
                + self.quantity_name
                + "_prediction_"
                + str(i)
                + ".png",
                self.data_processor.data_covmat,
            )

        pass


class GP(BaseClass):

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)

        # default hyperparameters
        defaulthyperparameters = {
            # Kernel type
            "kernel": "RBF",
            # Exponential decay learning rate
            "learning_rate": 0.02,
            # Number of iterations
            "num_iters": 100,
            # plotting directory
            "plotting_directory": None,
            # testset fraction. If we have a testset, which is not None, then we will use this fraction of the data as a testset
            "testset_fraction": None,
            # error
            "error_tolerance": 1.0,
            # numer of test samples to determine the quality of the emulator
            "N_quality_samples": 5,
            # number of sparse GP points
            "sparse_GP_points": 0,
            "is_sparse": False,  # we could have sparse_GP_points > 0 and is_sparse = False if sparse fails to converge
        }

        # The hyperparameters are a dictionary of the hyperparameters for the different quantities. The keys are the names of the quantities.
        self.hyperparameters = defaulthyperparameters

        # Flag that indicates whether it is required to recompute the inverse kernel matrix (inv_Kxx) of the training data set.
        self.recompute_kernel_matrix = False
        self.inv_Kxx = None

        self.D = None
        self.opt_posterior = None

        for key, value in kwargs.items():
            self.hyperparameters[key] = value

        pass

    # def __del__(self):
    #     # remove the GP
    #     del self.D
    #     del self.opt_posterior
    #     self.opt_posterior = None
    #     self.D = None
    #     self.input_data = None
    #     self.output_data = None

    #     gc.collect()

    def load_data(self, input_data, output_data):
        # Load the data from the data processor.
        self.recompute_kernel_matrix = True
        self.input_data = input_data
        self.output_data = output_data[:, None]
        del self.D
        gc.collect()
        self.D = gpx.Dataset(self.input_data, self.output_data)
        self.test_D = None

        # if we have a test fraction, then we will split the data into a training and a test set
        if self.hyperparameters["testset_fraction"] is not None:
            self.debug("Splitting data into training and test set")
            np.random.seed(0)
            train_indices, test_indices = np.split(
                np.random.permutation(self.D.n),
                [int((1 - self.hyperparameters["testset_fraction"]) * self.D.n)],
            )
            self.D = gpx.Dataset(
                self.input_data[train_indices], self.output_data[train_indices]
            )
            self.test_D = gpx.Dataset(
                self.input_data[test_indices], self.output_data[test_indices]
            )

        pass

    def train(self):
        # Train the GP emulator.

        # Create the kernel

        if self.hyperparameters["kernel"] == "RBF":

            kernelNoiseFree = gpx.kernels.RBF() + gpx.kernels.Polynomial(
                degree=1
            )  # + gpx.kernels.White()
            # we add a linear kernel to see if it imporves performance. The idea is that the GP for points
            # far away from support becomes constant and does not give the sampler a lot of usefull information
            # the lienar kernel will at least provide a slope that is meaningfull and allws the sampler to find the
            # best-fit
            # kernelM52 = gpx.kernels.Matern52()
            # kernelLin = gpx.kernels.Polynomial(degree=1)
        else:
            raise ValueError("Kernel not implemented")

        use_nonsparse = True

        if self.hyperparameters["sparse_GP_points"] > 0:

            self.hyperparameters["is_sparse"] = True

            sparse_trained = False
            use_nonsparse = False

            if self.hyperparameters["error_tolerance"] == 0.0:
                kernel = kernelNoiseFree
                use_nonsparse = True
                sparse_trained = True
                print("sparse GP training requires error_tolerance")
            else:

                kernelWhite = gpx.kernels.White(
                    variance=self.hyperparameters["error_tolerance"]
                    * self.hyperparameters["error_boost"]
                )
                # we reduce the white noise term by a factor to leave room for the unceratinity of the sparese GP. A boost of zero removes the white noise
                # and applies all error budget to the sparse GP. Close to a value of one the sparse GP cannot converge as all error budget is used by the white noise
                # a small allocation of white noise may be numerically ideal to smooth out the noise present in the data
                kernelError = kernelWhite.replace_trainable(variance=False)
                kernel = gpx.kernels.SumKernel(kernels=[kernelNoiseFree, kernelError])

            meanf = gpx.mean_functions.Zero()
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
            prior = prior.replace(jitter=1e-20)

            lr = (
                lambda t: self.hyperparameters["learning_rate"] / 3.0
            )  # sparse GP typically require a lower learning rate

            del self.opt_posterior
            self.opt_posterior = None
            gc.collect()

            # Create the likelihood
            likelihood = gpx.gps.Gaussian(num_datapoints=self.D.n)
            likelihood = likelihood.replace_trainable(obs_stddev=False)
            # the target_error defines an error for each point which is nessecary for training a sparse GP
            # ideally we want to set it very small, but this is numerically unstable. Since we know our target
            # accuracy we should make it small compared to that so that the information in the points is still used optimally
            target_error = jnp.sqrt(self.hyperparameters["error_tolerance"]* self.hyperparameters["error_boost"]) / 100.0
            likelihood = likelihood.replace(obs_stddev=target_error)

            posterior = prior * likelihood
            posterior = posterior.replace(jitter=1e-20)

            elbo = gpx.objectives.CollapsedELBO(negative=True)

            while not sparse_trained:
                # cosntruct a sparse GP
                # define initial inducing points
                z = self.input_data[: self.hyperparameters["sparse_GP_points"]]

                if (
                    len(z)
                    >= len(self.input_data)
                    - self.hyperparameters["kernel_fitting_frequency"]
                ):
                    sparse_trained = True
                    use_nonsparse = True
                    print("falling back to normal GP")

                if not use_nonsparse:
                    q = gpx.variational_families.CollapsedVariationalGaussian(
                        posterior=posterior, inducing_inputs=z
                    )

                    self.opt_posterior, history = gpx.fit(
                        model=q,
                        objective=elbo,
                        train_data=self.D,
                        optim=ox.adamw(learning_rate=lr),
                        num_iters=self.hyperparameters["num_iters"],
                        safe=True,
                        key=jax.random.PRNGKey(0),
                    )

                    # now we check if our error on the training points is too large
                    latent_dist = self.opt_posterior(self.input_data, train_data=self.D)
                    predictive_dist = self.opt_posterior.posterior.likelihood(
                        latent_dist
                    )
                    predictive_std = predictive_dist.stddev()

                    add_points = False
                    n_poor = 0
                    for std in predictive_std:
                        if std**2 > self.hyperparameters["error_tolerance"]:

                            n_poor += 1

                    print(
                        "npoor/points ",
                        n_poor,
                        "/",
                        self.hyperparameters["sparse_GP_points"],
                    )

                    if (
                        n_poor
                        > len(predictive_std) * self.hyperparameters["excess_fraction"]
                    ):  # acceptable ratio, allowing too much will overaquire points, allowing too little forces too many spares points
                        add_points = True
                    if add_points:
                        self.hyperparameters[
                            "sparse_GP_points"
                        ] += self.hyperparameters["kernel_fitting_frequency"]
                        self.hyperparameters["sparse_GP_points"] = min(
                            self.hyperparameters["sparse_GP_points"],
                            len(self.input_data)
                            - self.hyperparameters["kernel_fitting_frequency"],
                        )

                    if not add_points:
                        # done
                        sparse_trained = True

        if use_nonsparse:

            self.hyperparameters["is_sparse"] = False

            if self.hyperparameters["error_tolerance"] > 0.0:
                kernelWhite = gpx.kernels.White(
                    variance=self.hyperparameters["error_tolerance"]
                )
                kernelError = kernelWhite.replace_trainable(variance=False)
                kernel = gpx.kernels.SumKernel(kernels=[kernelNoiseFree, kernelError])
            else:
                kernel = kernelNoiseFree

            meanf = gpx.mean_functions.Zero()
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
            prior = prior.replace(jitter=1e-20)

            lr = lambda t: self.hyperparameters["learning_rate"]

            del self.opt_posterior
            self.opt_posterior = None
            gc.collect()

            # Create the likelihood
            likelihood = gpx.gps.Gaussian(num_datapoints=self.D.n)
            likelihood = likelihood.replace_trainable(obs_stddev=False)
            likelihood = likelihood.replace(obs_stddev=1e-10)

            posterior = prior * likelihood
            posterior = posterior.replace(jitter=1e-20)

            negative_mll = gpx.objectives.ConjugateMLL(negative=True)
            negative_mll(posterior, train_data=self.D)

            # fit
            self.opt_posterior, history = gpx.fit(
                model=posterior,
                objective=negative_mll,
                train_data=self.D,
                optim=ox.adam(learning_rate=lr),
                num_iters=self.hyperparameters["num_iters"],
                safe=True,  # what does this do?
                key=jax.random.PRNGKey(0),
            )

        # negative_mll._clear_cache()
        # del negative_mll
        # negative_mll = None

        gc.collect()

        # some debugging output
        if self.hyperparameters["plotting_directory"] is not None:
            # creat directory if not exist
            import os

            if not os.path.exists(
                self.hyperparameters["plotting_directory"] + "/loss/"
            ):
                os.makedirs(self.hyperparameters["plotting_directory"] + "/loss/")
            loss_plot(
                history,
                self._name,
                self.hyperparameters["plotting_directory"]
                + "/loss/"
                + self._name
                + "_loss.png",
            )

            # plot a slice of the trained GP
            x = jnp.linspace(-1.5, 1.5, 100)
            y = []
            std = []
            for xval in x:
                inputDat = jnp.array([jnp.ones(len(self.input_data[0])) * xval])
                mean, stdev = self.predict_value_and_std(inputDat)
                y.append(mean)
                std.append(stdev)
            y = jnp.array(y)
            std = jnp.array(std)
            error_plot(
                x,
                y,
                std,
                self.hyperparameters["plotting_directory"]
                + "/loss/"
                + self._name
                + "_slice.png",
            )

            if self.hyperparameters["testset_fraction"] is not None:
                # check that there are test data
                if self.test_D.n > 0:
                    self.run_test_set_tests()

        pass

    # @partial(jit, static_argnums=0) 
    def predict(self, input_data):
        # Predict the output data for the given input data.


        # OLD CODE
        # latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
        # predictive_dist = self.opt_posterior.likelihood(latent_dist)
        # predictive_mean = predictive_dist.mean()
        # ab = self.opt_posterior.predict_mean_single(input_data, self.D)
        if not self.hyperparameters['is_sparse']:
        
            if self.recompute_kernel_matrix:
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                #self.recompute_kernel_matrix = False    # TODO: This leads to memory leaks in jit mode
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(input_data, self.D, inv_Kxx)

            return ac
        
        else:

            if self.recompute_kernel_matrix:
                #self.recompute_kernel_matrix = False    # TODO: This leads to memory leaks in jit mode
                self.inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
                latent_dist = self.opt_posterior(self.inducing_points, train_data=self.D)
                predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
                self.inducing_values = predictive_dist.mean()

                inv_Kxx = self.opt_posterior.posterior.compute_inv_Kxx_sparse(self.D,self.inducing_points)
                self.inv_Kxx = inv_Kxx
                
                
            

            else:
                inv_Kxx = self.inv_Kxx

            # Old code

            #self.inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
            #latent_dist = self.opt_posterior(self.inducing_points, train_data=self.D)
            #predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            #self.inducing_values = predictive_dist.mean()
            # have to save those along invkxx

            #latent_dist = self.opt_posterior(input_data, train_data=self.D)
            #predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            #predictive_mean = predictive_dist.mean()

            #test = self.opt_posterior.posterior.predict_mean_single_sparse(input_data, self.D, self.inducing_points,self.inducing_values)
            #print('test of convergence with fast v slow methoid')
            #print(predictive_mean[0])
            #print(test)

            #return predictive_mean[0]
        
            #new vwerion
            ac = self.opt_posterior.posterior.predict_mean_single_sparse_from_inv_Kxx(input_data, self.D, self.inducing_points,self.inducing_values,self.inv_Kxx)

            return ac
           
        
    # @partial(jit, static_argnums=0) 
    def predicttest(self, input_data):
        # Predict the output data for the given input data.

        # OLD CODE
        # latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
        # predictive_dist = self.opt_posterior.likelihood(latent_dist)
        # predictive_mean = predictive_dist.mean()
        # ab = self.opt_posterior.predict_mean_single(input_data, self.D)
        if not self.hyperparameters['is_sparse']:
        
            if self.recompute_kernel_matrix:
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                #self.recompute_kernel_matrix = False    # TODO: This leads to memory leaks in jit mode
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(input_data, self.D, inv_Kxx)

            return 'NOT SPARSE'
        
        else:

            if self.recompute_kernel_matrix:
                #self.recompute_kernel_matrix = False    # TODO: This leads to memory leaks in jit mode
                self.inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
                latent_dist = self.opt_posterior(self.inducing_points, train_data=self.D)
                predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
                self.inducing_values = predictive_dist.mean()

                inv_Kxx = self.opt_posterior.posterior.compute_inv_Kxx_sparse(self.D,self.inducing_points)
                self.inv_Kxx = inv_Kxx
                
                
            

            else:
                inv_Kxx = self.inv_Kxx # this actually does nothing

            # have to save those along invkxx

            latent_dist = self.opt_posterior(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            predictive_mean = predictive_dist.mean()

            test = self.opt_posterior.posterior.predict_mean_single_sparse(input_data, self.D, self.inducing_points,self.inducing_values)
            test2 = self.opt_posterior.posterior.predict_mean_single_sparse_from_inv_Kxx(input_data, self.D, self.inducing_points,self.inducing_values,self.inv_Kxx)
            
            #print('test of convergence with fast v slow methoid')
            #print(predictive_mean[0])
            #print(test)
            if 2.* predictive_mean[0] - test - test2 > 0.001:
                print('the methods do not agree!')

            return [predictive_mean[0],test,test2]

    # @partial(jit, static_argnums=0)
    def predict_value_and_std(self, input_data, return_std=False):
        # Predict the output data for the given input data.
        if not self.hyperparameters["is_sparse"]:   # those ifs schould be decided at trace time and therefore fine.. 
            inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(
                input_data, self.D, inv_Kxx
            )

            latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.likelihood(latent_dist)
            predictive_std = predictive_dist.stddev()

            return ac, predictive_std[0]

        else:  # not much optimisation to gain here ...
            latent_dist = self.opt_posterior(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            predictive_mean = predictive_dist.mean()
            predictive_std = predictive_dist.stddev()

            return predictive_mean[0], predictive_std[0]


    

    # @partial(jit, static_argnums=0)
    def sample(self, input_data, RNGkey=random.PRNGKey(time.time_ns())):
        # Predict the output data for the given input data.

        N = self.hyperparameters["N_quality_samples"]

        if not self.hyperparameters["is_sparse"]:

            if self.recompute_kernel_matrix:
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(
                input_data, self.D, inv_Kxx
            )

            # DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo
            latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.likelihood(latent_dist)
            predictive_std = predictive_dist.stddev()

            #  predictive_mean = predictive_dist.mean()

        else:
            latent_dist = self.opt_posterior(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            predictive_mean = predictive_dist.mean()
            predictive_std = predictive_dist.stddev()
            ac = predictive_mean[0]

        # should we fix mean to the true mean and make sure we draw sym around it to only emulate the var??

        # generate new key
        RNGkey, subkey = random.split(RNGkey)

        return (
            random.normal(key=subkey, shape=(N, 1)) * jnp.sqrt(predictive_std[0]) + ac,
            RNGkey,
        )

    def sample_noiseFree(self, input_data, RNGkey=random.PRNGKey(time.time_ns())):
        # Predict the output data for the given input data.

        N = self.hyperparameters["N_quality_samples"]

        if not self.hyperparameters["is_sparse"]:
            if self.recompute_kernel_matrix:
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(
                input_data, self.D, inv_Kxx
            )

            # DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo
            latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.likelihood(latent_dist)
            predictive_std = predictive_dist.stddev() - jnp.sqrt(
                self.hyperparameters["error_tolerance"]
            )

            # predictive_mean = predictive_dist.mean()

        # should we fix mean to the true mean and make sure we draw sym around it to only emulate the var??
        else:
            latent_dist = self.opt_posterior(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            predictive_mean = predictive_dist.mean()
            predictive_std = predictive_dist.stddev() - jnp.sqrt(
                self.hyperparameters["error_tolerance"]
            )
            ac = predictive_mean[0]

        # generate new key
        RNGkey, subkey = random.split(RNGkey)

        return (
            random.normal(key=subkey, shape=(N, 1)) * jnp.sqrt(predictive_std[0]) + ac,
            RNGkey,
        )

    def sample_mean(self, input_data, RNGkey=random.PRNGKey(time.time_ns())):
        # Predict the output data for the given input data.
        # this need updating as we really just want the mean, but for simplicity i want to keep the structure
        # identically. So i just draw the mean N times.
        N = self.hyperparameters["N_quality_samples"]

        if self.recompute_kernel_matrix:
            inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
            self.inv_Kxx = inv_Kxx
        else:
            inv_Kxx = self.inv_Kxx

        ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(
            input_data, self.D, inv_Kxx
        )

        RNGkey, subkey = random.split(RNGkey)

        return random.normal(key=subkey, shape=(N, 1)) * jnp.sqrt(0.0) + ac, RNGkey

    # def predict_gradient(self, input_data):
    #     # Predict the gradient of the output data for the given input data.
    #     gradient = grad(self.opt_posterior.predict_mean_single)

    #     return gradient(input_data, self.D)

    # Some debugging functions
    def run_test_set_tests(self):
        # This function is used to test the test set.
        means = jnp.zeros(self.test_D.n)
        stds = jnp.zeros(self.test_D.n)

        # predict the test set
        for i in range(self.test_D.n):
            mean, std = self.predict_value_and_std(jnp.array([self.test_D.X[i]]))
            means = means.at[i].set(mean)
            stds = stds.at[i].set(std)
            self.debug(
                "Predicted: %f True: %f Error: %f"
                % (mean, self.test_D.y[i], mean - self.test_D.y[i])
            )

        # calculate the mean squared error
        mse = jnp.mean((mean - self.test_D.y) ** 2)

        # test that the directory exists
        if not os.path.exists(
            self.hyperparameters["plotting_directory"] + "/test_set_prediction/"
        ):
            os.makedirs(
                self.hyperparameters["plotting_directory"] + "/test_set_prediction/"
            )

        # plot the mean and the std
        plot_pca_components_test_set(
            jnp.array(self.test_D.y)[:, 0],
            means,
            stds,
            self._name,
            self.hyperparameters["plotting_directory"]
            + "/test_set_prediction/"
            + self._name
            + ".png",
        )

        pass
