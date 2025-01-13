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
from OLE.utils.mpi import get_mpi_rank

from functools import partial
import gc
import copy

from jax import grad

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

import time

# Path: OLE/gp_predicter.py

from OLE.utils.base import BaseClass, constant
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
            # working directory
            "working_directory": './',
            # plotting directory
            "plotting_directory": None,
            # testset fraction. If we have a testset, which is not None, then we will use this fraction of the data as a testset
            "testset_fraction": None,
            # error
            "white_noise_level": 1.0,
            "error_boost": 2.,
            "kernel_fitting_frequency": 20,
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
    

    def predict_GP(self, parameters):
        # Predict the GPs of a quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros(self.num_GPs)
        for i in range(self.num_GPs):
            output_data_compressed = output_data_compressed.at[i].set(
                self.GPs[i].predict(parameters_normalized)
            )

        return output_data_compressed
    
    def predict_GP_value_and_std(self, parameters, include_white_noise = False):
        # Predict the GPs of a quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_value_compressed = jnp.zeros(self.num_GPs)
        output_std_compressed = jnp.zeros(self.num_GPs)
        for i in range(self.num_GPs):
            val, std, error_tol = self.GPs[i].predict_value_and_std(parameters_normalized)
            output_value_compressed = output_value_compressed.at[i].set(
                val
            )
            if include_white_noise:
                output_std_compressed = output_std_compressed.at[i].set(
                    std + error_tol
                )
            else:
                output_std_compressed = output_std_compressed.at[i].set(
                    std
                )
        return output_value_compressed, output_std_compressed


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
        output_err_tol_compressed = jnp.zeros(self.num_GPs)

        for i in range(self.num_GPs):
            v, s, err_tol = self.GPs[i].predict_value_and_std(parameters_normalized)
            output_data_compressed = output_data_compressed.at[i].set(v)
            output_std_compressed = output_std_compressed.at[i].set(s)
            output_err_tol_compressed = output_err_tol_compressed.at[i].set(err_tol)

        # Untransform the output data.
        output_data_normalized = self.data_processor.decompress_data(
            output_data_compressed
        )
        output_std_normalized = self.data_processor.decompress_std(
            output_std_compressed
        )
        output_err_tol_normalized = self.data_processor.decompress_std(
            output_err_tol_compressed
        )

        # Then we denormalize the output data.
        output_data = self.data_processor.denormalize_data(output_data_normalized)
        output_std = self.data_processor.denormalize_std(output_std_normalized)
        output_err_tol = self.data_processor.denormalize_std(output_err_tol_normalized)

        return output_data, output_std, output_err_tol

    # @partial(jit, static_argnums=0)
    def sample_prediction(self, parameters, N=1, noise = 0, RNGkey=random.PRNGKey(time.time_ns())):
        # Predict the quantity for the given parameters.
        # First we normalize the parameters.
        parameters_normalized = self.data_processor.normalize_input_data(parameters)

        # Then we predict the output data for the normalized parameters.
        output_data_compressed = jnp.zeros((N, self.num_GPs))
        for i in range(self.num_GPs):
            _, RNGkey = self.GPs[i].sample(parameters_normalized, noise=noise ,RNGkey=RNGkey)
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
            self.GPs[i].hyperparameters["white_noise_level"] = self.hyperparameters[
                "white_noise_level"
            ]

    def disable_error(self):

        self.hyperparameters["white_noise_level"] = 0.0

        for i in range(self.num_GPs):
            self.GPs[i].hyperparameters["white_noise_level"] = self.hyperparameters[
                "white_noise_level"
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

    #        if self.GPs[i].hyperparameters['white_noise_level'] > max_error**2:
    #            self.GPs[i].hyperparameters['white_noise_level'] = max_error[0]**2

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
            (self.hyperparameters["plotting_directory"] is not None)
            and (self.hyperparameters["testset_fraction"] is not None)
            and (get_mpi_rank() == 0)
        ):
            self.run_tests()

        pass

    def update(self):
        # Update the GP emulator.

        input_data = self.data_processor.input_data_normalized
        output_data = self.data_processor.output_data_emulator

        for i in range(self.num_GPs):

            # Load the data into the GP.
            self.GPs[i].load_data(input_data, output_data[:, i])

            # compute inverse kernel matrix
            self.GPs[i]._compute_inverse_kernel_matrix()
            

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

    def finalize_training(self, new_kernel = True):
        # Train the GP emulator.

        for i in range(self.num_GPs):

            # Train the GP.
            self.GPs[i].train(new_kernel=new_kernel)

        # if we have a plotting directory, we will run some tests
        if (
            (self.hyperparameters["plotting_directory"] is not None)
            and (self.hyperparameters["testset_fraction"] is not None)
            and (get_mpi_rank() == 0)

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
            prediction, std, err_tol = self.predict_value_and_std(
                self.data_processor.input_data_raw[jnp.array([test_indices[i]])]
            )

            true = self.data_processor.output_data_raw[jnp.array([test_indices[i]])]

            # check that plotting_dir/preictions/ exists
            if not os.path.exists(
                self.hyperparameters['working_directory'] + self.hyperparameters["plotting_directory"] + "/predictions/"
            ):
                os.makedirs(
                    self.hyperparameters['working_directory'] + self.hyperparameters["plotting_directory"] + "/predictions/"
                )

            # plot the prediction
            plot_prediction_test(
                prediction,
                true,
                std,
                err_tol,
                self.quantity_name,
                self.data_processor.input_data_raw[jnp.array(test_indices[i])],
                self.hyperparameters['working_directory'] 
                + self.hyperparameters["plotting_directory"]
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
            "learning_rate": 0.1,

            # Number of iterations # None
            "num_iters": None, # per default it will be deduced from the number of data points, but if set will be overwritten
            # Maximal number of iterations
            "max_num_iters": None, # per default it will be deduced from the number of data points, but if set will be overwritten

            # initial number of epochs per datapoint
            "num_epochs_per_dp": 30,
            # maximal number of epochs per datapoint
            "max_num_epochs_per_dp": 120,

            # Early stopping criterion
            "early_stopping": 0.1,
            # Early stopping averaging window
            "early_stopping_window": 10,
            # plotting directory
            "plotting_directory": None,
            # testset fraction. If we have a testset, which is not None, then we will use this fraction of the data as a testset
            "testset_fraction": None,
            # error
            "white_noise_level": 1.0,
            # numer of test samples to determine the quality of the emulator
            "N_quality_samples": 5,
            # number of sparse GP points
            "sparse_GP_points": 0,
            "is_sparse": False,  # we could have sparse_GP_points > 0 and is_sparse = False if sparse fails to converge
            # verbose of the training progress
            "training_verbose": True,
        }

        self.kernel = None

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

    def __del__(self):
        # Destructor
        del self.D
        del self.opt_posterior
        del self.kernel
        del self.inv_Kxx


        import gc 
        gc.collect()

    def load_data(self, input_data, output_data):
        # Load the data from the data processor.
        self.recompute_kernel_matrix = True
        self.input_data = input_data
        self.input_size = input_data.shape[1]
        self.output_data = output_data[:, None]
        del self.D
        gc.collect()
        self.D = gpx.Dataset(self.input_data, self.output_data)
        self.test_D = None

        # set the number of iterations
        if self.hyperparameters["num_iters"] is None:
            self.hyperparameters["num_iters"] = int(self.D.n * self.hyperparameters["num_epochs_per_dp"])
        if self.hyperparameters["max_num_iters"] is None:
            self.hyperparameters["max_num_iters"] = int(self.D.n * self.hyperparameters["max_num_epochs_per_dp"])

        # if we have a test fraction, then we will split the data into a training and a test set
        if (self.hyperparameters["testset_fraction"] is not None) and (get_mpi_rank() == 0):
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

    def train(self, new_kernel = True):
        # Train the GP emulator.

        # Create the kernel

        if self.hyperparameters["kernel"] == "RBF":

            kernelNoiseFree = gpx.kernels.RBF(lengthscale=jnp.ones(self.input_size)) + gpx.kernels.Linear(n_dims=self.input_size)  # + gpx.kernels.White()
            # we add a linear kernel to see if it imporves performance. The idea is that the GP for points
            # far away from support becomes constant and does not give the sampler a lot of usefull information
            # the lienar kernel will at least provide a slope that is meaningfull and allws the sampler to find the
            # best-fit
            # kernelM52 = gpx.kernels.Matern52()
            # kernelLin = gpx.kernels.Polynomial(degree=1)
        else:
            raise ValueError("Kernel not implemented")

        kernelWhite = gpx.kernels.White()
        kernelWhite.variance=constant(self.hyperparameters["white_noise_level"] )
            

        use_nonsparse = True

        if self.hyperparameters["sparse_GP_points"] > 0:

            self.hyperparameters["is_sparse"] = True

            sparse_trained = False
            use_nonsparse = False

            if self.hyperparameters["white_noise_level"] == 0.0:
                use_nonsparse = True
                #sparse_trained = True
                print("sparse GP training requires a white noise error")
        else: 
            use_nonsparse = True
        
        if not use_nonsparse:

            kernel = gpx.kernels.SumKernel(kernels=[kernelNoiseFree, kernelWhite])

            meanf = gpx.mean_functions.Zero()
            #kernel = gpx.kernels.RBF() + gpx.kernels.White() # TODO here do fix this PDF please :) Wenn man den linear kernel benutzt gibt collaped_elbo einen guten wert und danach nurnoch nan...
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

            lr = (
                lambda t: self.hyperparameters["learning_rate"]
            )  # sparse GP typically require a lower learning rate

            # Create the likelihood
            # the target_error defines an error for each point which is nessecary for training a sparse GP
            # ideally we want to set it very small, but this is numerically unstable. Since we know our target
            # accuracy we should make it small compared to that so that the information in the points is still used optimally
            target_error = jnp.sqrt(self.hyperparameters["white_noise_level"] ) / 100.0
            likelihood = gpx.gps.Gaussian(num_datapoints=self.D.n)
            likelihood.obs_stddev = constant(target_error)


            posterior = prior * likelihood

            while not sparse_trained:
                # cosntruct a sparse GP
                # define initial inducing points
                z1 = self.input_data[: self.hyperparameters["sparse_GP_points"]]
                z2 = self.input_data[1: self.hyperparameters["sparse_GP_points"]+1]
                z = 0.5 * (z1+z2)
                if (
                    len(z)
                    >= len(self.input_data)
                    - self.hyperparameters["kernel_fitting_frequency"]
                ):
                    sparse_trained = True
                    use_nonsparse = True
                    print("falling back to normal GP")
                if not sparse_trained:
                    q = gpx.variational_families.CollapsedVariationalGaussian(
                        posterior=posterior, inducing_inputs=z
                    )
                    num_init = int(self.hyperparameters["max_num_iters"]/2.) # dynamical setting not implemented yet

                    obj = lambda p, d: -gpx.objectives.collapsed_elbo(p, d)

                    self.opt_posterior, self.history = gpx.fit(
                        model=q,
                        objective=obj,
                        train_data=self.D,
                        optim=ox.adamw(learning_rate=lr),
                        num_iters=num_init,
                        key=jax.random.PRNGKey(0),
                        verbose=self.hyperparameters["training_verbose"],
                    )

                    del obj
                    del q


                    # now we check if our error on the training points is too large
                    latent_dist = self.opt_posterior(self.input_data, train_data=self.D)
                    predictive_dist = self.opt_posterior.posterior.likelihood(
                        latent_dist
                    )
                    predictive_std = predictive_dist.stddev()
                    
                    add_points = False
                    
                    
                    avg_err = 0. 
                    for std in predictive_std:
                        avg_err += std**2
                    
                    avg_err /= self.hyperparameters["white_noise_level"]*len(predictive_std)
                    print(f"average error {avg_err:.3f} at {len(z)} points")

                    if ( avg_err > self.hyperparameters["error_boost"]):  
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

            if self.hyperparameters["white_noise_level"] > 0.0:
                kernel = gpx.kernels.SumKernel(kernels=[kernelNoiseFree, kernelWhite])
            else:
                kernel = kernelNoiseFree

            if not new_kernel:
                kernel = self.kernel

            meanf = gpx.mean_functions.Zero()
            prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

            lr = lambda t: self.hyperparameters["learning_rate"]

            del self.opt_posterior
            self.opt_posterior = None
            gc.collect()

            # Create the likelihood
            likelihood = gpx.gps.Gaussian(num_datapoints=self.D.n)
            target_error = jnp.max(jnp.array([10e-10, jnp.sqrt(self.hyperparameters["white_noise_level"])/100.]))
            likelihood.obs_stddev = constant(target_error)

            self.posterior = prior * likelihood

            # fit
            # here we fit the GP and 
            converged = False
            self.history = np.array([])
            self.optimizer = ox.adam(learning_rate=lr)
            num_init = self.hyperparameters["num_iters"]
            if new_kernel:
                while not converged:
                    self.posterior, history = gpx.fit(
                        model=self.posterior,
                        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
                        train_data=self.D,
                        optim=self.optimizer,
                        num_iters=num_init,
                        key=jax.random.PRNGKey(0),
                        verbose=self.hyperparameters["training_verbose"],
                    )
                    self.history = np.append(self.history, history)

                    # check loss history. Check if the mean loss of the last 10 interations is not decreasing compared to the mean loss of the last 10 iterations before that
                    if len(self.history) > 2*self.hyperparameters["early_stopping_window"]:
                        mean_loss1 = np.mean(self.history[-self.hyperparameters["early_stopping_window"]:])
                        mean_loss2 = np.mean(self.history[-2*self.hyperparameters["early_stopping_window"]:-self.hyperparameters["early_stopping_window"]])

                        if mean_loss1 > mean_loss2-self.hyperparameters["early_stopping"]:
                            converged = True

                    if len(self.history) > self.hyperparameters["max_num_iters"]:
                        # give warning
                        self.warning("GP training did not converge within max_num_iters")
                        converged = True

                    num_init *= 2

            self.opt_posterior = self.posterior

        self._compute_inverse_kernel_matrix()

        # some debugging output
        if (self.hyperparameters["plotting_directory"] is not None):
            # creat directory if not exist
            import os

            if not os.path.exists(
                self.hyperparameters['working_directory'] + self.hyperparameters["plotting_directory"] + "/loss/"
            ):
                os.makedirs(self.hyperparameters['working_directory'] + self.hyperparameters["plotting_directory"] + "/loss/")
            loss_plot(
                self.history,
                self._name,
                self.hyperparameters['working_directory'] 
                + self.hyperparameters["plotting_directory"]
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
                mean, stdev, err_tol = self.predict_value_and_std(inputDat)
                y.append(mean)
                std.append(stdev)
            y = jnp.array(y)
            std = jnp.array(std)
            error_plot(
                x,
                y,
                std,
                self.hyperparameters['working_directory'] 
                + self.hyperparameters["plotting_directory"]
                + "/loss/"
                + self._name
                + "_slice.png",
            )

            if (self.hyperparameters["testset_fraction"] is not None) and (get_mpi_rank() == 0):
                # check that there are test data
                if self.test_D.n > 0:
                    self.run_test_set_tests()

        pass

    def _compute_inverse_kernel_matrix(self):
        # compute the inverse kernel matrix
        if not self.hyperparameters["is_sparse"]:
            
            del self.inv_Kxx
            inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
            self.inv_Kxx = inv_Kxx

        else:
            # for sparse GPs compute also inducing points and values
            self.inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
            latent_dist = self.opt_posterior(self.inducing_points, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            self.inducing_values = predictive_dist.mean()

            inv_Kxx = self.opt_posterior.posterior.compute_inv_Kxx_sparse(self.D,self.inducing_points)
            self.inv_Kxx = inv_Kxx

        self.recompute_kernel_matrix = False



    # @partial(jit, static_argnums=0) 
    def predict(self, input_data):
        # Predict the output data for the given input data.


        # OLD CODE
        # latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
        # predictive_dist = self.opt_posterior.likelihood(latent_dist)
        # predictive_mean = predictive_dist.mean()
        # ab = self.opt_posterior.predict_mean_single(input_data, self.D)
        if not self.hyperparameters['is_sparse']:
        
            if self.recompute_kernel_matrix: # SG: This is safe in jit! This flag will be altered 
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(input_data, self.D, inv_Kxx)

            return ac
        
        else:

            if self.recompute_kernel_matrix:
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
            ac = self.opt_posterior.posterior.calculate_mean_single_sparse_from_inv_Kxx(input_data, self.D, self.inducing_points,self.inducing_values,self.inv_Kxx)

            return ac
           
        
    def predicttest(self, input_data):
        # Predict the output data for the given input data.

        # OLD CODE
        # latent_dist = self.opt_posterior.predict(input_data, train_data=self.D)
        # predictive_dist = self.opt_posterior.likelihood(latent_dist)
        # predictive_mean = predictive_dist.mean()
        # ab = self.opt_posterior.predict_mean_single(input_data, self.D)
        if not self.hyperparameters['is_sparse']:
        
            if self.recompute_kernel_matrix: # SG: This is safe in jit! This flag will be altered 
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac = self.opt_posterior.calculate_mean_single_from_inv_Kxx(input_data, self.D, inv_Kxx)

            return 'NOT SPARSE'
        
        else:

            if self.recompute_kernel_matrix:
                self.inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
                latent_dist = self.opt_posterior(self.inducing_points, train_data=self.D)
                predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
                self.inducing_values = predictive_dist.mean()

                inv_Kxx = self.opt_posterior.posterior.compute_inv_Kxx_sparse(self.D,self.inducing_points)
                self.inv_Kxx = inv_Kxx
                 

            else:
                inv_Kxx = self.inv_Kxx # this actually does nothing

            latent_dist = self.opt_posterior(input_data, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            predictive_mean = predictive_dist.mean()

            test = self.opt_posterior.posterior.predict_mean_single_sparse(input_data, self.D, self.inducing_points,self.inducing_values)
            test2 = self.opt_posterior.posterior.calculate_mean_single_sparse_from_inv_Kxx(input_data, self.D, self.inducing_points,self.inducing_values,self.inv_Kxx)
            
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
            if self.recompute_kernel_matrix:
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx            

            ac,std = self.opt_posterior.calculate_mean_std_single_from_inv_Kxx(
                input_data, self.D, inv_Kxx
            )

            return ac, std, jnp.sqrt(self.hyperparameters["white_noise_level"])

        else:  
            #latent_dist = self.opt_posterior(input_data, train_data=self.D)
            #predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            #predictive_mean = predictive_dist.mean()
            #predictive_std = predictive_dist.stddev()

            #return predictive_mean[0], predictive_std[0]

            inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
            latent_dist = self.opt_posterior(inducing_points, train_data=self.D)
            predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
            inducing_values = predictive_dist.mean()

            inv_Kxx = self.opt_posterior.posterior.compute_inv_Kxx_sparse(self.D,inducing_points)
            
            ac,std = self.opt_posterior.posterior.calculate_mean_std_single_sparse_from_inv_Kxx(
                input_data, self.D, inducing_points, inducing_values, inv_Kxx
            )

            return ac, std, jnp.sqrt(self.hyperparameters["white_noise_level"])


    

    # @partial(jit, static_argnums=0)
    def sample(self, input_data, noise = 0,RNGkey=random.PRNGKey(time.time_ns())):
        # Predict the output data for the given input data.

        N = self.hyperparameters["N_quality_samples"]

        if not self.hyperparameters["is_sparse"]:

            if self.recompute_kernel_matrix:
                inv_Kxx = self.opt_posterior.compute_inv_Kxx(self.D)
                self.inv_Kxx = inv_Kxx
            else:
                inv_Kxx = self.inv_Kxx

            ac,std = self.opt_posterior.calculate_mean_std_single_from_inv_Kxx(
                input_data, self.D, inv_Kxx
            )

        else:

            if self.recompute_kernel_matrix: # CF: do we intend to keep this flag or remove it?
                self.inducing_points = jnp.array(self.opt_posterior.inducing_inputs)
                latent_dist = self.opt_posterior(self.inducing_points, train_data=self.D)
                predictive_dist = self.opt_posterior.posterior.likelihood(latent_dist)
                self.inducing_values = predictive_dist.mean()

                inv_Kxx = self.opt_posterior.posterior.compute_inv_Kxx_sparse(self.D,self.inducing_points)
                self.inv_Kxx = inv_Kxx
                 

            else:
                inv_Kxx = self.inv_Kxx # this actually does nothing

            ac,std = self.opt_posterior.posterior.calculate_mean_std_single_sparse_from_inv_Kxx(
                input_data, self.D, self.inducing_points, self.inducing_values, self.inv_Kxx
            )

        # should we fix mean to the true mean and make sure we draw sym around it to only emulate the var??

        # generate new key
        RNGkey, subkey = random.split(RNGkey)

        #remove the white noise, noise = 0 meand no white noise, noise = 1 is full noise

        #pre_std = std
        std -=  (1.-noise) *jnp.sqrt(self.hyperparameters["white_noise_level"])
        
        std = jnp.abs(std) 
        # Generate the random samples and apply transformation
        samples = 2. * jax.random.randint(key=subkey, shape=(N, 1), minval=0, maxval=2) - 1.
        scaled_samples = samples * std + ac

        # Prepend `ac` to the transformed array
        # result = jnp.vstack([jnp.array([[ac]]), scaled_samples])
        # new method: roll +- one std values only for new estimator of variance
        return (
            scaled_samples,
            RNGkey,
        )

        #return (
        #    random.normal(key=subkey, shape=(N, 1)) * jnp.sqrt(std) + ac,
        #    RNGkey,
        #)

    # TODO: SG: Is this still needed?
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

    # Some debugging functions
    def run_test_set_tests(self):
        # This function is used to test the test set.
        means = jnp.zeros(self.test_D.n)
        stds = jnp.zeros(self.test_D.n)
        err_tols = jnp.zeros(self.test_D.n)

        # predict the test set
        for i in range(self.test_D.n):
            mean, std, err_tol = self.predict_value_and_std(jnp.array([self.test_D.X[i]]))
            means = means.at[i].set(mean)
            stds = stds.at[i].set(std)
            err_tols = err_tols.at[i].set(err_tol)
            self.debug(
                "Predicted: %f True: %f Error: %f"
                % (mean, self.test_D.y[i].sum(), mean - self.test_D.y[i].sum())
            )

        # calculate the mean squared error
        mse = jnp.mean((mean - self.test_D.y) ** 2)

        # test that the directory exists
        if not os.path.exists(
            self.hyperparameters['working_directory'] + self.hyperparameters["plotting_directory"] + "/test_set_prediction/"
        ):
            os.makedirs(
                self.hyperparameters['working_directory'] + self.hyperparameters["plotting_directory"] + "/test_set_prediction/"
            )

        # plot the mean and the std
        plot_pca_components_test_set(
            jnp.array(self.test_D.y)[:, 0],
            means,
            stds,
            err_tols,
            self._name,
            self.hyperparameters['working_directory'] 
            + self.hyperparameters["plotting_directory"]
            + "/test_set_prediction/"
            + self._name
            + ".png",
        )

        pass
