from gpjax.gps import ConjugatePosterior

from abc import abstractmethod
from dataclasses import dataclass
from typing import overload

from beartype.typing import (
    Any,
    Callable,
    Optional,
)

import cola
from cola.ops import Dense

import jax.numpy as jnp
from jax.random import (
    PRNGKey,
    normal,
)
from jaxtyping import (
    Float,
    Num,
)

from gpjax.dataset import Dataset

from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
)

import time

#
# This is an additional method for the ConjugatePosterior class.
# It speeds up the default predict_mean method.
#

def predict_mean_single(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Observation noise o²
    #obs_noise = self.likelihood.obs_noise
    # mx = self.prior.mean_function(x)

    # Precompute Gram matrix, Kxx, at training inputs, x
    Kxx = self.prior.kernel.gram(x)
    Kxx += cola.ops.I_like(Kxx) * self.jitter

    # Σ = Kxx + Io²
    Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
    Sigma = cola.PSD(Sigma)

    if mask is not None:
        y = jnp.where(mask, 0.0, y)
        mx = jnp.where(mask, 0.0, mx)
        Sigma_masked = jnp.where(mask + mask.T, 0.0, Sigma.to_dense())
        Sigma = cola.PSD(Dense(jnp.where(jnp.diag(jnp.squeeze(mask)), 1 / (2 * jnp.pi), Sigma_masked)))

    Kxt = self.prior.kernel.cross_covariance(x, t)

    # Σ⁻¹ Kxt
    if mask is not None:
        Kxt = jnp.where(mask * jnp.ones((1, n_test), dtype=bool), 0.0, Kxt)
    Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    # mean = jnp.matmul(Sigma_inv_Kxt.T, y)
    mean = jnp.matmul(Sigma_inv_Kxt.T, y - mx)

    return jnp.atleast_1d(mean.squeeze())[0]


def predict_mean_single_sparse(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    inducing_points, 
    inducing_values,
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    #x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Observation noise o²
    #obs_noise = self.likelihood.obs_noise
    # mx = self.prior.mean_function(inducing_points).flatten()

    # Precompute Gram matrix, Kxx, at training inputs, x
    Kxx = self.prior.kernel.gram(inducing_points)
    Kxx += cola.ops.I_like(Kxx) * self.jitter

    # Σ = Kxx + Io²
    Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
    Sigma = cola.PSD(Sigma)

    Kxt = self.prior.kernel.cross_covariance(inducing_points, t)

    # Σ⁻¹ Kxt
    
    Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    mean = jnp.matmul(Sigma_inv_Kxt.T, inducing_values)
    # mean = jnp.matmul(Sigma_inv_Kxt.T, inducing_values - mx)

    #return mean
    return jnp.atleast_1d(mean.squeeze())[0]

#
# Compute Kxx matrix
#
def compute_inv_Kxx(
    self,
    train_data: Dataset,
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Observation noise o²
    #obs_noise = self.likelihood.obs_noise
    mx = self.prior.mean_function(x)

    # Precompute Gram matrix, Kxx, at training inputs, x
    Kxx = self.prior.kernel.gram(x)
    Kxx += cola.ops.I_like(Kxx) * self.jitter

    # Σ = Kxx + Io²
    Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
    Sigma = cola.PSD(Sigma)

    if mask is not None:
        y = jnp.where(mask, 0.0, y)
        mx = jnp.where(mask, 0.0, mx)
        Sigma_masked = jnp.where(mask + mask.T, 0.0, Sigma.to_dense())
        Sigma = cola.PSD(Dense(jnp.where(jnp.diag(jnp.squeeze(mask)), 1 / (2 * jnp.pi), Sigma_masked)))

    # invert Sigma
    Sigma_inv = cola.inv(Sigma)

    return Sigma_inv

def compute_inv_Kxx_sparse(
    self,
    train_data: Dataset,
    inducing_points,
):
    # Shorter prediction function for single test inputs

    # Precompute Gram matrix, Kxx, at training inputs, x
    Kxx = self.prior.kernel.gram(inducing_points)
    Kxx += cola.ops.I_like(Kxx) * self.jitter

    # Σ = Kxx + Io²
    Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
    Sigma = cola.PSD(Sigma)

    # invert Sigma
    Sigma_inv = cola.inv(Sigma)

    return Sigma_inv


def calculate_mean_single_sparse_from_inv_Kxx(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    inducing_points, 
    inducing_values,
    inv_Kxx: Num[Array, "N N"],
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    #x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Observation noise o²
    #obs_noise = self.likelihood.obs_noise
    # mx = self.prior.mean_function(inducing_points).flatten()

    Kxt = self.prior.kernel.cross_covariance(inducing_points, t)

    # Σ⁻¹ Kxt

    Sigma_inv_Kxt = inv_Kxx @ Kxt

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    # mean = jnp.matmul(Sigma_inv_Kxt.T, inducing_values - mx)
    mean = jnp.matmul(Sigma_inv_Kxt.T, inducing_values)

    #return mean
    return jnp.atleast_1d(mean.squeeze())[0]

def calculate_mean_std_single_sparse_from_inv_Kxx(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    inducing_points, 
    inducing_values,
    inv_Kxx: Num[Array, "N N"],
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    #x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Observation noise o²
    #obs_noise = self.likelihood.obs_noise
    # mx = self.prior.mean_function(inducing_points).flatten()

    Kxt = self.prior.kernel.cross_covariance(inducing_points, t)

    # Σ⁻¹ Kxt

    Sigma_inv_Kxt = inv_Kxx @ Kxt

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    # mean = jnp.matmul(Sigma_inv_Kxt.T, inducing_values - mx)
    mean = jnp.matmul(Sigma_inv_Kxt.T, inducing_values)

    # epsilon to ensure positive definiteness
    epsilon = 1e-14

    # std
    std = jnp.sqrt(jnp.abs(self.prior.kernel.cross_covariance(t,t) - jnp.matmul(Sigma_inv_Kxt.T, Kxt)) + epsilon) # abs for error prevention


    #return mean
    return jnp.atleast_1d(mean.squeeze())[0], std.squeeze()

def calculate_mean_single_from_inv_Kxx(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    inv_Kxx: Num[Array, "N N"],
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Σ = Kxx + Io²
    # Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
    a = time.time()
    Kxt = self.prior.kernel.cross_covariance(x, t)  # this is the slow part

    # Σ⁻¹ Kxt
    Sigma_inv_Kxt = inv_Kxx @ Kxt

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    mean = jnp.matmul(Sigma_inv_Kxt.T, y)

    return jnp.atleast_1d(mean.squeeze())[0]

def calculate_mean_std_single_from_inv_Kxx(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    inv_Kxx: Num[Array, "N N"],
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Σ = Kxx + Io²
    # Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
    a = time.time()
    Kxt = self.prior.kernel.cross_covariance(x, t)  # this is the slow part

    # Σ⁻¹ Kxt
    Sigma_inv_Kxt = inv_Kxx @ Kxt

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    mean = jnp.matmul(Sigma_inv_Kxt.T, y)

    # epsilon to ensure positive definiteness
    epsilon = 1e-14

    # std
    std = jnp.sqrt(jnp.abs(self.prior.kernel.cross_covariance(t,t) - jnp.matmul(Sigma_inv_Kxt.T, Kxt))+ epsilon)  # abs for error prevention

    return jnp.atleast_1d(mean.squeeze())[0], std.squeeze()




ConjugatePosterior.predict_mean_single = predict_mean_single
ConjugatePosterior.predict_mean_single_sparse = predict_mean_single_sparse
ConjugatePosterior.compute_inv_Kxx = compute_inv_Kxx
ConjugatePosterior.compute_inv_Kxx_sparse = compute_inv_Kxx_sparse
ConjugatePosterior.calculate_mean_single_from_inv_Kxx = calculate_mean_single_from_inv_Kxx
ConjugatePosterior.calculate_mean_std_single_from_inv_Kxx = calculate_mean_std_single_from_inv_Kxx
ConjugatePosterior.calculate_mean_single_sparse_from_inv_Kxx = calculate_mean_single_sparse_from_inv_Kxx
ConjugatePosterior.calculate_mean_std_single_sparse_from_inv_Kxx = calculate_mean_std_single_sparse_from_inv_Kxx
#CollapsedVariationalGaussian.predict_mean_single_sparse = predict_mean_single_sparse