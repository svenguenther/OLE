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

from gpjax.base import (
    Module,
    param_field,
    static_field,
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

    mean_t = self.prior.mean_function(t)
    Kxt = self.prior.kernel.cross_covariance(x, t)

    # Σ⁻¹ Kxt
    if mask is not None:
        Kxt = jnp.where(mask * jnp.ones((1, n_test), dtype=bool), 0.0, Kxt)
    Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

    return jnp.atleast_1d(mean.squeeze())[0]

#
# Compute Kxx matrix
#
def compute_Kxx(
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

    return Sigma

def calculate_mean_single_from_Kxx(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    Kxx: Num[Array, "N N"],
):
    # Shorter prediction function for single test inputs

    # Unpack training data
    x, y, n_test, mask = train_data.X, train_data.y, train_data.n, None

    # Unpack test inputs
    t = test_inputs

    # Σ = Kxx + Io²
    Sigma = Kxx #+ cola.ops.I_like(Kxx) * obs_noise
    mean_t = self.prior.mean_function(t)

    a = time.time()
    Kxt = self.prior.kernel.cross_covariance(x, t)  # this is the slow part

    # Σ⁻¹ Kxt
    Sigma_inv_Kxt = cola.solve(Sigma, Kxt)  # this is not as slow as Kxt but second slowest

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y)
    
    # std
    std = self.prior.kernel.cross_covariance(t,t) - jnp.matmul(Sigma_inv_Kxt.T, Kxt)

    print('mean')
    print(mean)
    print('std')
    print(std)


    return jnp.atleast_1d(mean.squeeze())[0]


ConjugatePosterior.predict_mean_single = predict_mean_single
ConjugatePosterior.compute_Kxx = compute_Kxx
ConjugatePosterior.calculate_mean_single_from_Kxx = calculate_mean_single_from_Kxx