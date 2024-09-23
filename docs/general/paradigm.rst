Paradigm
=================================================

The paradigm of OLÉ is to connect emulation with active and online learning. 
This allows for a precise emulator that can be trained on the fly on a efficient selection of training data.

OLÉ combines 3 key ideas outlined in the following:

Active and Online Learning
---------------------------

The training of the emulator is done while the sampler is running.
This ensures that the emulator is always trained on the most relevant data that are sufficiently close to high likelihood regions.
It can be specified by the `N_sigma` parameter in the emulator settings that specifies the number of standard deviations that are used to select the training data.
Data that are further away from the current high likelihood region are not used for training the emulator.

Furthermore the sampling of the emulator is done in an active way. 
Therefore, the emulator checks if its prediction fulfills a precision criterium. 
If not, the theory code is called to compute the theory model and the emulator is updated on this new data point.
This ensures that the emulator is only used in regions where it is sufficiently accurate. 

Interpretable Emulator
-----------------------

The emulator itself is based on a combination of Principal Component Analysis (PCA) and Gaussian Processes (GP).
The PCA is used to reduce the dimensionality of the data and the GP is used to interpolate the data.
GPs are a powerful tool to interpolate data and provide a prediction of the data with an uncertainty.
Since the PCA projects the correlated high-dimensional data to uncorrelated low-dimensional data, we can train one GP per uncorrelated dimension.

When transformed back to the high-dimensional space, the GP predictions are combined to a (correlated) single prediction.
The uncertainity prediction represents the lack of training data while the explained variance of the PCA gives us an idea of the systematic error of the emulator.
We can combine them to a total error prediction of the emulator.

We use the `GPJax <https://docs.jaxgaussianprocesses.com/>`_ package, that is based on `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ to implement the GP.
It allows for just-in-time compilation and automatic differentiation of the GP, which is crucial for the efficient training and evaluation of the emulator.

Uncertainty Quantification
--------------------------

The emulator provides a prediction of the data and an uncertainty. 
However, the uncertainty estimate of the emulator is hard to interpret in the context of parameter inference. 
Thus, we translate the emulator prediction uncertainity to a likelihood uncertainty.
This can be either done by sampling from the emulator prediction and compute the likelihood of the samples 
or by error propagation (if the likelihood is differentiable). 
While the latter is more efficient, the former is more general and can be used for any likelihood.

With this we can construct a precision criterium for the emulator. 
It should reflect the demand that the emulator should be precise close to high likelihood regions and less precise further away from high likelihood regions.
Therefore, we construct a function of allowed emulator error that depends on the likelihood value, the maximum likelihood value and the emulator error:

:math:`$\sigma_{\text{emulator}} = \sigma_{\text{const}} + \sigma_{\text{lin}} \cdot \left(\log \frac{\mathcal{L}_\text{best-fit}}{\mathcal{L}_\text{emulator}}\right)  + \sigma_{\text{quad}} \cdot \left(\log \frac{\mathcal{L}_\text{best-fit}}{\mathcal{L}_\text{emulator}}\right)^2$`

Both the constant and linear constants can be set in the emulator settings. 
In particular the linear constant can be interpreted as an acceptable bias in the posterior.
The quadratic term is constructed such that it dominates the precision criterium outside of the `N_sigma` region.
