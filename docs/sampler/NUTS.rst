NUTS
===============================

In the case of a **differentiable** likelihood, we can utilize more efficient Hamiltonian Monte Carlo (HMC) methods. 
The No-U-Turn Sampler (NUTS) is a variant of HMC that automatically tunes the step size and the number of steps per iteration. 
This is done by following the trajectory of the Hamiltonian dynamics until it starts to double back on itself. 


Inside the OLÉ package, the NUTS sampler is implemented in the `sampler` module. 
It comes with a number of features that will be described in the following sections.


In the general it consists of 3 steps: The obtaining training data, prepare the NUT sampler and finally run the sampler.


1. Obtaining training data
--------------------------

When starting from scratch, we need to obtain training data to construct the emulator.
Until then we do not have a differentiable pipeline to compute the likelihood. 
Thus, we use ``nwalkers`` in a Metropolis-Hastings sampler to explore the parameter space and collect training data.
However, we can utilize the differentiability of the likelihood to speed up this process.
This is done by fitting the nuisance parameters during the training stage. 
This is in fact extremely fast but can bias the samples to not follow the posterior anymore (Actually, this happens when the psoterior of the nuisance parameters are not Gaussian anymore). 
This is not a problem since the training data should only represent region where they can provide a good fit.
It can be turned off by setting ``minimize_nuisance_parameters`` to ``False``.

Once a sufficient amount of training data is collected, the emulator is trained and we can omit the Metropolis-Hastings sampler.


2. Prepare the NUTS sampler
---------------------------

Once the emulator is trained, we finally have a differentiable pipeline to compute the likelihood.
This allows us to find the bestfit point of the posterior extremely fast by utilizing the Minimie Sampler with native gradients.
Starting from the bestfit point helps to efficiently start the exploration of the parameter space with the NUTS sampler.
Furthermore, we get the Hessian matrix that can be inverted to get the Fisher matrix.
We do this to obtain the covariance matrix for the proposal distribution of the NUTS sampler to get even more efficient jumps in the parameter space.

In the next step we investigate a suitable stepsize and once it is found we keep it flexible for the NUTS sampler by performing ``M_adapt`` steps. 
Afterwards, the stepsize is fixed and the NUTS sampler is ready to run. 
The samples which were obtained from the stepsize testing already follow the posterior distribution and will be saved in the chain.


3. Run the NUTS sampler
------------------------

Finally, we can run the NUTS sampler.
Here we do not test the accuracy of each sample but only the one of the accepted ones.
This is in particular useful to have the constrion of the U-turns accelerated by jit.
The NUTS sampler will run until the desired number of samples is reached and save the chain in the output directory.


One potential problem of the NUT sampler are prior boundaries. They are not steady and can lead to a rejection of the sample and weird edge effects.
In order to avoid this, we can assign a posterior value outside the prior boundaries to the sample by summing up the likelihood at the projected point inside the prior and a penalty term for exceeding the prior boundaries.
Those points will not be saved in the chain. They do cost a bit of efficiency but are necessary to avoid edge effects.


Debugging
---------

It can be useful to create plots of the emulator to convince oneself that the emulator is working properly. This can be done by setting ``plotting_directory`` and ``testset_fraction``. 
Furthermore, both the sampler and the emulator have the option to create a ``logfile``. When setting these, they will provide a lot of information. Note that for the sampler you also need to ``initialize(debug=True)`` to get the full information.
