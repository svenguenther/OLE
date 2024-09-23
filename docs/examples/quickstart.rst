Quickstart
=================================================

This is a quickstart guide to get you up and running with OLE.
Running OLE for your inference task is a three step process:

1. Define your theory and data/likelihood model
2. Select your sampler and specify the emulation settings
3. Run the sampler

We will go through each of these steps with a simple example.

Step 1: Define your theory and data/likelihood model
----------------------------------------------------

OLE depends the inference problem to be seperable into two parts: the theory model and the data/likelihood model. 
The theory model is the model that predicts the (observational) data given a set of parameters. 
The data/likelihood model is the model that describes the data and computes the likelihood of the data.
Thus, we have two forward models: the theory model and the data/likelihood model of which we can cosntruct an inference pipeline.

This is represented in the following way in the theory and likelihood modules ::

    from OLE.theory import Theory
    from OLE.likelihood import Likelihood

The theory class has two relevant methods: `initialize()` and `compute()`.
The `initialize(self, **kwargs)` method is used to set up the theory model and can be used to set up the theory model with the parameters that can be specified in a dictionary that is handed to OLE.
Furthermore, we need to set the attribute `self.requirements` to a list of the parameters that are required to compute the theory model. 
In that way the sampler can check if the theory model can be computed with the current set of parameters.
The `compute(self, state)` method is used to compute the theory model. 
The state is a nested dictionary that contains a dictionary (under the keyword `parameters`) with the current parameter values and a dictionary with the current computed quantities of the theory model (under the keyword `quantities`).
Those `quantities` are set by the `compute` method which returns the state. This might look like this ::

    class MyTheory(Theory):
        def initialize(self, **kwargs):
            self.requirements = ['A', 'B', 'C']
        def compute(self, state):
            state['quantities']['D_scalar'] = [state['parameters']['A'][0] + state['parameters']['B'][0] + state['parameters']['C'][0]]
            state['quantities']['D_array'] = [state['parameters']['A'][0], state['parameters']['B'][0], state['parameters']['C'][0]]
            return state

The shape of the quantities is required to be an array with a fix shape that does not change during the inference process.

The likelihood class has two relevant methods: `initialize(self, **kwargs)` and `loglike(self, state)`. The `initialize` method is set up in the same manner as for the theory class.
The `loglike(self, state)` method is used to compute the logarithm of the likelihood of the data given the theory model.
It takes the state that was filled by the theory model and returns the logarithm of the likelihood. This might look like this ::

    class MyLikelihood(Likelihood):
        def initialize(self, **kwargs):
            self.requirements = ['D_scalar', 'D_array']
        def loglike(self, state):
            return -0.5 * (state['parameters']['D_scalar'][0] - 1)**2 - 0.5 * (state['parameters']['D_array'][0] - 1)**2 - 0.5 * (state['parameters']['D_array'][1] - 1)**2 - 0.5 * (state['parameters']['D_array'][2] - 1)**2

In general it is highly recommended to use the 'jax' library to compute the likelihood. 
This is because 'jax' allows for just-in-time compilation that can speed up the computation of the likelihood and the emulator significantly and OLE will try to use 'jax' if it is available.
Furthermore, OLE can use the 'jax' library to compute the gradient of the likelihood in combination with the emulator which can be used by the implemented NUT-MCMC sampler and the minimizer.

Note that the theory code does not have to be written in 'jax', since the emulator will be used to compute the theory model at some point of the inference that is fully compatible and differentable with 'jax'.

Step 2: Select your sampler and specify the emulation settings
--------------------------------------------------------------
Once you have defined your theory and likelihood model you can set up the sampler.
There are currently 3 samplers implemented in OLE: NUT-MCMC (required differentiability of the likelihood), Ensemble sampler of emcee and a minimizer.

If the likelihood is differentiable and written in 'jax' the NUT-MCMC sampler is recommended.

The NUT-MCMC sampler can be set up in the following way ::

    from OLE.sampler import NUTSSampler
    sampler = NUTSSampler()

Now you can specify the sampled parameters `A,B,C`. This consists of specifying the priors for the parameters, the initial values and the prosed initial step size.
This can be done in the following way ::

    parameters = {
        'A':    {'prior': {'min': 0.0, 'max': 3.0, 'type': 'uniform'},
                'ref': {'mean': 1.0, 'std': 0.1},
                'proposal': 1.0,},
        'B':    {'prior': {'min': 0.0, 'max': 3.0, 'mean': 1.0, 'std': 0.5, 'type': 'gaussian'},
                'ref': {'mean': 1.0, 'std': 0.5},
                'proposal': 1.0,},
        'C':   {'prior': {'min': 0.0, 'max': 3.0, 'type': 'uniform'},
                'ref': {'mean': 1.0, 'std': 0.1},
                'proposal': 1.0,},
    }

You can select between uniform, gaussian, log-normal and jeffreys priors. Once the parameters are set up you can set up parameters for the sampler and the emulator and intialize the sampler ::

    sampling_settings = {
        'output_dir': './output',
    }

    emulator_settings = {
        'min_data_points': 80,
        'logfile': './log.txt',
    }

    sampler.initialize(
        theory = MyTheory(), 
        likelihood = MyLikelihood(),
        parameters = parameters, 
        sampling_settings = sampling_settings, 
        emulator_settings = emulator_settings)

Step 3: Run the sampler
------------------------

Now you can run the sampler::

    nsteps = 1000
    sampler.run_mcmc(nsteps)

This will run the NUT-MCMC sampler and save the results in the output directory.

You can also access the chains and the results of the sampler by calling::

    chains = sampler.chain

You can also use MPI to run the sampler in parallel. They will all work with a shared cache and produce individual chains that are merged at the end of the run.

Furthermore, if it finds a output directory with the same name it will append its chain to the existing chain. Same hodls true for the cache file.




