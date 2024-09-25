Cobaya Interface
=================================================

Here we describe the interface for using Cobaya with OLE. Prerequisites are that you have both Cobaya and OLE installed.

In order to use Cobaya with OLE, OLE provides a Cobaya Interface that can be used to define an OLE and Cobaya compatible Theory model. 
The likelihood model can be defined in Cobaya. Thus, when building the pipeline it might look like this::

    from OLE.cobaya_interface import cobaya

    class MyLikelihood(cobaya.Likelihood):
        def ...

    class MyTheory(cobaya.Theory):
        def ...

The yaml file for the pipeline looks very similar to the vanilla cobaya one.
In fact there are only 2 differences:
1. The theory block has now the option `emulate`, that can be set to `True` in order to use OLE to emulate the theory.
2. The theory block has the new option of `emulator_settings`, that can be used to pass the OLE settings to the emulator.
This might look like this::

    likelihood:
    MyLikelihood:
        ...
    theory:
        MyTheory:
            emulate: True
            emulator_settings:
            ...
    sampler:
        ...

Otherwise the yaml file is defined as usual in Cobaya. 

We will per default use the just-in-time compilation of the emulator, 
which is the most efficient way to run the emulator. However, this is not the case of the (vanilla cobaya-)likelihood. 
Therefore, one cannot utilize the differentibility of the emulator in the inference process.

For efficiency reasons it is recommended to use `oversampling` in the early burn-in phase of the sampler (when the emulator is not yet trained).
However, once the emulator is trained, the inference pipeline should be run without oversampling since the runtime of the emulator is much faster than the runtime of the likelihood.
By default cobaya will meassure the runtime of each component and adjust the oversampling accordingly, which does not work for OLE which is slow in the beginning but becomes very fast once trained.

Further efficiency can be gained by running the emulator in parrallel via MPI. Note however, that the runtime will be dominated by the likelihood (usually when you use for example Planck likelihoods etc.).
Since they are not well parrallelized, it is recommended to use many chains but only with 1-2 cores per MPI task. OLE will use a shared cache for all chains in order to be as data efficient as possible.

When using the Cobaya interface with cosmological codes, such as CAMB or CLASS, we require a specific way of importing the interface. You find this in the examples directory of the OLE repository.

**IMPORTANT WHEN USING WITH CAMB:** 
When using the Cobaya interface with CAMB, you need to make sure that you **MANUALLY** set blocking such that **all** cosmo parameter are in one block!
