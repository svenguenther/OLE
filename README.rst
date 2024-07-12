.. image:: https://github.com/svenguenther/OLE/raw/main/docs/logos/OLE_trans.png
    :width: 800

Online Learning Émulator
===============================

The Online Learning Émulator - OLÉ is a framework to efficiently perform statistical analyeses in cases where one can distinguish between a Theory (simulation) code in which observables are computed and likelihood codes that compute a likelihood for a given computed observable. The efficiency comes from emulating the computationally expensive theory codes with 1-O(20) parameters. 

The required training sample for the emulator is gathered during the inference process (online learning) to ensure a coverage of the relevant parameter space but not of unintesting domains. Additionally this emulator provides the user with an uncertainty estimate of the given emulation call. As a consequence we can use an active sampling algorithm that only adds new training points when the accuracy is insufficient.

This implementation involves both the emulator that can be used independently (also with cobaya interface) and a collection of sampler which includes Ensemble/Minimizer/NUTS - Sampler. The minimizer and NUTS benefit from a differentiable likelihood.

Installation
------------

To install OLE run:

  git clone git@github.com:svenguenther/OLE.git
  pip install ./OLE

Documentation
-------------

There are some test examples in :code:`OLE/examples`. 
