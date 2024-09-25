.. image:: https://github.com/svenguenther/OLE/raw/main/docs/logos/OLE_trans.png
    :width: 400

.. |docsshield| image:: https://img.shields.io/readthedocs/ole
   :target: http://ole.readthedocs.io

Online Learning Émulator
===============================

:Documentation: |docsshield|

The Online Learning Émulator - OLÉ is a tool to efficiently accelerate statistical analyses with a focus on Cosmology.

It follows combines the idea of emulating computationally expensive codes with the idea of online learning. This is particularly useful when:

* The theory code is computationally expensive. In particular, if it is more expensive than the likelihood code

* A explicit likelihood is available

* The dimensionality of the theory code does not exceed about 20



Features of OLÉ:

* Easy to use framework

* `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ based implementation

* Implementation of various sampling algorithms like Ensemble, Minimizer, NUTS

* Interfaces with inference tools like `Cobaya <https://github.com/CobayaSampler/cobaya>`_ and `MontePython <https://github.com/brinckmann/montepython_public>`_.


Installation
------------

To install OLÉ run::

    git clone git@github.com:svenguenther/OLE.git
    pip install ./OLE

If you plan to use OLE with MontePython, you should change the variable ``MP_path`` in ``OLE/interfaces/montepython_interface.py`` to redirect to your ``MontePython/montepython`` directory.

Documentation
-------------

The documentation is available at `ReadTheDocs <https://ole.readthedocs.io>`_.


Examples
-------------

Examples on the different features cam be found in the `example directory <https://github.com/svenguenther/OLE/tree/main/OLE/examples>`_. 

.. image:: https://github.com/svenguenther/OLE/raw/main/docs/logos/TTK_logo.png
   :alt: RWTH Aachen
   :target: https://www.particle-theory.rwth-aachen.de/
   :height: 150px
