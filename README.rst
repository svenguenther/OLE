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

* The dimensionality of the theory code input does not exceed about 20



Features of OLÉ:

* Easy to use framework

* `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ based implementation

* Implementation of various sampling algorithms like Ensemble, Minimizer, NUTS

* Interfaces with inference tools like `Cobaya <https://github.com/CobayaSampler/cobaya>`_ and `MontePython <https://github.com/brinckmann/montepython_public>`_.


Installation
------------

To install OLÉ run::

    git clone git@github.com:svenguenther/OLE.git
    cd OLE
    pip install .

While not a strict requirement, ``mpi4py`` is recommended for running multiple parallel chains. You can either install this manually, or by installing OLÉ with::

    pip install .[MPI]

If you plan to make modifications to OLÉ, it is recommended to install OLÉ in editable mode by including the ``-e`` flag when pip installing.

If you plan to use OLE with MontePython, you should change the contents of the file ``MP_PATH`` in ``OLE/interfaces/`` to redirect to your ``MontePython`` directory.

Documentation
-------------

The documentation is available at `ReadTheDocs <https://ole.readthedocs.io>`_.


Examples
-------------

Examples on the different features cas be found in the `example directory <https://github.com/svenguenther/OLE/tree/main/OLE/examples>`_. 

.. image:: https://github.com/svenguenther/OLE/raw/main/docs/logos/TTK_logo.png
   :alt: RWTH Aachen
   :target: https://www.particle-theory.rwth-aachen.de/
   :height: 150px
