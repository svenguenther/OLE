MontePython Interface
=================================================

Here we describe the interface for using MontePython with OLE. Prerequisites are that you have both MontePython and OLE installed.

The key idea is that we specify the path where your MontePython installation is located into your OLE installation within the /OLE/OLE/interfaces/montepython_interface.py file.
This is done by setting the variable `MP_path` to the path of your MontePython installation. 

After that, the interace will load vanilla MontePython and apply changes that are necessary to make it compatible with OLE.
This will not change the MontePython installation itself, but only the instance that is used by OLE.

You can use the MontePython interface in the same way as you would use MontePython. ::

    $ python /path/to/OLE/OLE/interfaces/montepython_interface.py run -p param_file.param -o output_folder ...

This allows you to use the MontePython interface in the same way as you would use MontePython. 
Note however, that the interface is developed for MontePython v3.6.0 and might not work with other versions.

The param_file are the same as in MontePython, with the exception that you can now also use the emulator settings in the param file.
This might look like this ::

    ...
    data.parameters['omega_cdm']    = [ 0.12010,   None, None,     0.0013,    1, 'cosmo']

    #------ Mcmc parameters ----

    data.N=10000

    # Emulator settings
    data.emulator_settings['min_data_points'] = 80
    ...

An example of using MontePython with OLE can be found in the examples directory of the OLE repository.
