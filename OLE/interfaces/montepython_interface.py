# This is a MontePython - OLE wrapper. It reads your MontePython version and adds OLE functions to it.


"""
.. module:: MontePython
   :synopsis: Main module
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Monte Python, a Monte Carlo Markov Chain code (with Class!)
"""
import sys
import warnings

import numpy as np

#####################################
#  Set the path to MontePython here #
#####################################
MP_path = '/path/to/your/montepython_public/montepython'



# -----------------MAIN-CALL---------------------------------------------
if __name__ == '__main__':
    # use naive vanilla parser to get the path to montepython
    sys.path.insert(0, MP_path)

    import io_mp       # all the input/output mechanisms
    from run import run

    import os

    import time
    import jax

    from io_mp import dictitems,dictvalues,dictkeys

    import data
    data.Data.emulator_settings = {}
    data.Data.MP_path = None



    # HERE WE MODIFY CLASS
    import copy as cp



    import classy

    # deepcopy all callable methods of the Class class 
    callable_methods = {}

    for name in dir(classy.Class):
        if callable(getattr(classy.Class, name)) and not name.startswith("__"):
            def new_method(self, *args, **kwargs):
                my_attribute = self.current_attribute

                if my_attribute=='compute':
                    self.attributes_with_relevant_output = {} # reset the attributes with relevant output

                if my_attribute=='set':
                    self.number_of_compute_calls += 1
                    for key in self.attribute_couter.keys():
                        self.attribute_couter[key] = 0

                # if the current attribute is not in the list of attributes with relevant output, we do not want to call the method
                if self.use_emulated_result and my_attribute in self.emulated_result.keys():
                    if len(self.emulated_result[my_attribute]) > self.attribute_couter[my_attribute]:
                        res = cp.deepcopy(self.emulated_result[my_attribute][self.attribute_couter[my_attribute]])
                    else:
                        res = cp.deepcopy(self.emulated_result[my_attribute][-1])
                    self.attribute_couter[my_attribute] += 1

                    if self.attribute_couter[my_attribute] == self.attribute_max_couter[my_attribute]:
                        self.attribute_couter[my_attribute] = 0
                        # del self.attributes_with_relevant_output[my_attribute]
                    
                else:
                    res = self.cosmo.__getattribute__(my_attribute)(*args, **kwargs)
                        
                    if res is not None and type(res) is not bool:
                        # if my_attribute not in self.attributes_with_relevant_output.keys():
                        if my_attribute in self.attributes_with_relevant_output.keys():
                            if type(res) != dict:
                                if not res == self.attributes_with_relevant_output[my_attribute][-1]:
                                    self.attributes_with_relevant_output[my_attribute].append(cp.deepcopy(res))
                        else:
                            self.attributes_with_relevant_output[my_attribute] = [cp.deepcopy(res)]

                        if self.number_of_compute_calls==1: # count the total number of calls to each attribute
                            if my_attribute in self.attribute_max_couter.keys():
                                self.attribute_max_couter[my_attribute] += 1
                            else:
                                self.attribute_max_couter[my_attribute] = 1

                        self.attribute_couter[my_attribute] = 0

                return res  

            callable_methods[name] = new_method

    class Class_OLE():
        def __init__(self):
            import classy
            self.cosmo = classy.Class()
            self.state = None
            self.callable_methods = callable_methods
            self.current_attribute = None
            self.attribute_max_couter = {} # this is a dictonary with the attributes and the number of times they are typically called in a likelihood calculation
            self.attribute_couter = {} # this is a dictonary with the attributes and the number of times they were called in the current likelihood calculation
            self.number_of_compute_calls = 0
            self.attributes_with_relevant_output = {} # this is a dictonary with the relevant attributes and their output
            self.use_emulated_result = False
            self.emulated_result = {}

            # Emulator related attributes
            self.emulator = None

            # set all callable methods of the Class class to the Class_OLE class
            for name, method in callable_methods.items():
                setattr(Class_OLE, name, method)
            pass
        
        def __getattribute__(self, name):
            if name != 'current_attribute':
                self.current_attribute = name   
            return super().__getattribute__(name)

        def MP_state_to_OLE_state(self, data, MP_state):
            OLE_state = {'parameters': {},
                        'quantities': {},
                        'loglike': None}

            self.state = cp.deepcopy(MP_state)

            # go through all MP elements and addthem to the OLE quantities
            for key, value in MP_state.items():
                if type(value[0]) is dict:
                    for subkey, subvalue in value[0].items():
                        OLE_state['quantities'][subkey] = np.array(subvalue)
                else:
                    OLE_state['quantities'][key] = np.array(value)

            # we need to check the dimensionality of the quantities. They all have to be 1D arrays. If they are not, we need to covnert them
            for key, value in OLE_state['quantities'].items():
                if len(value.shape) >= 1:
                    OLE_state['quantities'][key] = value.flatten()
                else:
                    OLE_state['quantities'][key] = np.array([value])

            # add parameters from data
            for key in data.cosmo_arguments.keys():
                if key in data.parameters.keys():
                    OLE_state['parameters'][key] = np.array([data.cosmo_arguments[key]])

            return OLE_state

        def OLE_state_to_MP_state(self, OLE_state):
            # we deepcopy the
            self.emulated_result = cp.deepcopy(self.attributes_with_relevant_output)

            # we now go through all branches of the emulated result and update the values
            for key, value in self.emulated_result.items():
                if type(value[0]) is dict:
                    for subkey, subvalue in value[0].items():
                        if type(subvalue)== float:
                            self.emulated_result[key][0][subkey] = cp.deepcopy(np.array(OLE_state['quantities'][subkey]))[0]
                        else:
                            self.emulated_result[key][0][subkey] = cp.deepcopy(np.array(OLE_state['quantities'][subkey]))
                else:
                    self.emulated_result[key] = cp.deepcopy(np.array(OLE_state['quantities'][key]))

            self.use_emulated_result = True  

            return self.emulated_result

        def emulate_MP(self, data):
            # check if emulator exists and is trained
            if self.emulator is None:
                return False
            if not self.emulator.trained:
                return False

            # we start to create an OLE state
            OLE_state = {'parameters': {},
                            'quantities': {},
                            'loglike': None}

            # fill parameters from data
            for key in data.cosmo_arguments.keys():
                if key in data.parameters.keys():
                    OLE_state['parameters'][key] = np.array([data.cosmo_arguments[key]])

            # check if we reuqire a quality check
            if self.emulator.require_quality_check(OLE_state['parameters']):
                # if yes. We sample multiple emulator states and estiamte the accuracy on the loglikelihood prediction!
                # if not, we trust the emulator and we can just run it
                # sample multiple spectra
                #emulator_sample_states = self.emulator.emulate_samples(emulator_state['parameters'])
                local_key = jax.random.PRNGKey(time.time_ns())
                emulator_sample_states, _ = self.emulator.emulate_samples(OLE_state['parameters'], local_key)

                # compute the likelihoods
                emulator_sample_loglikes = []
                for emulator_sample_state in emulator_sample_states:

                    # Translate the emulator state to MP state
                    MP_sample_state = self.OLE_state_to_MP_state(emulator_sample_state)

                    lkl = 0.0
                    for likelihood in dictvalues(data.lkl):
                        value = likelihood.loglkl(self, data)
                        lkl += value

                    emulator_sample_loglikes.append(lkl)

                emulator_sample_loglikes = np.array(emulator_sample_loglikes)

                # we also need the reference likelihood which is going to be used eventually
                predictions = self.emulator.emulate(OLE_state['parameters'])
                MP_sample_state = self.OLE_state_to_MP_state(predictions)

                reference_loglike = 0.0
                for likelihood in dictvalues(data.lkl):
                    value = likelihood.loglkl(self, data)
                    reference_loglike += value

                # check whether the emulator is good enough
                if not self.emulator.check_quality_criterium(emulator_sample_loglikes, parameters=OLE_state['parameters'], reference_loglike = reference_loglike):
                    return False
                else:
                    # Add the point to the quality points
                    self.emulator.add_quality_point(OLE_state['parameters'])
                    return True

            else:
                # if we do not require a quality check, we can just run the emulator
                predictions = self.emulator.emulate(OLE_state['parameters'])
                MP_sample_state = self.OLE_state_to_MP_state(predictions)
                return True
            


    classy.Class_OLE = Class_OLE


    # HERE WE MODIFY MP

    def OLE_recover_cosmological_module(data):
        """
        From the cosmological module name, initialise the proper Boltzmann code

        .. note::

            Only CLASS is currently wrapped, but a python wrapper of CosmoMC should
            enter here.

        """
        # Importing the python-wrapped CLASS from the correct folder, defined in
        # the .conf file, or overwritten at this point by the log.param.
        # If the cosmological code is CLASS, do the following to import all
        # relevant quantities
        if data.cosmological_module_name == 'CLASS':
            try:
                classy_path = ''
                for elem in os.listdir(os.path.join(
                        data.path['cosmo'], "python", "build")):
                    if elem.find("lib.") != -1:
                        classy_path = os.path.join(
                            data.path['cosmo'], "python", "build", elem)
                        break
            except OSError:
                raise io_mp.ConfigurationError(
                    "You probably did not compile the python wrapper of CLASS. " +
                    "Please go to /path/to/class/python/ and do\n" +
                    "..]$ python setup.py build")

            # Inserting the previously found path into the list of folders to
            # search for python modules.
            sys.path.insert(1, classy_path)
            try:
                from classy import Class_OLE
            except ImportError:
                raise io_mp.MissingLibraryError(
                    "You must have compiled the classy.pyx file. Please go to " +
                    "/path/to/class/python and run the command\n " +
                    "python setup.py build")
            cosmo = Class_OLE()
        else:
            raise io_mp.ConfigurationError(
                "Unrecognised cosmological module. " +
                "Be sure to define the correct behaviour in MontePython.py " +
                "and data.py, to support a new one.")

        return cosmo

    # overwrite MP's recover_cosmological_module with our own
    import initialise
    initialise.recover_cosmological_module = OLE_recover_cosmological_module


    def compute_lkl_OLE(cosmo, data):
        """
        Compute the likelihood, given the current point in parameter space.

        This function now performs a test before calling the cosmological model
        (**new in version 1.2**). If any cosmological parameter changed, the flag
        :code:`data.need_cosmo_update` will be set to :code:`True`, from the
        routine :func:`check_for_slow_step <data.Data.check_for_slow_step>`.

        Returns
        -------
        loglike : float
            The log of the likelihood (:math:`\\frac{-\chi^2}2`) computed from the
            sum of the likelihoods of the experiments specified in the input
            parameter file.

            This function returns :attr:`data.boundary_loglike
            <data.data.boundary_loglike>`, defined in the module :mod:`data` if
            *i)* the current point in the parameter space has hit a prior edge, or
            *ii)* the cosmological module failed to compute the model. This value
            is chosen to be extremly small (large negative value), so that the step
            will always be rejected.


        """
        from classy import CosmoSevereError, CosmoComputationError

        # If the cosmological module has already been called once, and if the
        # cosmological parameters have changed, then clean up, and compute.
        # if cosmo.state and data.need_cosmo_update is True:
        #     cosmo.struct_cleanup()

        # If the data needs to change, then do a normal call to the cosmological
        # compute function. Note that, even if need_cosmo update is True, this
        # function must be called if the jumping factor is set to zero. Indeed,
        # this means the code is called for only one point, to set the fiducial
        # model.
        emulation_success = None
        if ((data.need_cosmo_update) or
                (not cosmo.state) or
                (data.jumping_factor == 0)):
            # Prepare the cosmological module with the new set of parameters
            cosmo.set(data.cosmo_arguments)

            # If possible run the emulator
            emulation_success = cosmo.emulate_MP(data)
            if not emulation_success:
                cosmo.use_emulated_result = False

            # If emulation was not possible for some reason, compute the model
            if not emulation_success:
                # Compute the model, keeping track of the errors

                # In classy.pyx, we made use of two type of python errors, to handle
                # two different situations.
                # - CosmoSevereError is returned if a parameter was not properly set
                # during the initialisation (for instance, you entered Ommega_cdm
                # instead of Omega_cdm).  Then, the code exits, to prevent running with
                # imaginary parameters. This behaviour is also used in case you want to
                # kill the process.
                # - CosmoComputationError is returned if Class fails to compute the
                # output given the parameter values. This will be considered as a valid
                # point, but with minimum likelihood, so will be rejected, resulting in
                # the choice of a new point.
                try:
                    data.cosmo_arguments['output']
                except:
                    data.cosmo_arguments.update({'output': ''})
                if 'SZ' in data.cosmo_arguments['output']:
                    try:
                        if 'SZ_counts':
                            cosmo.compute(["szcount"])
                        else:
                            cosmo.compute(["szpowerspectrum"])
                    except CosmoComputationError as failure_message:
                        sys.stderr.write(str(failure_message)+'\n')
                        sys.stderr.flush()
                        return data.boundary_loglike
                    except CosmoSevereError as critical_message:
                        raise io_mp.CosmologicalModuleError(
                            "Something went wrong when calling CLASS" +
                            str(critical_message))
                    except KeyboardInterrupt:
                        raise io_mp.CosmologicalModuleError(
                            "You interrupted execution")
                else:
                    try:
                        cosmo.compute(["lensing"])
                    except CosmoComputationError as failure_message:
                        # could be useful to uncomment for debugging:
                        #np.set_printoptions(precision=30, linewidth=150)
                        #print('cosmo params')
                        #print(data.cosmo_arguments)
                        #print(data.cosmo_arguments['tau_reio'])
                        sys.stderr.write(str(failure_message)+'\n')
                        sys.stderr.flush()
                        return data.boundary_loglike
                    except CosmoSevereError as critical_message:
                        raise io_mp.CosmologicalModuleError(
                            "Something went wrong when calling CLASS" +
                            str(critical_message))
                    except KeyboardInterrupt:
                        raise io_mp.CosmologicalModuleError(
                            "You interrupted execution")


        # For each desired likelihood, compute its value against the theoretical
        # model
        loglike = 0
        # This flag holds the information whether a fiducial model was written. In
        # this case, the log likelihood returned will be '1j', meaning the
        # imaginary number i.
        flag_wrote_fiducial = 0

        for likelihood in dictvalues(data.lkl):
            if likelihood.need_update is True:
                value = likelihood.loglkl(cosmo, data)
                # Storing the result
                likelihood.backup_value = value
            # Otherwise, take the existing value
            else:
                value = likelihood.backup_value
            if data.command_line.display_each_chi2:
                print("-> for ",likelihood.name,":  loglkl=",value,",  chi2eff=",-2.*value)
            loglike += value
            # In case the fiducial file was written, store this information
            if value == 1j:
                flag_wrote_fiducial += 1
        if data.command_line.display_each_chi2:
            print("-> Total:  loglkl=",loglike,",  chi2eff=",-2.*loglike)

        # Compute the derived parameters if relevant
        if data.get_mcmc_parameters(['derived']) != []:
            try:
                derived = cosmo.get_current_derived_parameters(
                    data.get_mcmc_parameters(['derived']))
                for name, value in dictitems(derived):
                    data.mcmc_parameters[name]['current'] = value
            except AttributeError:
                # This happens if the classy wrapper is still using the old
                # convention, expecting data as the input parameter
                cosmo.get_current_derived_parameters(data)
            except CosmoSevereError:
                raise io_mp.CosmologicalModuleError(
                    "Could not write the current derived parameters")

        # DCH adding a check to make sure the derived_lkl are passed properly
        if data.get_mcmc_parameters(['derived_lkl']) != []:
            try:
                for (name, value) in data.derived_lkl.items():
                    data.mcmc_parameters[name]['current'] = value
            except Exception as missing:
                raise io_mp.CosmologicalModuleError(
                    "You requested derived_lkl parameters, but you are missing the following ones in your param file:" + str(missing))

        for elem in data.get_mcmc_parameters(['derived']):
            data.mcmc_parameters[elem]['current'] /= \
                data.mcmc_parameters[elem]['scale']

        # If fiducial files were created, inform the user, and exit
        if flag_wrote_fiducial > 0:
            if flag_wrote_fiducial == len(data.lkl):
                raise io_mp.FiducialModelWritten(
                    "This is not an error but a normal abort, because " +
                    "fiducial file(s) was(were) created. " +
                    "You may now start a new run. ")
            else:
                raise io_mp.FiducialModelWritten(
                    "This is not an error but a normal abort, because " +
                    "fiducial file(s) was(were) created. " +
                    "However, be careful !!! Some previously non-existing " +
                    "fiducial files were created, but potentially not all of them. " +
                    "Some older fiducial files will keep being used. If you have doubts, " +
                    "you are advised to check manually in the headers of the " +
                    "corresponding files that all fiducial parameters are consistent "+
                    "with each other. If everything looks fine, "
                    "you may now start a new run.")

        # Check if we can build the emulator here
        if cosmo.emulator is None:
            # get initial state
            initial_state = cosmo.MP_state_to_OLE_state(data, cosmo.attributes_with_relevant_output)

            # select parameters taht are in data.cosmo_arguments and are keys in data.parameters
            for key in data.cosmo_arguments.keys():
                if key in data.parameters.keys():
                    initial_state['parameters'][key] = np.array([data.cosmo_arguments[key]])

            # get initial output
            initial_state['loglike'] = np.array([loglike])
            
            from OLE.emulator import Emulator
            cosmo.emulator = Emulator(**data.emulator_settings)
            cosmo.emulator.initialize(ini_state=initial_state, **data.emulator_settings)
        
        # Add state to emulator if it wasnt generated by the emulator
        if emulation_success is not None:
            if not emulation_success:
                emulator_state = cosmo.MP_state_to_OLE_state(data, cosmo.attributes_with_relevant_output)
                emulator_state['loglike'] = np.array([loglike])
                added_flag = cosmo.emulator.add_state(emulator_state)

        # if the emulator is not trained, train it, if enough states are available
        if not cosmo.emulator.trained:
            if len(cosmo.emulator.data_cache.states) >= cosmo.emulator.hyperparameters['min_data_points']:
                cosmo.emulator.train()


        return loglike/data.command_line.temperature

    # replace sampler.compute_lkl by compute_lkl_OLE

    import sampler
    sampler.compute_lkl = compute_lkl_OLE

    # Default action when facing a warning is being remapped to a custom one
    warnings.showwarning = io_mp.warning_message

    # MPI is tested for, and if a different than one number of cores is found,
    # it runs mpi_run instead of a simple run.
    MPI_ASKED = False
    try:
        from mpi4py import MPI
        NPROCS = MPI.COMM_WORLD.Get_size()
        if NPROCS > 1:
            MPI_ASKED = True
    # If the ImportError is raised, it only means that the Python wrapper for
    # MPI is not installed - the code should simply proceed with a non-parallel
    # execution.
    except ImportError:
        pass

    if MPI_ASKED:
        # This import has to be there in case MPI is not installed
        from run import mpi_run
        sys.exit(mpi_run())
    else:
        sys.exit(run())
