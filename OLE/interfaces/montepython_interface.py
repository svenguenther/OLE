# This is a MontePython - OLE wrapper. It reads your MontePython version and adds OLE functions to it.


"""
.. module:: MontePython
   :synopsis: Main module
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Monte Python, a Monte Carlo Markov Chain code (with Class!)
"""
import sys
import warnings
import os

import numpy as np

#####################################
#  Set the path to MontePython here #
#####################################

interface_path = os.path.dirname(__file__)
MP_path_filepath = os.path.join(interface_path, 'MP_PATH')

with open(MP_path_filepath, 'r') as file:
    MP_path = file.read().rstrip()
    MP_path = os.path.join(MP_path,'montepython')

# MP_path = '/home/path/to/montepython_public'


# -----------------MAIN-CALL---------------------------------------------
if __name__ == '__main__':
    # use naive vanilla parser to get the path to montepython

    # check that you accutally DID change the path to montepython
    if MP_path == '/path/to/your/montepython_public':
        raise ValueError("Please change the path to your montepython version in the /OLE/interfaces/MP_PATH file")
    sys.path.insert(0, MP_path)

    import io_mp       # all the input/output mechanisms
    from run import run

    # import os

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

    CLASS_functions_no_in_float_out = ['Omega_m','Omega_r','Omega_Lambda',
                               'Omega_g','Omega_b', 'h', 'T_cmb','age',
                               'n_s','tau_reio','theta_s_100','theta_star_100',
                               'omega_b','Neff','k_eq','z_eq','sigma8',
                               'S8','sigma8_cb','rs_drag','z_reio','T_cmb',
                               'Omega0_m','Omega0_k','Omega0_cdm','Omega0_idm',
                               ] # no input, float output
    CLASS_functions_no_in_array_out = ['spectral_distortion_amplitudes'] # no input, array output
    CLASS_functions_no_in_tuple_out = ['spectral_distortion']  # no input, output: tuple( np.array, np.array) 
    CLASS_functions_array_in_tuple_out = ['z_of_r']  # input: np.array, output: tuple( np.array, np.array) 
    CLASS_functions_array_in_array_out = ['get_pk','get_pk_cb', 'get_pk_lin','get_pk_all', 
                               'get_Pk_cb_m_ratio', 'get_pk_cb_lin','luminosity_distance',
                               'get_tk','nonlinear_scale','nonlinear_scale_cb',
                               'fourier_hmcode_sigma8','fourier_hmcode_sigma8_cb',
                               'fourier_hmcode_sigmadisp','fourier_hmcode_sigmadisp_cb',
                               'fourier_hmcode_sigmadisp100','fourier_hmcode_sigmadisp100_cb',
                               'fourier_hmcode_sigmaprime','fourier_hmcode_sigmaprime_cb',
                               'get_pk_array','get_pk_cb_array']  # input: np.array, output: N-d np.array
    CLASS_functions_any_in_flaot_out = ['angular_distance','pk','pk_cb', 'pk_lin', 
                               'pk_cb_lin','pk_tilt', 'sigma_cb', 'sigma', 
                               'Hubble','fourier_hmcode_window_nfw',
                               'angular_distance_from_to','comoving_distance',
                               'scale_independent_growth_factor',
                               'scale_independent_growth_factor_f',
                               'scale_dependent_growth_factor_f',
                               'scale_independent_f_sigma8','z_of_tau',
                               'effective_f_sigma8','effective_f_sigma8_spline',
                               'Hubble','Om_m','Om_b','Om_cdm','Om_ncdm',
                               'ionization_fraction','baryon_temperature'] # input: dict/array/float, output: float
    CLASS_functions_Cl_dict = ['lensed_cl','raw_cl','density_cl'] # input: int, output: cl_dict
    CLASS_functions_list_string_in_dict_out = ['get_current_derived_parameters'] # input: list of strings, output: dict with floats

    
    
    

    ALL_CLASS_functions = CLASS_functions_no_in_float_out + CLASS_functions_no_in_array_out + CLASS_functions_no_in_tuple_out + CLASS_functions_array_in_tuple_out + CLASS_functions_array_in_array_out + CLASS_functions_any_in_flaot_out + CLASS_functions_Cl_dict + CLASS_functions_list_string_in_dict_out


    for name in dir(classy.Class):
        if callable(getattr(classy.Class, name)) and not name.startswith("__"):
            def new_method(self, *args, **kwargs):
                my_attribute = self.current_attribute                    

                if my_attribute=='compute':
                    self.attributes_with_relevant_output = {} # In this dictionary we store the calls of classy
                    self.tuple_flag = {}

                if my_attribute=='set':
                    self.number_of_compute_calls += 1
                    for likelihood in self.all_likelihoods:
                        self.attribute_couter[likelihood] = {}
                        for key in self.attribute_max_couter[likelihood].keys():
                            self.attribute_couter[likelihood][key] = 0 # start counting the number of calls to each attribute from scratch

                # if the current attribute is not in the list of attributes with relevant output, we do not want to call the method
                if self.use_emulated_result and my_attribute in self.emulated_result.keys():
                    # 
                    # copy emulated result into res
                    # 

                    # this works for all functions but 'get_current_derived_parameters'. This function is also called by the sampler itself. It will be sorted into the last likelihood.
                    if my_attribute == 'get_current_derived_parameters':
                        res_OLE_shape = cp.deepcopy(self.emulated_result[my_attribute])
                        info = self.attribute_input_args[self.all_likelihoods[-1]][my_attribute][0]
                        res, _ = self.transform_from_OLE_to_CLASSY(my_attribute, res_OLE_shape, info)
                    else:
                        # cut our chunk of the emulated result
                        lower_index = self.indexing[self.current_likelihood][my_attribute][self.attribute_couter[self.current_likelihood][my_attribute]]
                        upper_index = self.indexing[self.current_likelihood][my_attribute][self.attribute_couter[self.current_likelihood][my_attribute]+1]

                        # the extra info can provide shape and other things that allows to shape the OLE output to the CLASS output
                        info = self.attribute_input_args[self.current_likelihood][my_attribute][self.attribute_couter[self.current_likelihood][my_attribute]]

                        res_OLE_shape = cp.deepcopy(self.emulated_result[my_attribute][lower_index:upper_index])

                        # transform the OLE output to the CLASS output
                        res, _ = self.transform_from_OLE_to_CLASSY(my_attribute, res_OLE_shape, info)
                        self.attribute_couter[self.current_likelihood][my_attribute] += 1

                        # if for example we use oversampling, we need to provide the same output multiple times. In that case we need to start counting from scratch
                        if self.attribute_couter[self.current_likelihood][my_attribute] == self.attribute_max_couter[self.current_likelihood][my_attribute]:
                            self.attribute_couter[self.current_likelihood][my_attribute] = 0
                else:
                    # USE THE ORIGINAL CLASS
                    res = self.cosmo.__getattribute__(my_attribute)(*args, **kwargs)
                    res_OLE_shape, extra_info = self.transform_from_CLASSY_to_OLE(my_attribute, res)

                    
                    # If we run the pipline for the first time, we need to count the number of calls to each attribute
                    if self.number_of_compute_calls==1: # count the total number of calls to each attribute

                        # only check implemented functions
                        if my_attribute in ALL_CLASS_functions:

                            # check if the attribute was already called earlier
                            if self.current_likelihood not in self.attribute_max_couter.keys():
                                self.attribute_max_couter[self.current_likelihood] = {}
                                self.attribute_input_args[self.current_likelihood] = {}
                                self.indexing[self.current_likelihood] = {}
                                self.all_likelihoods.append(self.current_likelihood)

                            # check if the attribute was already called earlier
                            if my_attribute not in self.attribute_max_couter[self.current_likelihood].keys():
                                self.attribute_max_couter[self.current_likelihood][my_attribute] = 0
                                self.attribute_input_args[self.current_likelihood][my_attribute] = []

                                # check if the attribute was already called by another likelihood
                                max_index = 0
                                for likelihood in self.all_likelihoods:
                                    if likelihood != self.current_likelihood:
                                        if my_attribute in self.attribute_max_couter[likelihood].keys():
                                            max_index = max(max_index, self.indexing[likelihood][my_attribute][-1])


                                self.indexing[self.current_likelihood][my_attribute] = [max_index, max_index + len(res_OLE_shape)]
                            else:
                                self.indexing[self.current_likelihood][my_attribute].append(self.indexing[self.current_likelihood][my_attribute][-1] + len(res_OLE_shape))

                            # store the number of input arguments
                            self.attribute_max_couter[self.current_likelihood][my_attribute] += 1
                            self.attribute_input_args[self.current_likelihood][my_attribute].append((len(res_OLE_shape), extra_info))                                

                        elif my_attribute in ['set','compute']:
                            pass

                        else:
                            raise ValueError(f'CLASS function {my_attribute} is not implemented in the OLE wrapper')
                        
                    # For all calls:
                    if my_attribute in ALL_CLASS_functions:

                        # if it is the first call since the compute function was called, we need to create the output entry
                        if my_attribute not in self.attributes_with_relevant_output.keys():
                            self.attributes_with_relevant_output[my_attribute] = cp.deepcopy(res_OLE_shape)
                        else:
                            # fill the output entry
                            self.attributes_with_relevant_output[my_attribute] = np.append(self.attributes_with_relevant_output[my_attribute],cp.deepcopy(res_OLE_shape))

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
            self.attribute_input_args = {} # this is a dictonary with the attributes and their input arguments
            self.attribute_couter = {} # this is a dictonary with the attributes and the number of times they were called in the current likelihood calculation
            self.number_of_compute_calls = 0
            self.attributes_with_relevant_output = {} # this is a dictonary with the relevant attributes and their output
            self.use_emulated_result = False
            self.emulated_result = {}
            self.current_likelihood = None
            self.all_likelihoods = []
            self.indexing = {} # this gives a mapping from the output of the OLE wrapper to the output of the CLASS code

            np.set_printoptions(precision=8)

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
        
        def transform_from_OLE_to_CLASSY(self, attribute, res, info):
            # This function reads the original output of the OLE code and transforms it into the CLASS format. 
            # It might give extra information about the transformation such as shape etc...

            if attribute in CLASS_functions_no_in_float_out:
                return res[0], None
            
            elif attribute in CLASS_functions_no_in_array_out:
                # get entry with the same arg_hash
                res_shape = info[1]['shape']

                # reshape the output
                res = res.reshape(res_shape)

                return res, None
            
            elif attribute in CLASS_functions_no_in_tuple_out:
                # split the output into two arrays
                out1, out2 = np.split(res, 2)
                return (out1,out2), None
            
            elif attribute in CLASS_functions_array_in_tuple_out:
                # split the output into two arrays
                out1, out2 = np.split(res, 2)
                return (out1,out2), None
            
            elif attribute in CLASS_functions_array_in_array_out:
                # get entry with the same arg_hash
                res_shape = info[1]['shape']

                # reshape the output
                res = res.reshape(res_shape)

                return res, None
            
            elif attribute in CLASS_functions_any_in_flaot_out:
                return res[0], None

            elif attribute in CLASS_functions_Cl_dict:
                # split the output into the different keys
                out = {}
                for i, key in enumerate(info[1]['keys']):
                    out[key] = res[i*(info[1]['max_l']+1):(i+1)*(info[1]['max_l']+1)]

                return out, info
            
            elif attribute in CLASS_functions_list_string_in_dict_out:
                # split the output into the different keys
                out = {}
                for i, key in enumerate(info[1]['keys']):
                    out[key] = res[i]

                return out, info
            
            elif attribute in ['set','compute']:
                return res, None
            
            else:
                raise ValueError(f'CLASS function {attribute} is not implemented in the OLE wrapper')

        
        def transform_from_CLASSY_to_OLE(self, attribute, res):
            # This function reads the original output of the CLASS code and transforms it into the OLE format. 
            # It might give extra information about the transformation such as shape etc...

            if attribute in CLASS_functions_no_in_float_out:
                return np.array([res]), None
            
            elif attribute in CLASS_functions_no_in_array_out:
                # save shape of the output
                res_shape = res.shape

                # flatten the output
                res = res.flatten()

                return res, {'shape': res_shape}
            
            elif attribute in CLASS_functions_no_in_tuple_out:
                out = np.hstack([res[0],res[1]])
                return out, None
            
            elif attribute in CLASS_functions_array_in_tuple_out:
                out = np.hstack([res[0],res[1]])
                return out, None
            
            elif attribute in CLASS_functions_array_in_array_out:
                # save shape of the output
                res_shape = res.shape

                # flatten the output
                res = res.flatten()

                return res, {'shape': res_shape}
            
            elif attribute in CLASS_functions_any_in_flaot_out:
                return np.array([res]), None
            
            elif attribute in CLASS_functions_Cl_dict:
                info = {'keys': list(res.keys()), 'max_l': max(res['ell'])}
                # flatten the out
                res = np.hstack([res[key] for key in info['keys']])

                return res, info
            
            elif attribute in CLASS_functions_list_string_in_dict_out:
                info = {'keys': list(res.keys())}
                res = np.array([res[key] for key in info['keys']])

                return res, info
            
                        
            elif attribute in ['set','compute']:
                return res, None

            else:
                raise ValueError(f'CLASS function {attribute} is not implemented in the OLE wrapper')
        

        def MP_state_to_OLE_state(self, data, MP_state):
            OLE_state = {'parameters': {},
                        'quantities': {},
                        'total_loglike': None}

            self.state = cp.deepcopy(MP_state)

            # go through all MP elements and addthem to the OLE quantities
            for key, value in MP_state.items():

                # here we translate the lensed_cl into the different ingredients
                if key == 'lensed_cl':
                    for likelihood in self.all_likelihoods:
                        if 'lensed_cl' in self.attribute_input_args[likelihood].keys():
                            info = self.attribute_input_args[likelihood]['lensed_cl'][0]


                    for i, key2 in enumerate(info[1]['keys']):
                        OLE_state['quantities'][key2] = MP_state['lensed_cl'][i*(info[1]['max_l']+1):(i+1)*(info[1]['max_l']+1)]
                else:
                    OLE_state['quantities'][key] = value

            # add parameters from data
            np.set_printoptions(precision=8)
            for key in data.get_mcmc_parameters(['cosmo']):
                OLE_state['parameters'][key] = np.array([data.mcmc_parameters[key]['current'] * data.mcmc_parameters[key]['scale']*1.0])

            return OLE_state

        def OLE_state_to_MP_state(self, OLE_state):
            # we deepcopy the
            self.emulated_result = cp.deepcopy(self.attributes_with_relevant_output)

            # we now go through all branches of the emulated result and update the values
            for key, value in self.emulated_result.items():

                # same as above:
                if key == 'lensed_cl':
                    N_cl_dependent_likelihoods = 0
                    for likelihood in self.all_likelihoods:
                        if 'lensed_cl' in self.attribute_input_args[likelihood].keys():
                            info = self.attribute_input_args[likelihood]['lensed_cl'][0]
                            N_cl_dependent_likelihoods += 1

                    for j in range(N_cl_dependent_likelihoods):
                        index_offset = j*(info[1]['max_l']+1)*len(info[1]['keys'])
                        for i, key2 in enumerate(info[1]['keys']):
                            self.emulated_result[key][index_offset+i*(info[1]['max_l']+1):index_offset+(i+1)*(info[1]['max_l']+1)] = OLE_state['quantities'][key2]
                else:
                    self.emulated_result[key] = np.array(OLE_state['quantities'][key])
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
                            'total_loglike': None}

            # fill parameters from data
            np.set_printoptions(precision=8)
            for key in data.get_mcmc_parameters(['cosmo']):
                OLE_state['parameters'][key] = np.array([data.mcmc_parameters[key]['current'] * data.mcmc_parameters[key]['scale']])

            # check if we reuqire a quality check
            if self.emulator.require_quality_check(OLE_state['parameters']):
                # if yes. We sample multiple emulator states and estiamte the accuracy on the loglikelihood prediction!
                # if not, we trust the emulator and we can just run it
                # sample multiple spectra
                #emulator_sample_states = self.emulator.emulate_samples(emulator_state['parameters'])
                local_key = jax.random.PRNGKey(time.time_ns())
                emulator_sample_states, _ = self.emulator.emulate_samples(OLE_state['parameters'], local_key)

                # start the likelihood_testing counter
                self.emulator.start("likelihood_testing")

                # compute the likelihoods
                emulator_sample_loglikes = []
                for emulator_sample_state in emulator_sample_states:

                    # Translate the emulator state to MP state
                    MP_sample_state = self.OLE_state_to_MP_state(emulator_sample_state)

                    lkl = 0.0
                    for likelihood in dictvalues(data.lkl):
                        self.current_likelihood = likelihood.name
                        value = likelihood.loglkl(self, data)
                        lkl += value

                    emulator_sample_loglikes.append(lkl)

                emulator_sample_loglikes = np.array(emulator_sample_loglikes)

                # increment the likelihood_testing counter
                self.emulator.increment("likelihood_testing")

                # we also need the reference likelihood which is going to be used eventually
                predictions = self.emulator.emulate(OLE_state['parameters'])
                MP_sample_state = self.OLE_state_to_MP_state(predictions)



                reference_loglike = 0.0
                
                self.emulator.start("likelihood_testing")
                for likelihood in dictvalues(data.lkl):
                    self.current_likelihood = likelihood.name
                    value = likelihood.loglkl(self, data)
                    reference_loglike += value
                self.emulator.increment("likelihood_testing")

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
        total_loglike : float
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
                if cosmo.emulator is not None:
                    cosmo.emulator.start('theory_code')
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
                if cosmo.emulator is not None:
                    cosmo.emulator.increment('theory_code')
                    cosmo.emulator.print_status()


        # For each desired likelihood, compute its value against the theoretical
        # model
        if cosmo.emulator is not None:
            cosmo.emulator.start('likelihood')
        loglike = 0
        # This flag holds the information whether a fiducial model was written. In
        # this case, the log likelihood returned will be '1j', meaning the
        # imaginary number i.
        flag_wrote_fiducial = 0

        for likelihood in dictvalues(data.lkl):
            if likelihood.need_update is True:
                cosmo.current_likelihood = likelihood.name
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

        if cosmo.emulator is not None:
            cosmo.emulator.increment('likelihood')

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

            # select varying cosmological parameters
            for key in data.get_mcmc_parameters(['cosmo']):
                initial_state['parameters'][key] = \
                    np.array([data.mcmc_parameters[key]['current'] * \
                              data.mcmc_parameters[key]['scale']])

            # get initial output
            initial_state['total_loglike'] = np.array([loglike])

            from OLE.emulator import Emulator
            cosmo.emulator = Emulator(**data.emulator_settings)
            cosmo.emulator.initialize(ini_state=initial_state, **data.emulator_settings)

        # Add state to emulator if it wasnt generated by the emulator
        elif emulation_success is not None:
            if not emulation_success:
                emulator_state = cosmo.MP_state_to_OLE_state(data, cosmo.attributes_with_relevant_output)
                emulator_state['total_loglike'] = np.array([loglike])
                added_flag = cosmo.emulator.add_state(emulator_state)

        # if the emulator is not trained, train it, if enough states are available
        if not cosmo.emulator.trained:
            if len(cosmo.emulator.data_cache.states) >= cosmo.emulator.hyperparameters['min_data_points']:
                cosmo.emulator.train()


        if (data.jumping != 'global') and (cosmo.emulator.rejit_flag_emulator==False):
            # change the sampling method to 'global' sampling to accelerate the MCMC itself :)
            data.jumping = 'global'
                
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
