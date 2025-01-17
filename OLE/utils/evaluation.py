# here are some useful tools and functions to evaluate the performance of OLE.
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import gc

event_dict = {'likelihood': 'likelihood evaluation', 'theory_code': 'CLASS', 'add_state': 'Cache interaction', 'train': 'train emulator', 'emulate_samples': 'sample testing states', 'likelihood_testing':'run likelihood for testing', 'emulate':'evaluate emulator', 'update': 'updating emulator'}

# change style of plots
# use serif 
plt.style.use('ggplot')

# This function takes the logfile of an OLE run and creates usful statistics and plots.

def load_logfile(logfile_path):
    # read logfile
    with open(logfile_path, 'r') as f:
        lines = f.readlines()

    # there are N events, they can come with a different number of lines
    
    # thus we need to identify the different events

    # possible events are:


    # 1. Successful adding a new state. The lines will look like:

    # 2025-01-08 16:02:20 State added to emulator: A_s: [2.12164569e-09]; N_ur: [2.08078396]; Omega_k: [-0.00158179]; w0_fld: [-0.55431533]; wa_fld: [-1.24543999]; h: [0.64656423]; n_s: [0.96927103]; omega_b: [0.02262131]; omega_cdm: [0.11837504]; tau_reio: [0.05797674] at loglike: [-2044.35640691] max. loglike: -2044.3564069112174
    # 2025-01-08 16:02:20 Current data cache size: 2

    # Here we want to extract the timestamp, the parameter names and values, the loglike and the max loglike. Additionally in the next line we want to extract the current data cache size.


    # 2. Not adding a new state. The lines will look like:

    # 2025-01-08 16:03:08 State not added to emulator: A_s: [2.13995974e-09]; N_ur: [1.97225448]; Omega_k: [-0.00220065]; w0_fld: [-0.50189047]; wa_fld: [-1.13416006]; h: [0.62253727]; n_s: [0.96923092]; omega_b: [0.02287428]; omega_cdm: [0.11420609]; tau_reio: [0.06757026] at loglike: [-2010.50667361] max. loglike: -1957.7798079628894
    # 2025-01-08 16:03:08 Current data cache size: 6

    # Here we want to extract the timestamp, the parameter names and values, the loglike and the max loglike. Additionally in the next line we want to extract the current data cache size.


    # 3. Training the emulator. The lines will look like:

    # 2025-01-08 16:09:44 Training emulator

    # Here we want to extract the timestamp.


    # 4. Succesfully evaluating the emulator. The lines will look like:

    # 2025-01-08 16:12:18 Emulated state: A_s: [2.04184147e-09] N_ur: [1.76437728] Omega_k: [-0.00209169] w0_fld: [-0.83821857] wa_fld: [-0.88649299] h: [0.67278963] n_s: [0.95680798] omega_b: [0.02224177] omega_cdm: [0.11682955] tau_reio: [0.05801073]
    # 2025-01-08 16:12:19 Quality criterium fulfilled; A_s: [2.04184147e-09]; N_ur: [1.76437728]; Omega_k: [-0.00209169]; w0_fld: [-0.83821857]; wa_fld: [-0.88649299]; h: [0.67278963]; n_s: [0.95680798]; omega_b: [0.02224177]; omega_cdm: [0.11682955]; tau_reio: [0.05801073] Max loglike: -1924.864692, Reference loglike: -2023.376961, delta loglikes: -2023.2454345063952 -2024.1368312453064 -2022.9861117099408 -2023.4816996325576 -2022.1590204821705

    # Here we want to extract the timestamp, the parameter names and values, the max loglike, the reference loglike and the delta loglikes. Additionally in the next line we want to extract the emulated state.


    # 5. Unsuccessfully evaluating the emulator. The lines will look like:

    # 2025-01-08 16:23:26 Emulated state: A_s: [2.09208049e-09] N_ur: [1.76222719] Omega_k: [0.00078406] w0_fld: [-0.97056177] wa_fld: [-0.19497179] h: [0.67614466] n_s: [0.95590118] omega_b: [0.02233593] omega_cdm: [0.11409116] tau_reio: [0.05745564]
    # 2025-01-08 16:23:30 Quality criterium NOT fulfilled; A_s: [2.09208049e-09]; N_ur: [1.76222719]; Omega_k: [0.00078406]; w0_fld: [-0.97056177]; wa_fld: [-0.19497179]; h: [0.67614466]; n_s: [0.95590118]; omega_b: [0.02233593]; omega_cdm: [0.11409116]; tau_reio: [0.05745564] Max loglike: -1916.756999, Reference loglike: -1912.979046, delta loglikes: -1913.9999968967759 -1913.7105333735678 -1913.7177631822021 -1913.9003592767394 -1913.9196168452136

    # Here we want to extract the timestamp, the parameter names and values, the max loglike, the reference loglike and the delta loglikes. Additionally in the next line we want to extract the emulated state.


    # 6. Running the emulator without testing. The lines will look like:
    # 2025-01-08 16:30:44 Quality check not required. Test emulator is False 
    # 2025-01-08 16:30:44 Emulated state: A_s: [2.06889013e-09] N_ur: [1.72019626] Omega_k: [-0.00127581] w0_fld: [-1.24377076] wa_fld: [0.30437687] h: [0.70327196] n_s: [0.95274422] omega_b: [0.02247115] omega_cdm: [0.11378999] tau_reio: [0.05858385]

    # Here we want to extract the timestamp and the emulated state.


    # 7. Using the emulator but eventually not using its result since it is far away from the reference. The lines will look like:

    # 2025-01-08 16:12:39 Emulated state: A_s: [2.03204753e-09] N_ur: [2.56535371] Omega_k: [-0.01351084] w0_fld: [-0.91112423] wa_fld: [-1.30297816] h: [0.74366375] n_s: [0.98670492] omega_b: [0.0232239] omega_cdm: [0.1226264] tau_reio: [0.03595985]
    # 2025-01-08 16:12:40 Loglikes are too far away from best-fit point. Not using OLE; A_s: [2.03204753e-09]; N_ur: [2.56535371]; Omega_k: [-0.01351084]; w0_fld: [-0.91112423]; wa_fld: [-1.30297816]; h: [0.74366375]; n_s: [0.98670492]; omega_b: [0.0232239]; omega_cdm: [0.1226264]; tau_reio: [0.03595985] Max loglike: -1924.490800, Reference loglike: -2288.280161, delta loglikes: -2298.7242653023104 -2287.4768098623813 -2274.732069733664 -2271.862059438489 -2301.328913000306

    # Here we want to extract the timestamp, the parameter names and values, the max loglike, the reference loglike and the delta loglikes.


    # 8. Updating the emulator. The lines will look like:

    # 2025-01-08 16:14:20 Update emulator

    # Here we want to extract the timestamp.


    # 9. Setting groundlevel probability. The lines will look like:

    # 2025-01-08 16:17:25 Groundlevel testing probability set to 0.05051514007229092

    # Here we want to extract the timestamp and the probability.


    # 10. Printing status update. The lines will look like:

    # 2025-01-08 17:55:14 Emulator status: [08/01/25 17:55:14]
    # 2025-01-08 17:55:14 Number of data points in cache: 151
    # 2025-01-08 17:55:14 Number of emulation calls: 56600
    # 2025-01-08 17:55:14 Number of quality check successful calls: 4168
    # 2025-01-08 17:55:14 Number of quality check unsuccessful calls: 25
    # 2025-01-08 17:55:14 Number of not tested calls: 52407
    # 2025-01-08 17:55:14 Time spent in different parts of the code: 
    # 2025-01-08 17:55:14 Timing for add_state: total: 68.2066s calls: 87 avg: 0.7840s 
    # 2025-01-08 17:55:14 Timing for train: total: 947.7833s calls: 6 avg: 157.9639s 
    # 2025-01-08 17:55:14 Timing for emulate_samples: total: 495.2022s calls: 4193 avg: 0.1181s 
    # 2025-01-08 17:55:14 Timing for likelihood_testing: total: 1372.3190s calls: 4193 avg: 0.3273s 
    # 2025-01-08 17:55:14 Timing for emulate: total: 390.3868s calls: 56599 avg: 0.0069s 
    # 2025-01-08 17:55:14 Timing for likelihood: total: 3990.4867s calls: 56599 avg: 0.0705s 
    # 2025-01-08 17:55:14 Timing for update: total: 227.6766s calls: 8 avg: 28.4596s 

    # Here we want to extract the timestamp, the number of data points in cache, the number of emulation calls, the number of quality check successful calls, the number of quality check unsuccessful calls, the number of not tested calls.
    # Furthermore we want to extract the time spent in different parts of the code. There we want the name of the event (e.g. add_state, train, emulate_samples, likelihood_testing, emulate, likelihood, update), the total time spent, the number of calls and the average time per call.
    



    # we want to store each type of event in a list of dictionaries
    adding_events = []
    not_adding_events = []
    training_events = []
    successfully_evaluating_events = []
    unsuccessfully_evaluating_events = []
    running_events = []
    not_using_events = []
    updating_events = []
    groundlevel_events = []
    status_events = []

    # now we iterate over the lines and fill the lists
    for i, line in enumerate(lines):
        try:
            if "State added to emulator" in line:
                # extract timestamp, parameter names and values, loglike, max loglike
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]

                # extract parameter names and values
                parameter_names_and_values = [name.split(":")[0] for name in line.split(" ")[6:-6]]

                # every second element is a value
                parameter_names = parameter_names_and_values[::2]
                parameter_values = [float(value.split("[")[1].split("]")[0]) for value in parameter_names_and_values[1::2]]

                # extract loglike and max loglike
                loglike = float( line.split(" ")[-4].split("[")[1].split("]")[0] )
                max_loglike = float(line.split(":")[-1])
                # extract current data cache size
                data_cache_size = int(lines[i+1].split(":")[-1])
                adding_events.append({"type": "adding_event", "timestamp": timestamp, "parameter_names": parameter_names, "parameter_values": parameter_values, "loglike": loglike, "max_loglike": max_loglike, "data_cache_size": data_cache_size})
            elif "State not added to emulator" in line:
                # extract timestamp, parameter names and values, loglike, max loglike
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                # extract parameter names and values
                parameter_names_and_values = [name.split(":")[0] for name in line.split(" ")[7:-6]]
                # every second element is a value
                parameter_names = parameter_names_and_values[::2]
                parameter_values = [float(value.split("[")[1].split("]")[0]) for value in parameter_names_and_values[1::2]]
                # extract loglike and max loglike
                loglike = float( line.split(" ")[-4].split("[")[1].split("]")[0] )
                max_loglike = float(line.split(":")[-1])
                # extract current data cache size
                data_cache_size = int(lines[i+1].split(":")[-1])
                not_adding_events.append({"type": "not_adding_events", "timestamp": timestamp, "parameter_names": parameter_names, "parameter_values": parameter_values, "loglike": loglike, "max_loglike": max_loglike, "data_cache_size": data_cache_size})
            elif "Training emulator" in line:
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                training_events.append({"type": "training_events", "timestamp": timestamp})
            elif "Quality criterium fulfilled" in line:
                # extract timestamp, parameter names and values, max loglike, reference loglike, delta loglikes
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                # extract parameter names and values
                # check if the line before is in the correct format of '2025-01-08 16:12:18 Emulated state: parama1: [value1] param2: [value2] ...'
                if "Emulated state" in lines[i-1]:
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i-1].split(" ")[4:]]
                else:
                    # go through lines until we find the correct line
                    j = 1
                    while "Emulated state" not in lines[i-j]:
                        j += 1
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i-j].split(" ")[4:]]

                # every second element is a value
                parameter_names = parameter_names_and_values[::2]
                parameter_values = [float(value.split("[")[1].split("]")[0]) for value in parameter_names_and_values[1::2]]
                # extract max loglike, reference loglike, delta loglikes
                # split the line and search for 'Max loglike' to get the max loglike
                max_likelihood_position = [i for i, s in enumerate(line.split()) if 'Max' in s][0]
                max_loglike = float(line.split()[max_likelihood_position+2][:-1])
                reference_loglike = float(line.split()[max_likelihood_position+5][:-1])

                delta_loglikes = (line.split()[max_likelihood_position+8:])
                delta_loglikes = [float(loglike) for loglike in delta_loglikes]

                successfully_evaluating_events.append({"type": "successfully_evaluating_events", "timestamp": timestamp, "parameter_names": parameter_names, "parameter_values": parameter_values, "max_loglike": max_loglike, "reference_loglike": reference_loglike, "delta_loglikes": delta_loglikes})
            elif "Quality criterium NOT fulfilled" in line:
                # extract timestamp, parameter names and values, max loglike, reference loglike, delta loglikes
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                # extract parameter names and values
                if "Emulated state" in lines[i-1]:
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i-1].split(" ")[4:]]
                else:
                    # go through lines until we find the correct line
                    j = 1
                    while "Emulated state" not in lines[i-j]:
                        j += 1
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i-j].split(" ")[4:]]
                # every second element is a value
                parameter_names = parameter_names_and_values[::2]
                parameter_values = [float(value.split("[")[1].split("]")[0]) for value in parameter_names_and_values[1::2]]
                # extract max loglike, reference loglike, delta loglikes
                # split the line and search for 'Max loglike' to get the max loglike
                max_likelihood_position = [i for i, s in enumerate(line.split()) if 'Max' in s][0]
                max_loglike = float(line.split()[max_likelihood_position+2][:-1])
                reference_loglike = float(line.split()[max_likelihood_position+5][:-1])

                delta_loglikes = (line.split()[max_likelihood_position+8:])
                delta_loglikes = [float(loglike) for loglike in delta_loglikes]

                unsuccessfully_evaluating_events.append({"type": "unsuccessfully_evaluating_events", "timestamp": timestamp, "parameter_names": parameter_names, "parameter_values": parameter_values, "max_loglike": max_loglike, "reference_loglike": reference_loglike, "delta_loglikes": delta_loglikes})
            elif "Quality check not required" in line:
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]

                # extract parameter names and values
                # check if the line after is in the correct format of '2025-01-08 16:12:18 Emulated state: parama1: [value1] param2: [value2] ...'
                if "Emulated state" in lines[i+1]:
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i+1].split(" ")[4:]]
                else:
                    # go through lines until we find the correct line
                    j = 1
                    while "Emulated state" not in lines[i+j]:
                        j += 1
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i+j].split(" ")[4:]]
                # every second element is a value
                parameter_names = parameter_names_and_values[::2]
                parameter_values = [float(value.split("[")[1].split("]")[0]) for value in parameter_names_and_values[1::2]]
                running_events.append({"type": "running_events", "timestamp": timestamp, "parameter_names": parameter_names, "parameter_values": parameter_values})
            elif "Loglikes are too far away from best-fit point" in line:
                # extract timestamp, parameter names and values, max loglike, reference loglike, delta loglikes
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                # extract parameter names and values
                if "Emulated state" in lines[i-1]:
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i-1].split(" ")[4:]]
                else:
                    # go through lines until we find the correct line
                    j = 1
                    while "Emulated state" not in lines[i-j]:
                        j += 1
                    parameter_names_and_values = [name.split(":")[0] for name in lines[i-j].split(" ")[4:]]
                # every second element is a value
                parameter_names = parameter_names_and_values[::2]
                parameter_values = [float(value.split("[")[1].split("]")[0]) for value in parameter_names_and_values[1::2]]
                # extract max loglike, reference loglike, delta loglikes
                # split the line and search for 'Max loglike' to get the max loglike
                max_likelihood_position = [i for i, s in enumerate(line.split()) if 'Max' in s][0]
                max_loglike = float(line.split()[max_likelihood_position+2][:-1])
                reference_loglike = float(line.split()[max_likelihood_position+5][:-1])

                delta_loglikes = (line.split()[max_likelihood_position+8:])
                delta_loglikes = [float(loglike) for loglike in delta_loglikes]
                not_using_events.append({"type": "not_using_events", "timestamp": timestamp, "parameter_names": parameter_names, "parameter_values": parameter_values, "max_loglike": max_loglike, "reference_loglike": reference_loglike, "delta_loglikes": delta_loglikes})
            elif "Update emulator" in line:
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                updating_events.append({"type": "updating_events", "timestamp": timestamp})
            elif "Groundlevel testing probability set" in line:
                # extract timestamp, probability
                # extract timestamp
                timestamp = line.split(" ")[0] + " " + line.split(" ")[1]
                # extract probability
                probability = float(line.split()[-1])
                groundlevel_events.append({"type": "groundlevel_events", "timestamp": timestamp, "probability": probability})
            elif "Emulator status" in line:
                # extract timestamp, number of data points in cache, number of emulation calls, number of quality check successful calls, number of quality check unsuccessful calls, number of not tested calls
                # extract timestamp
                timestamp = datetime.datetime.strptime(line.split(" ")[0] + " " + line.split(" ")[1], '%Y-%m-%d %H:%M:%S') 
                # extract number of data points in cache, number of emulation calls, number of quality check successful calls, number of quality check unsuccessful calls, number of not tested calls
                data_cache_size = int(lines[i+1].split(":")[-1])
                emulation_calls = int(lines[i+2].split(":")[-1])
                quality_check_successful_calls = int(lines[i+3].split(":")[-1])
                quality_check_unsuccessful_calls = int(lines[i+4].split(":")[-1])
                not_tested_calls = int(lines[i+5].split(":")[-1])

                timings = {}

                # extract time spent in different parts of the code
                j = 7
                while "Timing" in lines[i+j]:
                    event = lines[i+j].split(" ")[4][:-1]
                    total_time = float(lines[i+j].split(":")[4].split(" ")[1][:-1])
                    calls = int(lines[i+j].split(":")[5].split(" ")[1])
                    avg_time = float(lines[i+j].split(":")[6].split(" ")[1][:-1])
                    timings[event] = {"total_time": total_time, "calls": calls, "avg_time": avg_time}
                    j += 1

                status_events.append({"type": "status_events", "timestamp": timestamp, "data_cache_size": data_cache_size, "emulation_calls": emulation_calls, "quality_check_successful_calls": quality_check_successful_calls, "quality_check_unsuccessful_calls": quality_check_unsuccessful_calls, "not_tested_calls": not_tested_calls, "timings": timings})
            else:
                pass
        except:
            pass

    ts_start = datetime.datetime.strptime(lines[0].split(" ")[0] + " " + lines[0].split(" ")[1], '%Y-%m-%d %H:%M:%S') 
    ts_end = datetime.datetime.strptime(lines[-1].split(" ")[0] + " " + lines[-1].split(" ")[1], '%Y-%m-%d %H:%M:%S')

    return adding_events, not_adding_events, training_events, successfully_evaluating_events, unsuccessfully_evaluating_events, running_events, not_using_events, updating_events, groundlevel_events, status_events, ts_start, ts_end

def plot_timings(logfile_path, plot_dir='./', use_timestamps=True):

    adding_events, not_adding_events, training_events, successfully_evaluating_events, unsuccessfully_evaluating_events, running_events, not_using_events, updating_events, groundlevel_events, status_events, ts_start, ts_end = load_logfile(logfile_path)
    # supi dupi, now we have all the events in the lists
    # now we can do some statistics and plots

    # check if the plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #
    # 1. Plot: Show the total time spent in the different parts of the code
    #

    # first we want to plot the timings of the emulator. Here we plot on the xaxis the timestamp and on the yaxis the total time spent in the different parts of the code
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    
    # plot total time spent
    total_seconds = (ts_end - ts_start).total_seconds()
    if use_timestamps:
        ax.plot([ts_start, ts_end], [0, total_seconds], label="Total time", linestyle="--", color="black")
    else:
        ax.plot([0, total_seconds], [0, total_seconds], label="Total time", linestyle="--", color="black")
    
    if False:
        for event in list(status_events[-1]['timings'].keys()):
            if use_timestamps:
                ts = [status['timestamp'] for status in status_events if event in status['timings']]
            else:
                ts = [(status['timestamp']-ts_start).total_seconds() for status in status_events if event in status['timings']] 
            total_times = [status['timings'][event]['total_time'] for status in status_events if event in status['timings']]
            ax.plot(ts, total_times, label=event_dict[event])
    else:
        for event in ['likelihood','theory_code','train','update','emulate']:
            if use_timestamps:
                ts = [status['timestamp'] for status in status_events if event in status['timings']]
            else:
                ts = [(status['timestamp']-ts_start).total_seconds() for status in status_events if event in status['timings']] 
            total_times = [status['timings'][event]['total_time'] for status in status_events if event in status['timings']]
            ax.plot(ts, total_times, label=event_dict[event])
        
        # add emulate_samples and likelihood_testing for testing
        if use_timestamps:
            ts = [status['timestamp'] for status in status_events if event in status['timings']]
        else:
            ts = [(status['timestamp']-ts_start).total_seconds() for status in status_events if event in status['timings']] 
        total_times = [status['timings']['emulate_samples']['total_time']+status['timings']['likelihood_testing']['total_time'] for status in status_events if event in status['timings']]
        ax.plot(ts, total_times, label='testing emulator')
    


    # get unaccounted time at every event
    other = [(status['timestamp']-ts_start).total_seconds()-sum([status['timings'][event]['total_time'] for event in status['timings'] if event!='add_state'])  for status in status_events]

    # plot unaccounted time
    if use_timestamps:
        ax.plot([status['timestamp'] for status in status_events], other, label="other")
    else:
        ax.plot([(status['timestamp']-ts_start).total_seconds() for status in status_events], other, label="other")

    # display 20 uniformly distributed timestamps between ts_start and ts_end
    if use_timestamps:
        all_ts = [ts_start, *ts, ts_end]
        all_ts_in_seconds = [(ts-ts_start).total_seconds() for ts in all_ts]
    else:
        all_ts = [0, *ts, total_seconds]
        all_ts_in_seconds = all_ts
    unif = np.linspace(0, total_seconds, 20)
    # get all indices of the closest timestamps
    indices = [np.argmin(np.abs(np.array(all_ts_in_seconds)-u)) for u in unif]
    #remove double countings
    indices = list(set(indices))
    ax.set_xticks([all_ts[i] for i in indices])
    ax.set_xticklabels([all_ts[i] for i in indices], rotation=90)


    ax.set_ylim(0, total_seconds*1.1)
    if use_timestamps:
        ax.set_xlim(ts_start, ts_end)
    else:
        ax.set_xlim(0, total_seconds)
    ax.legend(ncol=3, loc='upper left')

    if use_timestamps:
        ax.set_xlabel("Timestamp")
    else:
        ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total time [s]")
    ax.set_title("Timings of the emulator")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir + "timings.png")



    #
    # 2. Plot: Show the same as plot1 but as relative time compared to the total time
    #

    # first we want to plot the timings of the emulator. Here we plot on the xaxis the timestamp and on the yaxis the total time spent in the different parts of the code
    fig,ax = plt.subplots(1,1,figsize=(10,5))

    if False:
        for event in list(status_events[-1]['timings'].keys()):
            time_from_beginning = [(status['timestamp']-ts_start).total_seconds() for status in status_events]
            if use_timestamps:
                ts = [status['timestamp'] for status in status_events if event in status['timings']]
            else:
                ts = [(status['timestamp']-ts_start).total_seconds() for status in status_events if event in status['timings']] 
            relative_times = [status['timings'][event]['total_time']/time_from_beginning[i] for i,status in enumerate(status_events) if event in status['timings']]
            ax.plot(ts, relative_times, label=event_dict[event])
    else:
        for event in ['likelihood','theory_code','train','update','emulate']:
            time_from_beginning = [(status['timestamp']-ts_start).total_seconds() for status in status_events]
            if use_timestamps:
                ts = [status['timestamp'] for status in status_events if event in status['timings']]
            else:
                ts = [(status['timestamp']-ts_start).total_seconds() for status in status_events if event in status['timings']] 
            relative_times = [status['timings'][event]['total_time']/time_from_beginning[i] for i,status in enumerate(status_events) if event in status['timings']]
            ax.plot(ts, relative_times, label=event_dict[event])
        
        # add emulate_samples and likelihood_testing for testing
        if use_timestamps:
            ts = [status['timestamp'] for status in status_events if event in status['timings']]
        else:
            ts = [(status['timestamp']-ts_start).total_seconds() for status in status_events if event in status['timings']] 
        relative_times = [(status['timings']['emulate_samples']['total_time']+status['timings']['likelihood_testing']['total_time'])/time_from_beginning[i] for i,status in enumerate(status_events) if event in status['timings']]
        ax.plot(ts, relative_times, label='testing emulator')

    # get unaccounted time at every event
    other = [((status['timestamp']-ts_start).total_seconds()-sum([status['timings'][event]['total_time'] for event in status['timings'] if event!='add_state']))/time_from_beginning[i] for i,status in enumerate(status_events)]

    # plot unaccounted time
    if use_timestamps:
        ax.plot([status['timestamp'] for status in status_events], other, label="other")#, linestyle="--", color="red")
    else:
        ax.plot([(status['timestamp']-ts_start).total_seconds() for status in status_events], other, label="other")

    
    # display 20 uniformly distributed timestamps between ts_start and ts_end
    ax.set_xticks([all_ts[i] for i in indices])
    ax.set_xticklabels([all_ts[i] for i in indices], rotation=90)


    ax.set_ylim(0, 1)
    if use_timestamps:
        ax.set_xlim(ts_start, ts_end)
    else:
        ax.set_xlim(0, total_seconds)
    ax.legend(ncol=3)
    if use_timestamps:
        ax.set_xlabel("Timestamp")
    else:
        ax.set_xlabel("Time [s]")
    ax.set_ylabel("Fraction of cumulated runtime")
    ax.set_title("Timings of the emulator")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir + "timings_relative.png")


def plot_parameter(logfile_paths, parameter, plot_dir='./', min_index=0, max_index=-1):
    # append all events from all logfiles
    for i, logfile_path in enumerate(logfile_paths):

        if i == 0:
            adding_events, not_adding_events, training_events, successfully_evaluating_events, unsuccessfully_evaluating_events, running_events, not_using_events, updating_events, groundlevel_events, status_events, ts_start, ts_end = load_logfile(logfile_path)
        else:
            adding_events_, not_adding_events_, training_events_, successfully_evaluating_events_, unsuccessfully_evaluating_events_, running_events_, not_using_events_, updating_events_, groundlevel_events_, status_events_, ts_start_, ts_end_ = load_logfile(logfile_path)
            adding_events += adding_events_
            not_adding_events += not_adding_events_
            training_events += training_events_
            successfully_evaluating_events += successfully_evaluating_events_
            unsuccessfully_evaluating_events += unsuccessfully_evaluating_events_
            running_events += running_events_
            not_using_events += not_using_events_
            updating_events += updating_events_
            groundlevel_events += groundlevel_events_
            status_events += status_events_
            ts_start = min(ts_start, ts_start_)
            ts_end = max(ts_end, ts_end_)

    # oki, we will have all the events in the lists.
    # We now will do a scatter plot of the parameter values.

    # we are interested in 'adding_events', 'not_adding_events', 'successfully_evaluating_events', 'unsuccessfully_evaluating_events', 'running_events', 'not_using_events'
    # the total number of events
    all_events = adding_events + not_adding_events + successfully_evaluating_events + unsuccessfully_evaluating_events + running_events + not_using_events

    # we want to sort all events by timestamp
    all_events = sorted(all_events, key=lambda x: x['timestamp'])

    all_events = all_events[min_index:max_index]

    N_total = len(all_events)

    # give it indices
    for event in all_events:
        event['index'] = all_events.index(event)

    # now we want to plot the parameter values
    fig,ax = plt.subplots(1,1,figsize=(10,5))

    adding_events = [event for event in all_events if event['type'] == 'adding_event']
    not_adding_events = [event for event in all_events if event['type'] == 'not_adding_events']
    successfully_evaluating_events = [event for event in all_events if event['type'] == 'successfully_evaluating_events']
    unsuccessfully_evaluating_events = [event for event in all_events if event['type'] == 'unsuccessfully_evaluating_events']
    running_events = [event for event in all_events if event['type'] == 'running_events']
    not_using_events = [event for event in all_events if event['type'] == 'not_using_events']


    # plot grey dot with 0.5 alpha for not adding events
    ax.scatter([event['index'] for event in not_adding_events], [event['parameter_values'][event['parameter_names'].index(parameter)] for event in not_adding_events], label="CLASS call NOT added to cache", color="grey", s=10, alpha=0.5)

    # plot adding events # they will have a grey cross
    ax.scatter([event['index'] for event in adding_events], [event['parameter_values'][event['parameter_names'].index(parameter)] for event in adding_events], label="CLASS call added to cache", color="black", s=10, marker="x", zorder=15)

    # plot yellow dot for not using events
    ax.scatter([event['index'] for event in not_using_events], [event['parameter_values'][event['parameter_names'].index(parameter)] for event in not_using_events], label="Run CLASS for outlier", color="yellow", s=10)

    # plot blue dot for running events
    ax.scatter([event['index'] for event in running_events], [event['parameter_values'][event['parameter_names'].index(parameter)] for event in running_events], label="Using emulator without testing", color="blue", s=3)

    # plot green dot for successfully evaluating events
    ax.scatter([event['index'] for event in successfully_evaluating_events], [event['parameter_values'][event['parameter_names'].index(parameter)] for event in successfully_evaluating_events], label="Using emulator with testing", color="green", s=3)

    # plot red dot for unsuccessfully evaluating events
    ax.scatter([event['index'] for event in unsuccessfully_evaluating_events], [event['parameter_values'][event['parameter_names'].index(parameter)] for event in unsuccessfully_evaluating_events], label="Testing of emulator failed", color="red", s=3)
    
    ax.set_xlabel("Step")
    ax.set_ylabel(parameter)
    ax.legend(ncols=2)

    plt.tight_layout()

    plt.savefig(plot_dir + "parameter_" + parameter + ".png")

def plot_1d_cache_video(i, all_events, parameter, delta_loglike, first_training_timestamp, plot_dir, training_events):

    loglikes = [event['loglike'] for event in all_events]
    # remove infinities
    loglikes = [loglike if loglike != float("inf") else max(loglikes) for loglike in loglikes]
    
    fig,ax = plt.subplots(1,1,figsize=(10,5))

    max_loglike = all_events[i]['max_loglike']

    min_allowed_loglike = max_loglike - delta_loglike

    N_total = 0
    N_cache = 0


    for event in all_events[:i]:
        N_total += 1
        # check if we are before or aftr first training
        if event['timestamp'] < first_training_timestamp:
            if event['type'] == 'adding_event':
                if event['loglike'] > min_allowed_loglike:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter)], event['loglike'], color="blue", marker = 'x', s=10)
                    N_cache += 1
                else:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter)], event['loglike'], color="blue", marker = 'x', alpha=0.1, s=10)
            elif event['type'] == 'not_adding_events':
                ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter)], event['loglike'], color="blue", alpha=0.1, s=10)
        else:
            if event['type'] == 'adding_event':
                if event['loglike'] > min_allowed_loglike:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter)], event['loglike'], color="green", marker = 'x', s=10)
                    N_cache += 1
                else:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter)], event['loglike'], color="green", marker = 'x', alpha=0.1, s=10)
            elif event['type'] == 'not_adding_events':
                ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter)], event['loglike'], color="green", alpha=0.1, s=10)

    mean_x = np.mean([event['parameter_values'][event['parameter_names'].index(parameter)] for event in all_events])
    std_x = np.std([event['parameter_values'][event['parameter_names'].index(parameter)] for event in all_events])

    ax.set_xlim(mean_x - 4*std_x, mean_x + 4*std_x)
    ax.set_ylim(max(loglikes)+10 -2*delta_loglike, max(loglikes)+10)

    emulator_trained_flag = False
    for event in training_events:
        if event['timestamp'] < all_events[i]['timestamp']:
            emulator_trained_flag = True



    
    if emulator_trained_flag:
        _ = ax.text(0.05, 0.15, "Emulator IS trained\nTotal number of events: " + str(N_total) + "\nNumber of points in cache: " + str(N_cache), transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
        s0=ax.scatter(0, 0, label="After initial training: ", color="white", alpha=0.1, s=10)
        s1=ax.scatter(0, 0, label="Point in cache", color="green", marker = 'x', s=10)
        s2=ax.scatter(0, 0, label="Point removed from cache", color="green", marker = 'x', alpha=0.1, s=10)
        s3=ax.scatter(0, 0, label="Not adding point", color="green", alpha=0.1, s=10)
        s35=ax.scatter(0, 0, label="Before initial training: ", color="white", alpha=0.1, s=10)
        s4=ax.scatter(0, 0, label="Point in cache", color="blue", marker = 'x', s=10)
        s5=ax.scatter(0, 0, label="Point removed from cache", color="blue", marker = 'x', alpha=0.1, s=10)
        s6=ax.scatter(0, 0, label="Not adding point", color="blue", alpha=0.1, s=10)

        # set legend to lower right
        ax.legend(loc='lower right')
    else:
        s35=ax.scatter(0, 0, label="Before initial training: ", color="white", alpha=0.1, s=10)
        s1=ax.scatter(0, 0, label="Current point in cache", color="blue", marker = 'x', s=10)
        s2=ax.scatter(0, 0, label="Point removed from cache", color="blue", marker = 'x', alpha=0.1, s=10)
        s3=ax.scatter(0, 0, label="Not adding point", color="blue", alpha=0.1, s=10)

        _ = ax.text(0.05, 0.15, "Emulator NOT trained yet\nTotal number of events: " + str(N_total) + "\nNumber of points in cache: " + str(N_cache), transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        # set legend to lower right
        ax.legend(loc='lower right')


    ax.set_xlabel(parameter)
    ax.set_ylabel("Loglike")

    
    plt.tight_layout()
    plt.savefig(plot_dir + "parameter_" + parameter + "_frame_" + str(i) + ".jpg")
    plt.close()

    gc.collect()

def make_1d_cache_video(logfile_paths, parameter, plot_dir='./temp_directory_for_cache_video_1d/', sigma=4, dim =31, min_index=0, max_index=-1):
    from scipy.stats import chi2
    # we need to estimate the quality_threshold_quadratic
    # we use the N_sigma to estimate the quality_threshold_quadratic
    # we estimate the quality_threshold_quadratic such that it becomes dominant over the linear term at the N_sigma point

    # up to which p value do we want to be accurate?
    delta_loglike = -1.53901996e-03 * dim**2 + 3.46998485e-01  * sigma**2 + 5.55189162e-02 * dim * sigma + 6.39086834e-01 * dim + 2.36251372e+00 * sigma + -5.14787690e+00

    # compute delta_loglike
    import cv2
    # append all events from all logfiles
    for i, logfile_path in enumerate(logfile_paths):

        if i == 0:
            adding_events, not_adding_events, training_events, successfully_evaluating_events, unsuccessfully_evaluating_events, running_events, not_using_events, updating_events, groundlevel_events, status_events, ts_start, ts_end = load_logfile(logfile_path)
        else:
            adding_events_, not_adding_events_, training_events_, successfully_evaluating_events_, unsuccessfully_evaluating_events_, running_events_, not_using_events_, updating_events_, groundlevel_events_, status_events_, ts_start_, ts_end_ = load_logfile(logfile_path)
            adding_events += adding_events_
            not_adding_events += not_adding_events_
            training_events += training_events_
            successfully_evaluating_events += successfully_evaluating_events_
            unsuccessfully_evaluating_events += unsuccessfully_evaluating_events_
            running_events += running_events_
            not_using_events += not_using_events_
            updating_events += updating_events_
            groundlevel_events += groundlevel_events_
            status_events += status_events_
            ts_start = min(ts_start, ts_start_)
            ts_end = max(ts_end, ts_end_)

    # oki, we will have all the events in the lists.

    # here we are mainly interested in the 'adding' and 'not_adding' events. 

    # We will addpend all 'adding' and 'not_adding' events to a list and order them by timestamp. Then we will create a video of the added/not added points over time. On the x-axis we will have the parameter value and on the y-axis the loglike.

    # We will have each event as a frame in the video. Additionally we will have a counter in the video showing the total number of events and the number of added points.

    # All plots will be temorially stored at a directory.

    # check if the plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    all_events = adding_events + not_adding_events

    # find the ealiest timestamp of training_events
    first_training_timestamp = min([event['timestamp'] for event in training_events])

    # we want to sort all events by timestamp
    all_events = sorted(all_events, key=lambda x: x['timestamp'])

    # remove events which are not in the range of min_index and max_index
    all_events = all_events[:max_index]

    # give it indices
    for event in all_events:
        event['index'] = all_events.index(event)


    # parallelize following for loop with threads
    try:
        import mpi4py 
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except:
        rank = 0
        size = 1


    for i in range(min_index, len(all_events)):
        if i % size == rank:
            plot_1d_cache_video(i, all_events, parameter, delta_loglike, first_training_timestamp, plot_dir, training_events)
            print(i)
        

    # now we want to create a video from the frames
    if size > 1:
        comm.Barrier()
        if rank == 0:
            print("Rank 0 is done")
    else:
        print("Done")

    if rank == 0:
        # get all frames
        frames = [plot_dir + "parameter_" + parameter + "_frame_" + str(i) + ".jpg" for i in range(min_index, len(all_events))]
        # get the first frame
        frame = cv2.imread(frames[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter(plot_dir + "parameter_" + parameter + "_video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))

        for frame in frames:
            video.write(cv2.imread(frame))

        cv2.destroyAllWindows()
        video.release()

        # remove all frames
        for frame in frames:
            os.remove(frame)

import copy as cp
    
def plot_2d_cache_video(i, all_events, parameter1, parameter2, delta_loglike, first_training_timestamp, plot_dir, training_events):
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    max_loglike = all_events[i]['max_loglike']

    min_allowed_loglike = max_loglike - delta_loglike

    N_total = 0
    N_cache = 0

    import cv2
    for event in all_events[:i]:
        N_total += 1
        # check if we are before or aftr first training
        if event['timestamp'] < first_training_timestamp:
            if event['type'] == 'adding_event':
                if event['loglike'] > min_allowed_loglike:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter1)], event['parameter_values'][event['parameter_names'].index(parameter2)], color="blue", marker = 'x', s=10)
                    N_cache += 1
                else:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter1)], event['parameter_values'][event['parameter_names'].index(parameter2)], color="blue", marker = 'x', alpha=0.1, s=10)
            elif event['type'] == 'not_adding_events':
                ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter1)], event['parameter_values'][event['parameter_names'].index(parameter2)], color="blue", alpha=0.1, s=10)
        else:
            if event['type'] == 'adding_event':
                if event['loglike'] > min_allowed_loglike:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter1)], event['parameter_values'][event['parameter_names'].index(parameter2)], color="green", marker = 'x', s=10)
                    N_cache += 1
                else:
                    ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter1)], event['parameter_values'][event['parameter_names'].index(parameter2)], color="green", marker = 'x', alpha=0.1, s=10)
            elif event['type'] == 'not_adding_events':
                ax.scatter(event['parameter_values'][event['parameter_names'].index(parameter1)], event['parameter_values'][event['parameter_names'].index(parameter2)], color="green", alpha=0.1, s=10)

    mean_x = np.mean([event['parameter_values'][event['parameter_names'].index(parameter1)] for event in all_events])
    std_x = np.std([event['parameter_values'][event['parameter_names'].index(parameter1)] for event in all_events])
    mean_y = np.mean([event['parameter_values'][event['parameter_names'].index(parameter2)] for event in all_events])
    std_y = np.std([event['parameter_values'][event['parameter_names'].index(parameter2)] for event in all_events])
    

    ax.set_xlim(mean_x - 4*std_x, mean_x + 4*std_x)
    ax.set_ylim(mean_y - 4*std_y, mean_y + 4*std_y)

    emulator_trained_flag = False
    for event in training_events:
        if event['timestamp'] < all_events[i]['timestamp']:
            emulator_trained_flag = True



    
    if emulator_trained_flag:
        _ = ax.text(0.05, 0.15, "Emulator IS trained\nTotal number of events: " + str(N_total) + "\nNumber of points in cache: " + str(N_cache), transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
        s0=ax.scatter(0, 0, label="After initial training: ", color="white", alpha=0.1, s=10)
        s1=ax.scatter(0, 0, label="Point in cache", color="green", marker = 'x', s=10)
        s2=ax.scatter(0, 0, label="Point removed from cache", color="green", marker = 'x', alpha=0.1, s=10)
        s3=ax.scatter(0, 0, label="Not adding point", color="green", alpha=0.1, s=10)
        s35=ax.scatter(0, 0, label="Before initial training: ", color="white", alpha=0.1, s=10)
        s4=ax.scatter(0, 0, label="Point in cache", color="blue", marker = 'x', s=10)
        s5=ax.scatter(0, 0, label="Point removed from cache", color="blue", marker = 'x', alpha=0.1, s=10)
        s6=ax.scatter(0, 0, label="Not adding point", color="blue", alpha=0.1, s=10)

        # set legend to lower right
        ax.legend(loc='lower right')
    else:
        s35=ax.scatter(0, 0, label="Before initial training: ", color="white", alpha=0.1, s=10)
        s1=ax.scatter(0, 0, label="Current point in cache", color="blue", marker = 'x', s=10)
        s2=ax.scatter(0, 0, label="Point removed from cache", color="blue", marker = 'x', alpha=0.1, s=10)
        s3=ax.scatter(0, 0, label="Not adding point", color="blue", alpha=0.1, s=10)

        _ = ax.text(0.05, 0.15, "Emulator NOT trained yet\nTotal number of events: " + str(N_total) + "\nNumber of points in cache: " + str(N_cache), transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        # set legend to lower right
        ax.legend(loc='lower right')


    ax.set_xlabel(parameter1)
    ax.set_ylabel(parameter2)

    
    plt.tight_layout()
    plt.savefig(plot_dir + "parameter_" + parameter1 + "_" + parameter2 + "_frame_" + str(i) + ".jpg")
    plt.close()

    gc.collect()


def make_2d_cache_video(logfile_paths, parameter1, parameter2, plot_dir='./temp_directory_for_cache_video_2d/', sigma=4, dim =31, min_index=0, max_index=-1):
    from scipy.stats import chi2
    # we need to estimate the quality_threshold_quadratic
    # we use the N_sigma to estimate the quality_threshold_quadratic
    # we estimate the quality_threshold_quadratic such that it becomes dominant over the linear term at the N_sigma point

    # up to which p value do we want to be accurate?
    delta_loglike = -1.53901996e-03 * dim**2 + 3.46998485e-01  * sigma**2 + 5.55189162e-02 * dim * sigma + 6.39086834e-01 * dim + 2.36251372e+00 * sigma + -5.14787690e+00

    # compute delta_loglike

    # append all events from all logfiles
    for i, logfile_path in enumerate(logfile_paths):

        if i == 0:
            adding_events, not_adding_events, training_events, successfully_evaluating_events, unsuccessfully_evaluating_events, running_events, not_using_events, updating_events, groundlevel_events, status_events, ts_start, ts_end = load_logfile(logfile_path)
        else:
            adding_events_, not_adding_events_, training_events_, successfully_evaluating_events_, unsuccessfully_evaluating_events_, running_events_, not_using_events_, updating_events_, groundlevel_events_, status_events_, ts_start_, ts_end_ = load_logfile(logfile_path)
            adding_events += adding_events_
            not_adding_events += not_adding_events_
            training_events += training_events_
            successfully_evaluating_events += successfully_evaluating_events_
            unsuccessfully_evaluating_events += unsuccessfully_evaluating_events_
            running_events += running_events_
            not_using_events += not_using_events_
            updating_events += updating_events_
            groundlevel_events += groundlevel_events_
            status_events += status_events_
            ts_start = min(ts_start, ts_start_)
            ts_end = max(ts_end, ts_end_)

    # oki, we will have all the events in the lists.

    # here we are mainly interested in the 'adding' and 'not_adding' events. 

    # We will addpend all 'adding' and 'not_adding' events to a list and order them by timestamp. Then we will create a video of the added/not added points over time. On the x-axis we will have the parameter value and on the y-axis the loglike.

    # We will have each event as a frame in the video. Additionally we will have a counter in the video showing the total number of events and the number of added points.

    # All plots will be temorially stored at a directory.

    # check if the plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    all_events = adding_events + not_adding_events

    # we want to sort all events by timestamp
    all_events = sorted(all_events, key=lambda x: x['timestamp'])

    # remove events which are not in the range of min_index and max_index
    all_events = all_events[:max_index]

    # give it indices
    for event in all_events:
        event['index'] = all_events.index(event)

    # find the ealiest timestamp of training_events
    first_training_timestamp = min([event['timestamp'] for event in training_events])

    loglikes = [event['loglike'] for event in all_events]
    # remove infinities
    loglikes = [loglike if loglike != float("inf") else max(loglikes) for loglike in loglikes]

    # parallelize following for loop with threads
    import gc
    import threading


    # parallelize following for loop with threads
    try:
        import mpi4py 
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except:
        rank = 0
        size = 1


    for i in range(min_index, len(all_events)):
        if i % size == rank:
            plot_2d_cache_video(i, all_events, parameter1, parameter2, delta_loglike, first_training_timestamp, plot_dir, training_events)
            print(i)
            
    if size > 1:
        comm.Barrier()
        if rank == 0:
            print("Rank 0 is done")
    else:
        print("Done")

    if rank == 0:
        # now we want to create a video from the frames

        # get all frames
        frames = [plot_dir + "parameter_" + parameter1 + "_" + parameter2 + "_frame_" + str(i) + ".jpg" for i in range(min_index, len(all_events))]
        # get the first frame
        frame = cv2.imread(frames[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter(plot_dir + "parameter_" + parameter1 + "_" + parameter2 + "_video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))

        for frame in frames:
            video.write(cv2.imread(frame))

        cv2.destroyAllWindows()
        video.release()

        # remove all frames
        for frame in frames:
            os.remove(frame)

def plot_errors_in_max_sigma_equals_zero_mode(logfile_paths, plot_dir='./', quality_constant=0.1, quality_linear=0.01, quality_quadratic=0.001, min_index=0, max_index=-1, dim=None, N_sigma=None):

    # set quadratic term if dim and N_sigma are given
    if dim is not None and N_sigma is not None:
        from scipy.stats import chi2
        p_val = chi2.cdf(N_sigma**2, 1)
        accuracy_loglike = chi2.ppf(p_val, dim)/2
        quality_quadratic = (quality_constant + quality_linear*accuracy_loglike)/accuracy_loglike**2
        print("Quality quadratic: ", quality_quadratic)




    # append all events from all logfiles
    for i, logfile_path in enumerate(logfile_paths):

        if i == 0:
            adding_events, not_adding_events, training_events, successfully_evaluating_events, unsuccessfully_evaluating_events, running_events, not_using_events, updating_events, groundlevel_events, status_events, ts_start, ts_end = load_logfile(logfile_path)
        else:
            adding_events_, not_adding_events_, training_events_, successfully_evaluating_events_, unsuccessfully_evaluating_events_, running_events_, not_using_events_, updating_events_, groundlevel_events_, status_events_, ts_start_, ts_end_ = load_logfile(logfile_path)
            adding_events += adding_events_
            not_adding_events += not_adding_events_
            training_events += training_events_
            successfully_evaluating_events += successfully_evaluating_events_
            unsuccessfully_evaluating_events += unsuccessfully_evaluating_events_
            running_events += running_events_
            not_using_events += not_using_events_
            updating_events += updating_events_
            groundlevel_events += groundlevel_events_
            status_events += status_events_
            ts_start = min(ts_start, ts_start_)
            ts_end = max(ts_end, ts_end_)

    # oki, we will have all the events in the lists.
    # We now want to plot the errors in the max_sigma_equals_zero mode

    # here we are only interested in 'not_using_events' and 'not_adding_events'. Every call of 'not_using_events' should have a corresponding 'not_adding_events' call with the same parameter values. 
    # We will plot the difference in loglike between the two calls. 

    # We will have a scatter plot with the parameter values on the x-axis and the difference in loglike on the y-axis.

    # We will have a line at y=0. If the difference in loglike is positive, the emulator is not able to predict the loglike correctly. If the difference in loglike is negative, the emulator is able to predict the loglike correctly.



    # we have to find the corresponding 'not_adding_event' for each 'not_using_event'. We do this by comparing the parameter values of the two events.
    for not_using_event in not_using_events:
        for not_adding_event in not_adding_events:
            if not_using_event['parameter_values'] == not_adding_event['parameter_values']:
                not_using_event['not_adding_event'] = not_adding_event
                break
    
    # remove all events which do not have a corresponding 'not_adding_event'
    not_using_events = [event for event in not_using_events if 'not_adding_event' in event]

    # compute median std
    variances_loglikes = np.array([( event['reference_loglike'] - np.array(event['delta_loglikes']))**2 for event in not_using_events])
    std_loglike = np.sqrt(np.median(variances_loglikes, axis=1))

    delta_loglikes = np.abs(np.array([event['reference_loglike'] - np.array(event['max_loglike']) for event in not_using_events]))

    acc_std = quality_constant + quality_linear * delta_loglikes + quality_quadratic * delta_loglikes**2

    acc_flag = std_loglike < acc_std

    # make plot std_loglike vs acc_std
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.scatter(std_loglike, acc_std)
    ax.plot([0, max(std_loglike)], [0, max(std_loglike)], color='black', linestyle='--')
    ax.set_xlabel("std_loglike")
    ax.set_ylabel("acc_std")
    plt.tight_layout()
    plt.savefig(plot_dir + "std_loglike_vs_acc_std.png")

    # add acc_flag to event
    for i,event in enumerate(not_using_events):
        event['acc_flag'] = acc_flag[i]

    print(len(std_loglike))
    print(len(not_using_events))

    # we want to sort all events by timestamp
    all_events = not_using_events

    N_total = len(all_events)

    delta_loglike = 100

    # now we want to plot the loglike differences
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.scatter([event['reference_loglike'] - event['not_adding_event']['loglike'] for event in all_events if event['acc_flag']], [event['not_adding_event']['loglike'] for event in all_events if event['acc_flag']], label="Accepted points")
    ax.scatter([event['reference_loglike'] - event['not_adding_event']['loglike'] for event in all_events if not event['acc_flag']], [event['not_adding_event']['loglike'] for event in all_events if not event['acc_flag']], label="Unaccepted points")
    ax.axvline(0, color='black', linestyle='--')
    ax.set_ylabel("True loglike")
    ax.set_xlabel("Difference in loglike")

    ax.set_ylim(max([event['not_adding_event']['loglike'] for event in all_events])+10, max([event['not_adding_event']['loglike'] for event in all_events])+10-delta_loglike)

    plt.tight_layout()

    plt.savefig(plot_dir + "errors_in_max_sigma_equals_zero_mode.png")

    inv_acc_flag = [not _ for _ in acc_flag]

    norm_devs = np.array([event['reference_loglike'] - event['not_adding_event']['loglike'] for event in all_events])/std_loglike

    # now do the same plot but scale the x-axis by the std_loglike
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    mean = np.mean(norm_devs)
    std = np.std(norm_devs[abs(norm_devs)<50.0])
    print(max(norm_devs))
    print(std)
    ax.scatter([event['reference_loglike'] - event['not_adding_event']['loglike'] for event in all_events if event['acc_flag']]/std_loglike[acc_flag], [event['not_adding_event']['loglike'] for event in all_events if event['acc_flag']], label="Difference in loglike between not_using_event and not_adding_event")
    ax.scatter([event['reference_loglike'] - event['not_adding_event']['loglike'] for event in all_events if not event['acc_flag']]/std_loglike[inv_acc_flag], [event['not_adding_event']['loglike'] for event in all_events if not event['acc_flag']], label="Difference in loglike between not_using_event and not_adding_event")
    ax.axvline(0, color='black', linestyle='--')
    ax.axvline(mean, color='red', linestyle='--')
    ax.axvline(mean-std, color='red', linestyle='--')
    ax.axvline(mean+std, color='red', linestyle='--')
    ax.set_ylabel("True loglike")
    ax.set_xlabel("Difference in loglike scaled by std_loglike")
    ax.set_ylim(max([event['not_adding_event']['loglike'] for event in all_events])+10, max([event['not_adding_event']['loglike'] for event in all_events])+10-delta_loglike)
    ax.set_xlim(-10,10)
    plt.tight_layout()

    plt.savefig(plot_dir + "errors_in_max_sigma_equals_zero_mode_scaled_by_std_loglike.png")
