import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from analysis_tools.dependencies import *

from Signal_source.Measurement_schemes import UniformMeasurement
from Signal_source.Model_signals import HornSignal, NoisyComplexSignal, SimpleTransmittedSignal, get_wavenums, create_coeff_list
from Signal_source.Measurement_Systems import MeasurementSystem, FreqSignalMeasurementSystem
from Signal_source.Fitter import HornTransmissionFitter
from analysis_tools.Save_as_file import Save_simulation

from concurrent.futures import ProcessPoolExecutor, as_completed

def create_comp_lists(comp_list, trans, Measure_scheme, frequencies):

    #start time
    time_taken = time.time()

    
    if hasattr(frequencies, '__iter__'):
        coeff_list = np.empty((len(frequencies),len(comp_list), len(Measure_scheme)), dtype=complex)    


        for i, freq in enumerate(frequencies):
            coeff_list[i] = create_coeff_list(comp_list, trans, Measure_scheme, get_wavenums(freq))
    else:
       coeff_list = np.empty((len(comp_list), len(Measure_scheme)), dtype=complex)

       coeff_list = create_coeff_list(comp_list, trans, Measure_scheme, get_wavenums(frequencies)) 

    #take end time and take away start time
    time_taken = time.time() - time_taken

    return time_taken, coeff_list

#-----------------------------------------------------------------------------#

def simulate_given_signal(num_simulations, freq, Signal, measure_scheme, comp_list, Amp_noise, Phase_noise):
        #start time
    time_taken = time.time()

    #create horn and measure scheme
    #move noise into passed in signal object
    noisy_signal = NoisyComplexSignal(Signal,
                                Amp_noise, Phase_noise)
    
    measure_syst = FreqSignalMeasurementSystem(measure_scheme, noisy_signal)

    coeffs = create_coeff_list(comp_list, 1, measure_scheme, get_wavenums(freq))

    #fitter which is used to fit the data
    fitter = HornTransmissionFitter(coeffs)

    #generate data num_simulations times
    fitted_params = np.empty((num_simulations, len(comp_list)), dtype=complex)

    for i in range(num_simulations):
        
        #simulate data + fit parameters
        data = measure_syst.Measure(freq)

        fitted_params[i] = fitter.fit_points(data)
        
    #take end time and take away start time
    time_taken = time.time() - time_taken

    return fitted_params, num_simulations/time_taken

def multithread_tasks_given_signal(*, foldername, filename, num_simulations, freqs, 
                                 coeff_list, Measure_scheme, Signal, Amp_noise, 
                                 Phase_noise):
    

    time_start = time.time()

    results = np.empty((len(freqs), num_simulations, len(coeff_list)), dtype=complex)
    result_times = np.empty(len(freqs), dtype=float)

    with ProcessPoolExecutor() as executor:
    
        futures = {
            executor.submit(
                simulate_given_signal,
                num_simulations,
                freq,
                Signal,
                Measure_scheme,
                coeff_list,
                Amp_noise,
                Phase_noise
            ): i
            for i, freq in enumerate(freqs)
        }

        for future in as_completed(futures):
            i = futures[future]
            res, tim = future.result()

            results[i] = res
            result_times[i] = tim
    
    time_tot = time.time() - time_start

    result_table = np.empty((len(freqs) * num_simulations, 2*len(results[0][0])+1), dtype=float)

    column_names = []

    column_names.append('f[Ghz]')

    counter = 0

    for i , (f, amps) in enumerate(zip(freqs, results)):

        amps_split = np.empty((2*len(amps[0])+1), dtype=float)

        amps_split[0] = f

        for j, amp in enumerate(amps):
            for z, a in enumerate(amp):
                if counter == 0:
                    column_names.append('A'+str(z)+'Real')
                    column_names.append('A'+str(z)+'Angle')

                amps_split[2*z+1] = np.real(a)
                amps_split[2*z+2] = np.angle(a)
            
            result_table[counter] = amps_split
            counter += 1
            

    saver = Save_simulation(foldername,filename)
    saver.save_data(result_table, column_names, [num_simulations, 
                                                 coeff_list,
                                                 Measure_scheme,
                                                 amps,
                                                 Amp_noise, 
                                                 Phase_noise])

    return result_table, time_tot