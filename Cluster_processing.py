import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

from analysis_tools.dependencies import *

from Signal_source.Measurement_schemes import UniformMeasurement
from Signal_source.Model_signals import HornSignal, NoisyComplexSignal, get_wavenums, create_coeff_list
from Signal_source.Measurement_Systems import HornMeasurementSystem
from Signal_source.Fitter import HornTransmissionFitter

def create_comp_lists(N, M, trans, Measure_scheme, frequencies):

    #start time
    time_taken = time.time()

    #create component list from given inputs
    distances = Measure_scheme.get_points()
    
    if hasattr(frequencies, '__iter__'):
        comp_list = np.empty((len(frequencies),M*N, len(distances)), dtype=complex)    


        for i, freq in enumerate(frequencies):
            comp_list[i] = create_coeff_list(M, N, trans, distances, get_wavenums(freq))
    else:
       comp_list = np.empty((M*N, len(distances)), dtype=complex)

       comp_list = create_coeff_list(M, N, trans, distances, get_wavenums(frequencies)) 

    #take end time and take away start time
    time_taken = time.time() - time_taken

    return time_taken, comp_list

#-----------------------------------------------------------------------------#

def simulate_measurements(num_simulations, Amplitudes, measure_scheme, comp_list, Amp_noise, Phase_noise):

    #start time
    time_taken = time.time()

    #create horn and measure scheme
    signal = NoisyComplexSignal(HornSignal(comp_list), Amp_noise, Phase_noise)
    
    measure_syst = HornMeasurementSystem(measure_scheme, signal)

    #fitter which is used to fit the data
    fitter = HornTransmissionFitter(comp_list)

    #generate data num_simulations times
    fitted_params = np.empty((num_simulations, len(comp_list)), dtype=complex)

    for i in range(num_simulations):
        
        #simulate data + fit parameters
        data = measure_syst.Measure(Amplitudes)

        fitted_params[i] = fitter.fit_points(data)
        
    #take end time and take away start time
    time_taken = time.time() - time_taken

    return fitted_params, num_simulations/time_taken

#-----------------------------------------------------------------------------#

def divide_tasks(num_simulations, freqs, coeffs_lists, Amp_lists, Measure_scheme, Amp_noise, Phase_noise):
    
    #Currently assumes both coefficients and frequencies are meant to be itterated

    results = np.zeros((len(Amp_lists), len(coeffs_lists), num_simulations, len(coeffs_lists[0])), dtype=complex)
    rates = np.zeros((len(Amp_lists), len(coeffs_lists)), dtype=float)

    for i, amps in enumerate(Amp_lists):
        for j, coeff in enumerate(coeffs_lists):
        #these could be split into separate jobs
            results[i, j], rates[i, j] = simulate_measurements(num_simulations, amps, Measure_scheme, coeff, Amp_noise, Phase_noise)

    # create the table to be returned
    tables = []

    for i, amps_set in enumerate(results):
        tables.append([])
        tables[i].append(freqs)
        comps = np.empty((num_simulations,len(coeffs_lists[0])), dtype=complex)
        for z, simulation_results in enumerate(amps_set):
            for components in simulation_results:
                comps[z] = components
        for k in range(len(comps[0])):
            tables[i].append(comps[:, k])
        for k in Amp_lists[i]:
            tables[i].append(k)

    return tables

N = 2
M = 2

amps = [[ 1+0.5j, -0.002-0.02e-1j, 4e-2-0.001e-2j, 6e-10-0.019e-12j], 
        [ 5+0.5j, -0.002-0.02e-1j, 4e-2-0.001e-2j, 6e-10-0.019e-12j], 
        [ 2+0.5j, -0.002-0.02e-1j, 4e-2-0.001e-2j, 6e-10-0.019e-12j]]

freqs = np.linspace(10, 20, 10)

distances = UniformMeasurement(-80, 80, 200)

t, comps = create_comp_lists(N, M, False, distances, freqs)

fitted_params = divide_tasks(100, freqs, comps, amps, distances, 0.01, 0.02)

for i in fitted_params:
    
    plt.hist(np.real(i[:, :, 0]))
    plt.show()