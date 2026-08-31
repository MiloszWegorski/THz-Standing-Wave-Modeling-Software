import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

from tools.dependencies import *

from Signal_source.Measurement_schemes import UniformMeasurement
from Signal_source.Model_signals import HornSignal, NoisyComplexSignal
from Signal_source.Measurement_Systems import FreqSignalMeasurementSystem
from Signal_source.Fitter import HornTransmissionFitter, complex_to_mag_and_phase

def simulateHornFreqSweep(freqs, signal, num_trails, Measurement_scheme, tolerance, components):

    #measurement scheme to retrieve data for given measurement scheme
    measureSystem = FreqSignalMeasurementSystem(scheme=Measurement_scheme, signal=signal)

    #fitter to fit simulated data
    fitter = HornTransmissionFitter(components=components)


    result_array = np.empty((len(freqs), num_trails, len(components)), dtype=complex)

    for j, freq in enumerate(freqs):

        #create array to store the fitted parameters
        fit_params = np.zeros((num_trails, len(components)), dtype=list)

        for i in range(num_trails):
            #generate simulated data
            sim_data = measureSystem.Measure(freq=freq)

            #store fitted parameters
            fit_params[i] = fitter.fit_points(Amplitudes=sim_data[1], freq=freq,scheme=Measurement_scheme)

        #add results to a 2d array to store
        result_array[j] = fit_params

    stdevs = np.empty(len(freqs), dtype=float)

    result_mags, result_pha = complex_to_mag_and_phase(result_array)

    for i, results in enumerate(result_array):

        result_mag, results_pha = complex_to_mag_and_phase(results[:, 0])

        stdevs[i] = np.std(result_mag)

    return stdevs, result_mags, result_pha


class Analyze_Scheme(BaseClass):

    def __init__(self, Num_trails, tolerance):
        
        self.num_trails = Num_trails
        self.tolerance = tolerance


    def SimulateHornMeasurement(self, signal, measurement_scheme, component_matrix, freq):

        #create measurement system to allow for repeated simulated measurements
        system = MeasurementSystem(measurement_scheme, signal)

        # Create fitter for measurement
        fitter = HornTransmissionFitter(component_matrix)

        N, M = get_M_and_N(component_matrix)
        
        result_params = np.empty((self.num_trails, N*M), dtype=complex)

        simulations_per_second = time.time()

        for i in range(self.num_trails):
            data = system.Measure()

            result_params[i] = fitter.fit_points(data)

        simulations_per_second = self.num_trails/(time.time() - simulations_per_second)

        print(f'Number of simulations per second preformed = {simulations_per_second}')


        result_amps, result_phase = complex_to_mag_and_phase(result_params)

        std_mags = np.array([np.abs(np.std(i)/np.average(i)) for i in np.rot90(result_amps)])
        std_phase = np.array([np.abs(np.std(i)/np.average(i)) for i in np.rot90(result_phase)])

        amp_fail_rate = np.empty(len(result_amps[0]), dtype=float)
        phase_fail_rate = np.empty(len(result_phase[0]), dtype=float)
        success_rate = np.empty(len(result_amps[0]), dtype=float)

        mins = np.empty(len(result_amps[0]), dtype=float)
        maxs = np.empty(len(result_amps[0]), dtype=float)

        for i, comp in enumerate(component_matrix):
            amplitude, phase = complex_to_mag_and_phase(comp[2])

            amp_upper_limit = amplitude * (1 + self.tolerance)
            amp_lower_limit = amplitude * (1 - self.tolerance)

            phase_upper_limit = phase * (1 + self.tolerance)
            phase_lower_limit = phase * (1 - self.tolerance)

            amp_fail_rate[i] = ((np.logical_or(result_amps[:, i] >= amp_upper_limit,
                                            result_amps[:, i] <= amp_lower_limit)).sum()/self.num_trails)*100
            
            phase_fail_rate[i] = ((np.logical_or(result_phase[:, i] >= phase_upper_limit, 
                                                result_phase[:, i] <= phase_lower_limit)).sum()/self.num_trails)*100
            
            success_rate[i] = (((np.logical_and(result_amps[:, i] <= amp_upper_limit, 
                                                result_amps[:, i] >= amp_lower_limit)).sum() +
                            (np.logical_and(result_phase[:, i] <= phase_upper_limit, 
                                                result_phase[:, i] >= phase_lower_limit)).sum())/(2*self.num_trails))*100

            mins[i] = np.min((result_amps[:, i]/ amplitude)- 1)
            maxs[i] = np.max((result_amps[:, i]/ amplitude)- 1)


        mag_variance = {np.var(std_mags)*100}
        print(f'magnitude variance = {mag_variance}')
        phase_variance = {np.var(std_phase)*100}
        print(f'phase variance = {phase_variance}')

        amp, phase = complex_to_mag_and_phase(component_matrix[0][2])


        

        stats_matrix = [[[success_rate, freq], [amp_fail_rate, freq], [phase_fail_rate, freq]],
                        [[std_mags, freq], [std_phase, freq]],
                        #insert number of peaks measurement here
                        [[mins, freq], [maxs, freq]]]
        

        return stats_matrix

    def frequency_sweep(self, M, N, transmission, measurement_scheme, freqs):

        results = []

        for i in freqs:
            coeffs = create_coeff_list(M, N, transmission, measurement_scheme.get_points(), get_wavenums(i))

            result = self.SimulateHornMeasurement(N, M, measurement_scheme, coeffs, i)

            if any(result[0][0][0] < 100):
                results.append(result[0])
            
        print(results)

    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'num_trails':
                self.num_trails = param_value
            case 'reflections':
                self.reflections = param_value
            case 'components':
                self.components = param_value
            case 'phase_noise':
                self.phase_noise = param_value
            case 'amp_noise':
                self.amp_noise = param_value
            case 'components_matrix':
                self.components_matrix = param_value
            case 'tolerance':
                self.tolerance = param_value
            case _:
                raise ValueError(f'Unknown parameter name : {param_name}')

    def _get_param_value(self, param_name):
        match param_name:
            case 'num_trails':
                return self.num_trails
            case 'reflections':
                return self.reflections
            case 'components':
                return self.components
            case 'phase_noise':
                return self.phase_noise
            case 'amp_noise':
                return self.amp_noise
            case 'components_matrix':
                return self.components_matrix
            case 'tolerance':
                return self.tolerance
            case _:
                raise ValueError(f'Unknown parameter name : {param_name}')

    def _get_param_names(self):
        return ['num_trails', 'reflections', 'components', 'phase_noise', 'amp_noise', 'components_matrix', 'tolerance']
        
    def _get_name(self):
        return f'This class is used for testing measurement schemes and determining \
            whether the scheme tested is viable for a given measurement'
