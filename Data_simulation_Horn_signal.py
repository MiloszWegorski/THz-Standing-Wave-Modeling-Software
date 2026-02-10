import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

from analysis_tools.dependencies import *

from Signal_source.Measurement_schemes import UniformMeasurement
from Signal_source.Model_signals import HornSignal, NoisyComplexSignal, get_wavenums, create_coeff_list
from Signal_source.Measurement_Systems import MeasurementSystem
from Signal_source.Fitter import HornTransmissionFitter, complex_to_mag_and_phase

def simulateHornFreqSweep(frequency, num_trails, Measurement_scheme, N, M, Transmission, amplitude_noise, phase_noise, tolerance):


    #generate array to store simulation parameters
    complex_simulated_params = np.zeros(N*M, dtype=complex)


    #generate random parameters with values ranging from 0 - 1 in phase and mag
    for i in range(N*M):
        complex_simulated_params[i] = np.power(10, 1) *np.random.normal(0, 1)

    #signal object to generate simulated measurement
    signal = NoisyComplexSignal(HornSignal(frequency, N, M, Transmission, complex_simulated_params), amplitude_noise, phase_noise)

    # signal = HornSignal(frequency_range[0], N, M, Transmission, complex_simulated_params)

    #measurement scheme to retrieve data for given measurement scheme
    measureSystem = MeasurementSystem(Measurement_scheme, signal)

    #fitter to fit simulated data
    fitter = HornTransmissionFitter(frequency,  N, M, Transmission)

    result_array = []

    for j, freq in enumerate(frequency):

        #set frequency for measurement system and fitter to current freq
        measureSystem.set_param_value('wavenum', get_wavenums(freq))
        fitter.set_param_values('frequency', freq)

        #create array to store the fitted parameters
        fit_params = np.zeros(num_trails, dtype=list)

        for i in range(num_trails):
            #generate simulated data
            sim_data = measureSystem.Measure()

            #store fitted parameters
            fit_params[i] = fitter.fit_points(sim_data)

        #add results to a 2d array to store
        result_array.append(fit_params)
    
    #analysis of monte carlo results

    simulated_params_mag = []
    simulated_params_pha = []
    
    for i in result_array:
        temp_mag = []
        temp_pha = []
        for j in i:
            mag , pha = complex_to_mag_and_phase(j)

            temp_pha.append(pha)
            temp_mag.append(mag)

        simulated_params_mag.append(temp_mag)
        simulated_params_pha.append(temp_pha)
    
    simulated_params_mag = np.array(simulated_params_mag)
    simulated_params_pha = np.array(simulated_params_pha)

    mags, phas = complex_to_mag_and_phase(complex_simulated_params)

    fit_quality = 0
    percent_errors_pha = []
    percent_errors_mag = []

    for simulation in simulated_params_mag:

        for j, param_values in enumerate(np.transpose(simulation)):

            percent_errors_mag.append(1-(param_values/mags[j]))

            fit_qual = sum(percent_errors_mag[j] > tolerance)
            if fit_qual > fit_quality:
                fit_quality = fit_qual 
    
    for simulation in simulated_params_pha:

        for j, param_values in enumerate(np.transpose(simulation)):

            percent_errors_pha.append(1-(param_values/phas[j]))

            fit_qual = sum(percent_errors_pha[j] > tolerance)
            if fit_qual > fit_quality:
                fit_quality = fit_qual 
    


    for i in percent_errors_mag:
        plt.figure()
        plt.hist(i, bins=100)
    plt.show()

    for i in percent_errors_pha:
        plt.figure()
        plt.hist(i, bins=100)
    plt.show()

    return fit_quality


class Analyze_Scheme(BaseClass):

    def __init__(self, Num_trails, Phase_noise, Amplitude_noise, tolerance, real_amps):
        
        self.num_trails = Num_trails
        self.phase_noise = Phase_noise
        self.amp_noise = Amplitude_noise

        self.tolerance = tolerance
        self.real_amps = real_amps


    def SimulateHornMeasurement(self, N, M, measurement_scheme, coeff_matrix, freq):

        #create measurement system to allow for repeated simulated measurements
        signal = NoisyComplexSignal(HornSignal(coeff_matrix), self.amp_noise, self.phase_noise)
        system = MeasurementSystem(measurement_scheme, signal)

        # Create fitter for measurement
        fitter = HornTransmissionFitter(coeff_matrix)
        
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

        for i, comp in enumerate(coeff_matrix):
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
        phase_variance = {np.var(std_phase)*100}

        amp, phase = complex_to_mag_and_phase(coeff_matrix[0][2])


        

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

# seed = random.seed(576429)
np.random.seed(2000000)

#------------------------Simulate Measurements--------------------------------#

# N = 2
# M = 2

test_scheme = UniformMeasurement(-10, 10, 20)

freqs = [5]

amps = [20+10j, 10+30j, 10+10J,15+10j]

real_amps, phases = complex_to_mag_and_phase(np.array(amps))

# result = simulateHornFreqSweep(freqs, 1000, test_scheme, 2, 4, True, 0.04, 0.05)

scheme_analyzer = Analyze_Scheme(50000, 0.01, 0.02, 0.05, real_amps)

scheme_analyzer.frequency_sweep(2, 2, True, test_scheme, freqs)



print()