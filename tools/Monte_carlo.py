from Signal_source.Measurement_Systems import MeasurementSystem,FreqSignalMeasurementSystem
from Signal_source. Fitter import HornTransmissionFitter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from tools.dependencies import *

from Signal_source.Model_signals import HornSignal, NoisyComplexSignal, get_wavenums
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

class MonteCarloAnalyse():

    def __init__(self):
        self.freq_results = []
        self.amp_results = []
        self.offset_results = []
        self.phase_results = []


    def simulate(self, num_trails, measure_device, fitter):

        for i in range(num_trails):

            data = measure_device.Measure()
            
            popt = fitter.fit_points(data)

            
            self.freq_results.append(popt[0])
            self.amp_results.append(popt[1])
            self.offset_results.append(popt[2])
            self.phase_results.append(popt[3])

        
        return self
    
def sweep_variable(monte_carlo, num_trails:int ,scheme , model_signal,
                fitter, value_range:tuple, sweep_variable):


    sweep_range = np.linspace(value_range[0], value_range[1], value_range[2])

    result_values = {
        'freq_res' : [],
        'Amp_res' : [],
        'phase_res' : [],
        'offset' : []
    }

    measure_device = MeasurementSystem(scheme, model_signal)
    
    for i in tqdm(sweep_range):

        measure_device.set_param_value(sweep_variable, i)
        measure_device.set_param_value('num_points', int(i* 150))

        measure_device = MeasurementSystem(scheme, model_signal)
        
        results = monte_carlo.simulate(num_trails, measure_device, fitter)
    
        result_values['freq_res'].append(np.std(results.freq_results))
        result_values['Amp_res'].append(np.std(results.amp_results))
        result_values['offset'].append(np.std(results.offset_results))
        result_values['phase_res'].append(np.std(results.phase_results))

    fig2, ax2 = plt.subplots(2, 2)

    for i in ax2:
        for j in i:
            j.set_xlabel(sweep_variable.replace('_', ' '))
            j.set_ylabel('Standard Deviation')

    
    ax2[0][0].set_title('Frequency')
    ax2[0][0].plot(sweep_range, result_values['freq_res'])
    ax2[0][1].set_title('Amplitude')
    ax2[0][1].plot(sweep_range, result_values['Amp_res'])
    ax2[1][0].set_title('Offset')
    ax2[1][0].plot(sweep_range, result_values['offset'])
    ax2[1][1].set_title('Phase')
    ax2[1][1].plot(sweep_range, result_values['phase_res'])

    plt.show()

def compare_model_monte_carlo_analysis(*, num_trails : int, 
                         measurement_system: FreqSignalMeasurementSystem,
                         measurement_schemes, freq, models):

    results_full = []

    for i, (model, scheme) in tqdm(enumerate(zip(models, measurement_schemes))):

        results = np.empty((num_trails, len(model)), dtype=complex)
        fitter = HornTransmissionFitter(components=model)

        for j in tqdm(range(num_trails)):
            measurement = measurement_system.Measure(freq=freq)

            fit = fitter.fit_points(Amplitudes=measurement[1], freq=freq,
                                    scheme=scheme)

            results[j] = fit


        results_full.append(results)

    model_1_amps, model_1_pha = complex_to_mag_and_phase(results_full[0])
    model_2_amps, model_2_pha = complex_to_mag_and_phase(results_full[1])

    stdev_model_1_amps = np.std(model_1_amps[:, 0])
    average_model_1_amps = np.average(model_1_amps[:, 0])

    stdev_model_1_pha = np.std(model_1_pha[:, 0])
    average_model_1_pha = np.average(model_1_pha[:, 0])

    stdev_model_2_amps = np.std(model_2_amps[:, 0])
    average_model_2_amps = np.average(model_2_amps[:, 0])

    stdev_model_2_pha = np.std(model_2_pha[:, 0])
    average_model_2_pha = np.average(model_2_pha[:, 0])

    fig2, ax2 = plt.subplots(2, 2, figsize=(5.12*3, 2.88*3))

    #--------------------model 1 amp------------------------------------------#
    ax2[0][0].set_title('Model 1 Amplitude', fontsize=17)
    ax2[0][0].hist(model_1_amps[:, 0], bins=50)
    ax2[0][0].set_xlabel('Magnitude (db)',fontsize=15)

    label='Model 1 standard deviation'
    for x in [average_model_1_amps+stdev_model_1_amps,
                     average_model_1_amps-stdev_model_1_amps]:
        ax2[0][0].axvline(x,
                      color='red', linestyle='--', label=label)
        label = None

    label='Model 2 standard deviation'
    for x in [average_model_2_amps+stdev_model_2_amps,
                     average_model_2_amps-stdev_model_2_amps]:
        ax2[0][0].axvline(x,
                     color='green', linestyle='--', label=label)
        label = None



    #--------------------model 2 amp------------------------------------------#
    ax2[1][0].set_title('Model 2 Amplitude', fontsize=17)
    ax2[1][0].hist(model_2_amps[:, 0], bins=50)
    ax2[1][0].set_xlabel('Magnitude (db)',fontsize=15)

    label='Model 1 standard deviation'
    for x in [average_model_1_amps+stdev_model_1_amps,
                     average_model_1_amps-stdev_model_1_amps]:
        ax2[1][0].axvline(x,
                      color='red', linestyle='--', label=label)
        label = None

    label='Model 2 standard deviation'
    for x in [average_model_2_amps+stdev_model_2_amps,
                     average_model_2_amps-stdev_model_2_amps]:
        ax2[1][0].axvline(x,
                     color='green', linestyle='--', label=label)
        label = None


    #--------------------model 1 pha------------------------------------------#
    ax2[0][1].set_title('Model 1 Phase', fontsize=17)
    ax2[0][1].hist(model_1_pha[:, 0], bins=50)
    ax2[0][1].set_xlabel('Phase (deg)',fontsize=15)

    label='Model 1 standard deviation'
    for x in [average_model_1_pha+stdev_model_1_pha,
                     average_model_1_pha-stdev_model_1_pha]:
        ax2[0][1].axvline(x,
                      color='red', linestyle='--', label=label)
        label = None

    label = 'Model 2 standard deviation'
    for x in [average_model_2_pha+stdev_model_2_pha,
                     average_model_2_pha-stdev_model_2_pha]:
        ax2[0][1].axvline(x,
                     color='green', linestyle='--', label=label)
        label = None


    #--------------------model 2 pha------------------------------------------#
    ax2[1][1].set_title('Model 2 Phase', fontsize=17)
    ax2[1][1].hist(model_2_pha[:, 0], bins=50)
    ax2[1][1].set_xlabel('Phase (deg)',fontsize=15)

    label='Model 1 standard deviation'
    for x in [average_model_1_pha+stdev_model_1_pha,
                     average_model_1_pha-stdev_model_1_pha]:
        ax2[1][1].axvline(x,
                      color='red', linestyle='--', label=label)
        label=None

    label='Model 2 standard deviation'
    for x in [average_model_2_pha+stdev_model_2_pha,
                     average_model_2_pha-stdev_model_2_pha]:
        ax2[1][1].axvline(x,
                     color='green', linestyle='--',label=label)
        label=None

    for i in ax2:
        for j in i:
            j.legend()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


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
            coeffs = 0#create_coeff_list(M, N, transmission, measurement_scheme.get_points(), get_wavenums(i))

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
