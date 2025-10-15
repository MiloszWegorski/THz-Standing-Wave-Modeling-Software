import numpy as np
from abc import ABC, abstractmethod
from analysis_tools.dependencies import BaseClass

from Signal_source.Measurement_schemes import DummyMeasurementScheme

from Signal_source.Fitter import complex_to_mag_and_phase
from analysis_tools.Save_as_file import Mag_and_phase_to_complex

from analysis_tools.dependencies import get_wavenums, create_coeff_list, create_component_list

class Signal(BaseClass):

    def get_amplitudes(self, times):
        return np.array(self._get_amplitudes(times))
    
    @abstractmethod
    def _get_amplitudes(self, times):
        pass

class FrequencyDistanceBasedSignal(BaseClass):

    def get_dimension(self):
        return self._get_dimension()
    
    @abstractmethod
    def _get_dimension(self):
        pass 

    def get_spectrum(self, frequencies):

        scheme = DummyMeasurementScheme(self.get_dimension())

        # specifically gor 1D signal since distance has 1 dimension of 0
        return np.asarray([self.get_amplitudes(freq, scheme, noiseless=True)
                            for freq in frequencies])

    def get_amplitudes(self, frequency, distances,*, noiseless=False):
        return np.array(self._get_amplitudes(frequency, distances, noiseless=noiseless))
    
    @abstractmethod
    def _get_amplitudes(self, frequency, distances, *, noiseless):
        pass

class FrequencyDistanceBasedSignalAdapter(FrequencyDistanceBasedSignal):

    def __init__(self, signal : Signal):
        self.signal = signal

    def _get_dimension(self):
        return self.signal.get_dimension()
    
class ModelSineSignal(Signal):
    
    def __init__(self, freq, amp, phase ,offset):

        self.frequency = freq
        self.amplitude = amp
        self.phase = phase
        self.offset = offset

        self.name = f"Returns a perfect sinousidal wave"

    def _get_amplitudes(self, times):

        return self.amplitude * np.sin(2 * times * np.pi * self.frequency + 
                                       self.phase) + self.offset


    def _get_name(self):
        return f"Returns a perfect sinousidal wave"
    
    def _get_param_names(self):
        return ['frequency', 'amplitude', 'phase', 'offset']
    
    def _get_param_value(self, param_name):

        match param_name:
            case 'frequency':
                return self.frequency
            case 'amplitude':
                return self.amplitude
            case 'phase':
                return self.phase
            case 'offset':
                return self.offset
            case _:
                raise ValueError(f'Unknown parameter name : {param_name}') 
            
    def _set_param_value(self, param_name, param_value):

        match param_name:
            case 'frequency':
                self.frequency = param_value
            case 'amplitude':
                self.amplitude = param_value
            case 'phase':
                self.phase =param_value
            case 'offset':
                self.offset = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  

class HornSignal(FrequencyDistanceBasedSignal):
    def __init__(self, component_matrix, component_amplitudes):
        self.component_list = component_matrix
        self.component_amplitudes = component_amplitudes

    def _get_amplitudes(self, frequency, scheme, noiseless):

        coeff_list = create_coeff_list(self.component_list, scheme, get_wavenums(frequency))

        A_1 = np.zeros(len(scheme.get_points()), dtype= complex)

        for amp, comps in zip(self.component_amplitudes, coeff_list):

            A_1 += amp * comps

        return A_1
    
    def _get_dimension(self):
        return super()._get_dimension()

    def _get_param_value(self, param_name):
        match param_name:
            case 'coefficient_matrix':
                return self.components
            case 'wavenum':
                return self.wavenum
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  

    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'components':
                self.components = param_value
            case 'number_reflections':
                self.num_reflections = param_value
            case 'wavenum':
                self.wavenum = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  

    def _get_param_names(self):
        return ['components', 'number_reflections', 'wavenum']
    
    def _get_name(self):
        return f'Function of the fransmitted signal with distance at frequency\
        {self.freq}'    
class NoisySignal(Signal):
    
    def __init__(self, signal : Signal, noise_level =0, phase_noise = None):
        self.noise_level = noise_level
        self.signal = signal

    def _get_amplitudes(self, times):
        
        initial_amps = self.signal.get_amplitudes(times)

        return initial_amps + np.random.normal(0.0, self.noise_level, size=initial_amps.shape)
    
    def _get_name(self):
        return f"Returns a noisified signal with gaussian noise level = {self.noise_level}"
    
    def _get_param_names(self):
        parameters = self.signal.get_param_names()
        parameters.append('noise level')
        
        return parameters
    
    def _get_param_value(self, param_name):
        
        match param_name:
            case 'noise level':
                return self.noise_level
            case _:
                return self.signal.get_param_value(param_name)
            
    def _set_param_value(self, param_name, param_value):
        
        match param_name:
            case 'noise level':
                self.noise_level = param_value
            case _:
                return self.signal.set_param_value(param_name, param_value)
            
class NoisyComplexSignal(FrequencyDistanceBasedSignalAdapter):
    
    def __init__(self, signal : Signal, magnitude_noise_level =0, phase_noise_level = 0):
        
        super().__init__(signal)
        self.magnitude_noise_level = magnitude_noise_level
        self.phase_noise = phase_noise_level

    def _get_amplitudes(self, freq, times, *, noiseless):
       
        initial_complex = self.signal.get_amplitudes(freq, times, noiseless=noiseless)

        if noiseless:
            return initial_complex

        final_real = np.real(initial_complex) + np.random.normal(0.0, self.magnitude_noise_level, size=initial_complex.shape)

        final_imag = np.imag(initial_complex) + np.random.normal(0.0, self.magnitude_noise_level, size=initial_complex.shape)

        final_complex = np.array([complex(real, imag) for (real, imag) in zip(final_real, final_imag)]) 
        
        final_complex = final_complex * np.exp(1j * np.random.normal(0, self.phase_noise, size=initial_complex.shape))

        return final_complex
    
    def _get_name(self):
        return f"Returns a noisified signal with gaussian noise level = {self.noise_level}"
    
    def _get_param_names(self):
        parameters = self.signal.get_param_names()
        parameters.append('noise level')
        
        return parameters
    
    def _get_param_value(self, param_name):
        
        match param_name:
            case 'noise level':
                return self.noise_level
            case _:
                return self.signal.get_param_value(param_name)
            
    def _set_param_value(self, param_name, param_value):
        
        match param_name:
            case 'noise level':
                self.noise_level = param_value
            case _:
                return self.signal.set_param_value(param_name, param_value)
            
class SimpleTransmittedSignal(FrequencyDistanceBasedSignal):

    def __init__(self,*, Amplitude, Attenuation, gamma, source_distance):
        self.amp = Amplitude
        self.attenuation = Attenuation
        self.separation = source_distance
        self.gamma = gamma

    def _get_amplitudes(self, freq, scheme, *, noiseless):

        overall_propagation_distance = self.separation + scheme.get_points()

        wavenum = get_wavenums(freq)

        return self.amp * (
                np.exp(-overall_propagation_distance * self.attenuation)* 
                np.exp(-1j * wavenum * overall_propagation_distance) +
                self.gamma * np.exp(-3 * overall_propagation_distance * self.attenuation)* 
                np.exp(-3j * wavenum * overall_propagation_distance))
    
    def _get_dimension(self):
        return 1

    def _get_param_names(self):
        return ['amplitude', 'attenuation', 'wave_number', 'separation_distance']

    def _get_param_value(self, param_name):
        match param_name:
            case 'amplitude':
                return self.amp
            case 'attenuation':
                return self.attenuation
            case 'wave_number':
                return self.wavenum
            case 'separation_distance':
                return self.separation
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  

    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'amplitude':
                self.amp = param_value
            case 'attenuation':
                self.attenuation = param_value
            case 'wave_number':
                self.wavenum = param_value
            case 'separation_distance':
                self.separation = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  

    def _get_name(self):
        return "Signal object which generates a simple transmission signal"
    

class Gaussian_spectrum_signal(FrequencyDistanceBasedSignalAdapter):

    def __init__(self,*, signal : Signal, gamma : float, f_0 : float, Amplitude : float, orientation : int):

        super().__init__(signal)
        self.gamma = gamma # width of peak
        self.amplitude = Amplitude #amplitude of peak
        self.f_0 = f_0 #resonant freq
        self.orientation = orientation # changes between peak/troff if it is 1/-1

    
    def _get_amplitudes(self, freq, scheme, noiseless):
        
        sig = self.signal.get_amplitudes(freq, scheme, noiseless=noiseless)

        spectrum_modulation = (self.amplitude*(1/(4*(self.f_0/self.gamma)**2 *
                                       (1- (freq/self.f_0))**2 + 1)))*np.exp(1j * (np.arctan(self.gamma * freq/
                                    (self.f_0 - freq)))-np.pi/2)

        return sig*spectrum_modulation


    def _get_name(self):
        return f'Returns a lorenzian shape spectrum with the peak amplitude \
{self.amplitude} and the FWHM of {self.gamma} where the peak is at f_0 = {self.f_0}'
    
    def _get_param_names(self):
        return ['gamma', 'f_0', 'amplitude']
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'gamma':
                return self.gamma
            case 'f_0':
                return self.f_0
            case 'amplitude':
                return self.amplitude
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'gamma':
                self.gamma = self.gamma
            case 'f_0':
                self.f_0 = self.f_0
            case 'amplitude':
                self.amplitude = self.amplitude
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')