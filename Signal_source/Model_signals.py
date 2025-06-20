import numpy as np
from abc import ABC, abstractmethod
from analysis_tools.dependencies import BaseClass

from Signal_source.Fitter import complex_to_mag_and_phase
from analysis_tools.Save_as_file import Mag_and_phase_to_complex

from analysis_tools.dependencies import get_wavenums, create_coeff_list, create_component_list

class Signal(BaseClass):

    def get_amplitudes(self, times):
        return np.array(self._get_amplitudes(times))
    
    @abstractmethod
    def _get_amplitudes(self, times):
        pass

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

class HornSignal(Signal):
    def __init__(self, coeff_matrix):
        
        self.component_list = coeff_matrix

    def _get_amplitudes(self, amps):

        A_1 = np.zeros(len(self.component_list[0]), dtype= complex)

        for amp, comps in zip(amps, self.component_list):

            A_1 += amp * comps

        return A_1

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
            
class NoisyComplexSignal(Signal):
    
    def __init__(self, signal : Signal, magnitude_noise_level =0, phase_noise_level = 0):
        self.magnitude_noise_level = magnitude_noise_level
        self.phase_noise = phase_noise_level
        self.signal = signal

    def _get_amplitudes(self, times):
       
        initial_complex = self.signal.get_amplitudes(times)

        initial_amps, initial_phases = complex_to_mag_and_phase(initial_complex)

        final_amps = initial_amps + np.random.normal(0.0, self.magnitude_noise_level, size=initial_amps.shape)

        #implement absolute noise
        final_phases = initial_phases + np.random.normal(0.0, self.phase_noise, size=initial_phases.shape)

        initial_complex = Mag_and_phase_to_complex(final_amps, final_phases)

        return initial_complex
    
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
            


class SimpleTransmittedSignal(Signal):

    def __init__(self, Amplitude, Attenuation, frequency, source_distance):

        self.amp = Amplitude
        self.attenuation = Attenuation
        self.wavenum = get_wavenums(frequency)
        self.separation = source_distance

    def _get_amplitudes(self, distances):

        overall_propagation_distance = self.separation + distances
        return [distances, self.amp *np.exp(-overall_propagation_distance * self.attenuation) * np.exp(-1j * self.wavenum * overall_propagation_distance)]
    

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
    