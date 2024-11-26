import numpy as np
from abc import ABC, abstractmethod
from base_class import BaseClass

class Signal(ABC, BaseClass):

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
                raise ValueError('Unknown parameter name') 

class NoisySignal(Signal):
    
    def __init__(self, signal : Signal, noise_level =0):
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