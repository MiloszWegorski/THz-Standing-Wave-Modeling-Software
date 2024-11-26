import numpy as np
from abc import ABC, abstractmethod
from base_class import BaseClass

class noise(ABC, BaseClass):

    def noisify(self, amplitudes):
        return np.array(self._noisify(amplitudes))
    
    @abstractmethod
    def _noisify(self, amplitudes):
        pass


class Gaussian_noise(noise):

    def __init__(self, noise_level : float) -> None:
        
        self.noise_level = noise_level

        self.name = f""

    def _noisify(self, amplitudes : np.array):
        
        noise = np.random.normal(0, self.noise_level, 
                                 len(amplitudes))
        
        noisy_amps = np.empty_like(amplitudes)
        
        for i, j in zip(noise, range(len(amplitudes))):
            noisy_amps[j] = i + amplitudes[j]

        return noisy_amps
    
    def _get_params(self, param_name):
        
        match param_name:
            case 'noise_level': 
                return self.noise_level
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
            
    def _set_params(self, param_name, param_value):
        
        match param_name:
            case 'noise_level':
                self.noise_level = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  
            
    def _get_param_names(self):
        return ['noise_level']
