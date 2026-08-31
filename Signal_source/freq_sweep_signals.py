import numpy as np
from abc import ABC, abstractmethod
from tools.dependencies import BaseClass

from Signal_source.Fitter import complex_to_mag_and_phase
from tools.Save_as_file import Mag_and_phase_to_complex

from tools.dependencies import get_wavenums, create_coeff_list, create_component_list


class Sweep_Signal(BaseClass):

    def get_amplitudes(self, times):
        return np.array(self._get_amplitudes(times))
    
    @abstractmethod
    def _get_amplitudes(self, times):
        pass


class Const_Signal(Sweep_Signal):

    def __init__(self, freqs, distances):
        self.freqs = freqs
        self.distances = distances


    def _get_amplitudes(self, frequencies):
        
        return [frequencies ,np.ones(len(frequencies))]
    
    def _get_name(self):
        return super()._get_name()
    
    def _get_param_names(self):
        return super()._get_param_names()
    
    def _set_param_value(self, param_name, param_value):
        return super()._set_param_value(param_name, param_value)
    
    def _get_param_value(self, param_name):
        return super()._get_param_value(param_name)