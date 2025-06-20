import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from analysis_tools.dependencies import BaseClass


class MeasurementScheme(BaseClass):
    
    def get_points(self):
        return self._get_points()
    
    @abstractmethod
    def _get_points(self):
        pass


class UniformMeasurement(MeasurementScheme):

    def __init__(self, start_time, end_time, num_points):
        self.start_time = start_time
        self.end_time = end_time
        self.num_points = num_points

        self.name = f"Uniform distribution from {start_time} to {end_time}\
 with {num_points} points"

    def _get_points(self):
        return np.array(np.linspace(self.start_time, self.end_time, self.num_points))
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'start_time':
                return self.start_time
            case 'end_time':
                return self.end_time
            case 'num_points':
                return self.num_points
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'start_time':
                self.start_time = param_value
            case 'end_time':
                self.end_time = param_value
            case 'num_points':
                self.num_points = param_value
            case _:
                raise ValueError('Unknown parameter name') 
            
    def _get_param_names(self):
        return ['start_time', 'end_time', 'num_points']

class LogMeasurement(MeasurementScheme):

    def __init__(self, start_time, end_time, num_points):
        self.start_time = start_time
        self.end_time = end_time
        self.num_points = num_points

        self.name = f"Logarithmic distribution from {start_time} to {end_time}\
 with {num_points} points"

    def _get_points(self):
        return np.logspace(self.start, self.end, self.num_points)
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'start_time':
                return self.start_time
            case 'end_time':
                return self.end_time
            case 'num_points':
                return self.num_points
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'start_time':
                self.start_time = param_value
            case 'end_time':
                self.end_time = param_value
            case 'num_points':
                self.num_points = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
            
    def _get_param_names(self):
        return ['start_time', 'end_time', 'num_points']