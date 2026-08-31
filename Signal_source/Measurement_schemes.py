import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from tools.dependencies import BaseClass


class MeasurementScheme(BaseClass):
    
    def get_points(self):
        return self._get_points()
    
    @abstractmethod
    def _get_points(self):
        pass


class UniformMeasurement(MeasurementScheme):

    def __init__(self, *, start_position, end_position, num_points):
        self.start_position = start_position
        self.end_position = end_position
        self.num_points = num_points

        self.name = f"Uniform distribution from {start_position} to {end_position}\
 with {num_points} points"

    def _get_points(self):
        return np.linspace(self.start_position, self.end_position, self.num_points)
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'start_position':
                return self.start_position
            case 'end_position':
                return self.end_position
            case 'num_points':
                return self.num_points
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'start_position':
                self.start_position = param_value
            case 'end_position':
                self.end_position = param_value
            case 'num_points':
                self.num_points = param_value
            case _:
                raise ValueError('Unknown parameter name') 
            
    def _get_param_names(self):
        return ['start_position', 'end_position', 'num_points']

class RandomizedUniformMeasurement(MeasurementScheme):

    def __init__(self, *, start_position, end_position, num_points, randomization):
        self.start_position = start_position
        self.end_position = end_position
        self.num_points = num_points
        self.randomization_percentage = randomization

        uniform_scheme = np.array(np.linspace(self.start_position, self.end_position, self.num_points))

        step = uniform_scheme[1] - uniform_scheme [0]

        self.randomized_scheme = np.array(uniform_scheme + 
                            np.random.normal(0.0, step *self.randomization_percentage,
                                             size=uniform_scheme.shape))

        self.name = f"Uniform distribution from {start_position} to {end_position}\
 with {num_points} points"

    def _get_points(self):

        return self.randomized_scheme
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'start_position':
                return self.start_position
            case 'end_position':
                return self.end_position
            case 'num_points':
                return self.num_points
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'start_position':
                self.start_position = param_value
            case 'end_position':
                self.end_position = param_value
            case 'num_points':
                self.num_points = param_value
            case _:
                raise ValueError('Unknown parameter name') 
            
    def _get_param_names(self):
        return ['start_position', 'end_position', 'num_points']
    

class CustomArrayMeasurement(MeasurementScheme):

    def __init__(self, *, point_array):
        self.point_array = point_array

        self.name = f"Returns custom array as measurement scheme"

    def _get_points(self):

        return np.array(self.point_array)
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'point_array':
                return self.point_array
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'point_array':
                return self.point_array == param_value
            case _:
                raise ValueError('Unknown parameter name') 
            
    def _get_param_names(self):
        return ['point_array']


class LogMeasurement(MeasurementScheme):

    def __init__(self, start_position, end_position, num_points):
        self.start_position = start_position
        self.end_position = end_position
        self.num_points = num_points

        self.name = f"Logarithmic distribution from {start_position} to {end_position}\
 with {num_points} points"

    def _get_points(self):
        return np.logspace(self.start, self.end, self.num_points)
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'start_position':
                return self.start_position
            case 'end_position':
                return self.end_position
            case 'num_points':
                return self.num_points
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'start_position':
                self.start_position = param_value
            case 'end_position':
                self.end_position = param_value
            case 'num_points':
                self.num_points = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
            
    def _get_param_names(self):
        return ['start_position', 'end_position', 'num_points']
    

class DummyMeasurementScheme(MeasurementScheme):

    def __init__(self, dimensions):
        self.dimensions = dimensions

        self.name = f""

    def _get_points(self):
        if self.dimensions == 1:
            return np.zeros(1)
        else:
            return np.zeros((1, self.dimensions))
    
    def _get_name(self):
        return self.name
    
    def _get_param_value(self, param_name):
        match param_name:
            case 'dimensions':
                return self.dimensions
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  
    
    def _set_param_value(self, param_name, param_value):
        match param_name:
            case 'dimensions':
                self.dimensions = param_value
            case _:
                raise ValueError(f'Unknown parameter name {param_name}') 
            
    def _get_param_names(self):
        return ['dimensions']