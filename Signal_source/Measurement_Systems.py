import numpy as np
from analysis_tools.dependencies import BaseClass

class MeasurementSystem(BaseClass):
    """_summary_
    """
    def __init__(self, scheme, signal):
        """_summary_

        Args:
            scheme (Measurement Scheme): _description_
            signal (Signal): _description_
        """
        self.scheme = scheme
        self.signal = signal

    
    def Measure(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        times = self.scheme.get_points()
        
        return [np.array(times), np.array(self.signal.get_amplitudes(times))]
        #add get variables method
    
    def _get_name(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return f'I measure a normal sine wave'
    
    def _get_param_value(self, param_name):
        if param_name in self.signal.get_param_names():
            return self.signal.get_param_value(param_name)
        elif param_name in self.signal.get_param_names():
            return self.signal.get_param_value(param_name)
        else:
            raise ValueError(f'Unknown parameter name : {param_name}')
            
    def _set_param_value(self, param_name, value):
        if param_name in self.scheme.get_param_names():
            return self.scheme.set_param_value(param_name, value)
        elif param_name in self.signal.get_param_names():
            return self.signal.set_param_value(param_name, value)
        else:
            raise ValueError(f'Unknown parameter name : {param_name}')
            
    def _get_param_names(self):
        param_names = self.signal.get_param_names() + self.scheme.get_param_names()

        return param_names
    

class HornMeasurementSystem(BaseClass):
    """_summary_
    """
    def __init__(self, scheme, signal):
        """_summary_

        Args:
            scheme (Measurement Scheme): _description_
            signal (Signal): _description_
        """
        self.scheme = scheme
        self.signal = signal

    
    def Measure(self, amplitudes):
        """_summary_

        Returns:
            _type_: _description_
        """
        times = self.scheme.get_points()
        
        return [np.array(times), np.array(self.signal.get_amplitudes(amplitudes))]
        #add get variables method
    
    def _get_name(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return f'I measure a normal sine wave'
    
    def _get_param_value(self, param_name):
        if param_name in self.signal.get_param_names():
            return self.signal.get_param_value(param_name)
        elif param_name in self.signal.get_param_names():
            return self.signal.get_param_value(param_name)
        else:
            raise ValueError(f'Unknown parameter name : {param_name}')
            
    def _set_param_value(self, param_name, value):
        if param_name in self.scheme.get_param_names():
            return self.scheme.set_param_value(param_name, value)
        elif param_name in self.signal.get_param_names():
            return self.signal.set_param_value(param_name, value)
        else:
            raise ValueError(f'Unknown parameter name : {param_name}')
            
    def _get_param_names(self):
        param_names = self.signal.get_param_names() + self.scheme.get_param_names()

        return param_names