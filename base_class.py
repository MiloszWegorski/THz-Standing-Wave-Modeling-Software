from abc import ABC, abstractmethod

class BaseClass(ABC):

    def get_name(self):
        return self._get_name()
    
    @abstractmethod
    def _get_name(self):
        pass

    def get_param_value(self, param_name):
        return self._get_param_value(param_name)
    
    @abstractmethod
    def _get_param_value(self, param_name):
        pass

    def set_param_value(self, param_name, param_value):
        return self._set_param_value(param_name, param_value)
    
    @abstractmethod
    def _set_param_value(self, param_name, param_value):
        pass

    def get_param_names(self):
        return self._get_param_names()
    
    @abstractmethod
    def _get_param_names(self):
        pass
