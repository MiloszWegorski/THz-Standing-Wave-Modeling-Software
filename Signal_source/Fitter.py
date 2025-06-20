import numpy as np
import scipy
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit

from analysis_tools.dependencies import *
class Fitter(ABC):

    def fit_points(self, points):
        return self._fit_points(points)
    
    @abstractmethod
    def _fit_points(self):
        pass


    def get_name(self):
        return self._get_name()
    
    @abstractmethod
    def _get_name():
        pass

    def get_param_names(self):
        return self._get_param_names()
    

    @abstractmethod
    def _get_param_names():
        pass

    def set_param_values(self, param_name, value):
        return self._set_param_values(param_name, value)
    
    @abstractmethod
    def _set_param_values(self, param_name, value):
        pass

class ValleyFitter(Fitter):

    def __init__(self, threshhold : float):
        
        #floating point value between 0.0 and 1.0 representing a level at which
        #the data will be taken as a 'dip' later in the algorithm
        self.threshhold = threshhold


    def _fit_points(self, source_data : np.array):
        
        #passing the raw data through a low pass filter to get more robust data
        #points
        filtered_source_data = np.array([low_pass_filter(source_data[0], 3), 
                                        low_pass_filter(source_data[1], 3)])

        #extracting the values from the 2d array of points for easier indexing
        times = filtered_source_data[0]
        amplitudes = filtered_source_data[1]

        #calculate the minimum and maximum
        min_amp = np.min(amplitudes)
        max_amp = np.max(amplitudes)

        #get amplitude estimate from the minima and maxima found
        amplitude_estimate = (max_amp - min_amp)/2

        #get offset by taking the mean value of all data
        offset_estimate = np.mean(amplitudes)

        #get an array of absolute values of the amplitudes in the data 
        # while accounting for the offset 
        absolute_amplitudes = np.abs(amplitudes - offset_estimate)

        #filter all data which is below the threshhold to get all points where
        #the wave crosses the x axis
        dips = absolute_amplitudes < (self.threshhold * amplitude_estimate)

        #create an empty array same as the array of dips
        Clump_mask = np.ma.masked_array(np.zeros_like(dips))

        #create a mask which is an array of the size of the filtered data
        # this array is made up of true and false values where each mean 
        # whether a value in the source data is contained within the dips array 
        Clump_mask[dips] = np.ma.masked
        Clumps = np.ma.clump_masked(Clump_mask)

        #array to store the average time of each position where the wave 
        #the x axis
        averages_of_clumps = []

        #calculating the averages of these points
        for i in Clumps:
            averages_of_clumps.append(np.average(times[i]))

        #calculate the distance between the points where the sine wave crosses
        #the 0 line
        half_period = np.average(np.diff(averages_of_clumps))

        #estimate the frequency from the time above
        frequency_estimate = 1/ (2* half_period)

        #estimate phase from the first point where the wave crosses the x axis
        delta_t = averages_of_clumps[0]

        #calculate the index of the first peak
        index_peak = np.int32((Clumps[0].stop + Clumps[1].start)/2)

        #check whether the peak between the first and second clump is positive 
        # or negative
        if (amplitudes[index_peak]- offset_estimate) > 0:
            #for a negative peak the first peak is positive therefore the phase
            #can simply be returned
            
            phase = (((delta_t/half_period)+1)*np.pi)

        elif (amplitudes[index_peak] - offset_estimate) < 0:
            #if peak is positive the first peak is negative therefore the phase
            #has to be adjusted by half a wavelength
            phase = ((delta_t/half_period)*np.pi)

        #using curve fit to improve the estimate of the parameters of the wave
        popt, pcov = curve_fit(Sine_wav, times, amplitudes, 
                        p0=(frequency_estimate, amplitude_estimate, 
                            offset_estimate, phase), 
                        maxfev=int(1e6), 
                        bounds=([-np.inf, 0.5*amplitude_estimate, 
                                 -np.inf, 0], 
                                [np.inf, np.inf,
                                 np.inf, 2*np.pi]
                                )
                            )

        #popt = [frequency_estimate, amplitude_estimate, offset_estimate, phase]
        return popt
    

    def _get_name():
        pass

    def _get_param_names():
        return ['frequency', 'amplitude', 'offset', 'phase']

class HornTransmissionFitter(Fitter):


    def __init__(self, coefficient_matrix):
        
        self.coeff_list = coefficient_matrix

    def _fit_points(self, data):

        fit = np.linalg.lstsq(np.rot90(self.coeff_list), data[1], -1)
            
        return fit[0]

    def _get_name(self):
        return f'fitter which fits a signal with {self.num_reflections} reflections\
            and each of these having {self.components} components at wavenumber {self.wavenum}'

    def _get_param_names(self):
        return ['frequency', 'num_reflections', 'num_components', 'transmission']

    def _set_param_values(self, param_name, value):
        match param_name:
            case 'frequency':
                self.wavenum = get_wavenums(value)
            case 'num_reflections':
                self.num_reflections = value
                self.component_list = create_component_list(self.components, self.num_reflections, self.transmission)
            case 'num_components':
                self.components = value
                self.component_list = create_component_list(self.components, self.num_reflections, self.transmission)
            case 'transmission':
                self.transmission = value
                self.component_list = create_component_list(self.components, self.num_reflections, self.transmission)
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  

    def _get_param_values(self, param_name):
        match param_name:
            case 'components':
                return self.components
            case 'num_reflections':
                return self.num_reflections
            case 'num_components':
                return self.wavenum
            case 'transmission':
                return self.transmission
            case _:
                raise ValueError(f'Unknown parameter name {param_name}')  