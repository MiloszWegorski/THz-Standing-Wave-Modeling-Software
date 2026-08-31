from abc import ABC, abstractmethod
import numpy as np

c = 299.702547
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

def Sine_wav(times, freq, amp, offset, phase):

    return amp * np.sin(2 * times * np.pi * freq + phase) + offset

def get_wavenums(freq):
    #speed of light in units of mm*GHz

    wavenums = (2 * np.pi * freq)/c

    return wavenums

def get_wavelength(freq):
    
    wavelenght = c/freq

    return wavelenght

def complex_to_mag_and_phase(complex_number):

    magnitude = 20.0 * np.log10(np.abs(complex_number))

    phase = np.rad2deg(np.angle(complex_number))

    return magnitude, phase

def Mag_and_phase_to_complex(magnitude, phase):
    
    complex_num = (10**(magnitude/20.0)) * np.exp(1j * np.pi * (phase/180))

    return complex_num

def low_pass_filter(data, step):
    #average step number of points of all points
    filtered_data = []

    # take a value in the middle of a set of a step number value slice of an
    # array and average the slice returning the smaller array
    for i in range(int((step-1)/2), len(data), step):

        arr_slice = data[int(i-((step-1)/2)): int(i+((step-1)/2))]

        filtered_data.append(np.average(arr_slice))

    return filtered_data

def get_wavenums(freq):
    #speed of light in units of mm*GHz
    c = 299.792458

    wavenums = (2 * np.pi * freq)/c

    return wavenums

def create_component_list(M : int, N : int, transmission : bool, amps):

    index = 0

    if transmission:
        comp_list = np.empty(N*M, dtype=tuple)

        for i in range(N):
            for j in range(M):
                if len(amps) == 0:
                    comp_list[index] = (2*i+1, j)
                    
                else:
                    comp_list[index] = (2*i+1, j, amps[index])
                index += 1

    else:
        comp_list = np.empty(N*M-(M-1), dtype=tuple)


        for i in range(N):
            for j in range(M):
                if len(amps) == 0:
                    comp_list[index] = (2*i, j)
                    
                else:
                    comp_list[index] = (2*i, j, amps[index])
                index += 1

    
    return np.array(comp_list)

def _create_coeff_list(M : int, N : int, transmission : bool, dists, wavenum):

    index = 0

    if transmission:
        comp_list = np.empty((N*M, len(dists)), dtype=complex)

        for i in range(N):
            for j in range(M):
                comp_list[index] = np.power(-dists, j) * np.power(np.exp(-1j * wavenum * dists), 2*i+1)
                index += 1

    else:
        comp_list = np.empty((N*M, len(dists)), dtype=complex)


        for i in range(N):
            for j in range(M):
                comp_list[index] = np.power(-dists, j) * np.power(np.exp(-1j * wavenum * dists), 2*i)
                    
                index += 1

    
    return np.array(comp_list)

def create_coeff_matrix(comp_list, scheme, wavenum):

    dists = scheme.get_points().squeeze()


    coeff_list = np.empty((len(comp_list), len(dists)), dtype=complex)

    for i, (N, M) in enumerate(comp_list):
        coeff_list[i] = np.power(dists, M) * np.exp(-1j * N * wavenum * dists)

    
    return np.array(coeff_list).T

def add_N_component(*, comps):

    max_N = 0

    M_num = 1

    for N in comps:
        if N[0] > max_N:
            max_N = N[0]
            M_num = 1
        else:
            M_num += 1 

    # add N
    comps_new = np.empty(shape=len(comps)+M_num, dtype=tuple)

    for i, comp in enumerate(comps):
        comps_new[i] = comp

    for i, M in enumerate(reversed(range(M_num))):
        comps_new[-(M+1)] = (max_N+2, i)

    return comps_new

def add_M_component(*, comps, M_num):
    
    comps_new = []

    N, M = get_M_and_N(comps=comps)

    counter = 0
    z = 1

    comp_prev = (-1, -1)

    N_min = comps[0][0]

    for i, comp in enumerate(comps):
        comps_new.append(comp)

    for i in range(N_min, N, 2):
        comp_add = (i, M_num)

        if comp_add not in comps_new:
            comps_new.append(comp_add)

    return sorted(comps_new)

def get_M_and_N(comps):
    M_max = 0
    N_max = 0
    N_prev = -1


    for M in comps:
        if M[1] > M_max:
            M_max = M[1]

    for N in comps:
        if N[0] > N_prev:
            N_max += 1
            N_prev = N[0]


    return N_max, M_max
