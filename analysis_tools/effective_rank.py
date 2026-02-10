import numpy as np
import matplotlib.pyplot as plt
from libla import Matrix
from Signal_source.Model_signals import create_component_list, get_wavenums, create_coeff_list
import scipy

components = [(1,0), (1,1), (1,2), (3,0), (3,1), (3,2), (5,0), (5,1), (5,2)]

def get_effective_rank(*, freq, measurement_scheme, components=components):
    
    wavenum = get_wavenums(freq)
    
    coeffs = Matrix(create_coeff_list(comp_list=components, 
                                      scheme=measurement_scheme,
                                      wavenum=wavenum))
    
    return coeffs.rank_decomposition(method='svd'), coeffs
