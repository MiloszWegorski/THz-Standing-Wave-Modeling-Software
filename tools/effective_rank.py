import scipy.linalg
from tools.dependencies import *
import math
import sys
import numpy as np

def get_effective_rank(*, S):
    """Calculates effective rank using (shannon) entropy by using the S decomposition
    from a SVD of a matrix 

        reference : Roy, O. and Vetterli, M. (n.d.). THE EFFECTIVE RANK: A MEASURE OF EFFECTIVE DIMENSIONALITY.
    Args:
        S (np.ndarray): S decomposition from a SVD of a matrix
    Returns:
        Float : Effective rank of the original matrix  
    """

    p_i = S/sum(np.abs(S))

    erank = np.exp(-np.sum(p_i*np.log(p_i)))

    return erank


def get_lowest_rank_by_freq(*, components, freqs, scheme, normalize=True):

    lowest_erank = sys.maxsize

    for f in freqs:
        (_, S, _) = compute_SVD(freq=f, measurement_scheme=scheme, 
                                components=components, normalize=normalize)

        effective_rank = get_effective_rank(S=S)

        if effective_rank < lowest_erank:
            lowest_erank = effective_rank

    return lowest_erank

def compute_SVD(*, freq, measurement_scheme, components, normalize=False):
    
    wavenum = get_wavenums(freq)

    if normalize:
        coeffs = create_legrange_normalized_coeff_list(comp_list=components,
                                                       scheme=measurement_scheme,
                                                       wavenum=wavenum)
    else:
        coeffs = create_coeff_matrix(comp_list=components, 
                                      scheme=measurement_scheme,
                                      wavenum=wavenum)
    
    U, S, Vh = scipy.linalg.svd(coeffs, full_matrices=False)

    return (U, S, np.flip(Vh, axis=0))


def get_s_filter_sum(S, limit):

    S_normalized = S/sum(S)


    sums = []

    delta_sum = 0.0
    filt = []
    for i, val in enumerate(S_normalized):
        delta_sum += val

        sums.append(delta_sum)
        if delta_sum > limit and sums[i]-sums[i-1]< 1-limit:
            filt.append(False)
        else:
            filt.append(True)  


    return filt

def get_s_filter_threshold(S, limit):

    S_normalized = (S/S[0])**2

    filt = []

    for i, val in enumerate(S_normalized):

        if val < limit:
            filt.append(False)
        else:
            filt.append(True)   



    return filt

def get_s_filter_eff_rank(S):
    """Returns a filter which tells which modes are to be filtered out from
    the model

    Args:
        S (np.ndarray): S decomposition from a single value decomposition of the
        component matrix

    Returns:
        np.ndarray: filter made of true false values for the modes of the matrix
    """
    rank = get_effective_rank(S=S)

    filter = np.full(shape=len(S), fill_value=True, dtype=bool)

    for i in range(int(np.round(rank))):

        filter[-(i+1)] = False

    print(filter)

    return filter

def get_variance_of_parameters(*,S, Vh, delta, rcond=None):


    S = np.asarray(S)

    if rcond is None:
        mask = S > 0
    else:
        mask = S > rcond * S.max()

    inv_s2 = np.zeros_like(S, dtype=float)
    inv_s2[mask] = 1.0 / S[mask]**2

    V = Vh.conj().T

    cov = delta * V @ np.diag(inv_s2) @ V.conj().T

    variances = np.real(np.diag(cov))
    stddevs = np.sqrt(variances)

    return cov, variances, stddevs

def covariance_correlated_noise(A, R, rcond=None):
    # Cholesky whitening
    L = np.linalg.cholesky(R)

    # A_white = R^{-1/2} A
    A_white = np.linalg.solve(L, A)

    U, s, Vh = np.linalg.svd(A_white, full_matrices=False)

    if rcond is None:
        mask = s > 0
    else:
        mask = s > rcond * s.max()

    inv_s2 = np.zeros_like(s)
    inv_s2[mask] = 1.0 / s[mask]**2

    V = Vh.conj().T

    cov = V @ np.diag(inv_s2) @ V.conj().T

    variances = np.real(np.diag(cov))
    stddevs = np.sqrt(variances)

    return cov, variances, stddevs, s