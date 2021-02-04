# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import cmath
import os

import numpy as np
from numba import njit


# =======================================================================================================
# Simple functions that worked in Numba
# =======================================================================================================
@njit(fastmath=True, cache=True)
def downsample(x, n):
    len_y = int(np.ceil(len(x) / n))
    y = np.zeros(len_y, dtype=np.float64)
    for i in range(len_y):
        y[i] = x[i * n]

    return y


@njit(fastmath=True, cache=True)
def histc(X, bins):
    # https://stackoverflow.com/questions/32765333/how-do-i-replicate-this-matlab-function-in-numpy
    map_to_bins = np.digitize(X, bins)
    r = np.zeros(bins.shape, dtype=np.int32)
    for i in map_to_bins:
        r[i - 1] += 1
    return r, map_to_bins


@njit(fastmath=True, cache=True)
def numba_dft(x=None, sign=-1):
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        nFFT = int(2 ** (np.ceil(np.log2(np.abs(N))) + 1))
        z = np.zeros(nFFT, dtype=x.dtype)
        z[:N] = x
        x = z
        N = nFFT

    dft = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        series_element = 0
        for n in range(N):
            series_element += x[n] * cmath.exp(sign * 2j * cmath.pi * i * n * (1 / N))
        dft[i] = series_element
    if sign == 1:
        dft = dft / N
    return dft


@njit(fastmath=True, cache=True)
def numba_fft_v(x, sign=-1):
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        nFFT = int(2 ** (np.ceil(np.log2(np.abs(N))) + 1))
        z = np.zeros(nFFT, dtype=x.dtype)
        z[:N] = x
        x = z
        N = nFFT

    x = np.asarray(x, dtype=np.complex128)

    N_min = min(N, 2)

    n = np.arange(N_min, dtype=np.float64)
    k = n.T.reshape(-1, 1)
    M = np.exp(sign * 2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(sign * 1j * np.pi * np.arange(X.shape[0]) / X.shape[0]).T.reshape(-1, 1)
        X = np.vstack((X_even + terms * X_odd,
                       X_even - terms * X_odd))
    if sign == 1:
        return X.ravel() / N
    return X.ravel()


@njit(fastmath=True, cache=True)
def autocorr(y, fft):
    l = len(y)

    fft = fft * np.conjugate(fft)
    acf = numba_fft_v(fft, sign=1)

    acf = acf.real
    acf = acf / acf[0]
    acf = acf[:l]

    return acf


@njit(fastmath=True, cache=True)
def numba_std(values, mean):
    if len(values) == 1:
        return 0
    sum_squares_diff = 0
    for v in values:
        diff = v - mean
        sum_squares_diff += diff * diff

    return np.sqrt(sum_squares_diff / (len(values) - 1))


@njit(fastmath=True, cache=True)
def numba_min(a, b):
    if a < b:
        return a
    return b


@njit(fastmath=True, cache=True)
def numba_max(a, b):
    if a < b:
        return b
    return a


@njit(fastmath=True, cache=True)
def numba_linear_regression(x, y, n, lag):
    co = np.zeros(2, dtype=np.float64)
    sumx, sumx2, sumxy, sumy = 0, 0, 0, 0

    for i in range(lag, n + lag):
        sumx += x[i]
        sumx2 += x[i] * x[i]
        sumxy += x[i] * y[i]
        sumy += y[i]

    denom = n * sumx2 - sumx * sumx
    if denom != 0:
        co[0] = (n * sumxy - sumx * sumy) / denom
        co[1] = (sumy * sumx2 - sumx * sumxy) / denom

    return co


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path
