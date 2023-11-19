#######################################
## author: Jakob Buchheim
## company: Columbia University
##
## version history:
##      * created: 20190610
##
## description:
## *
## * creates fft transform of experiment
#######################################

import numpy as np
import gc
from scipy.signal import welch, periodogram

from numba import jit


def fft_welch(data_set, sampleRate, nperseg, noverlap):
    """
    FFT of maximal 20s of time series data.

    Parameters
    ----------
    data_set : nparray
        The data to fft.
    samplerate : float
        samplerate for the inbound dataset
    nperseg : int
        number of points per segment
    noverlap :  float
        overlap ratio for segments

    Returns
    -------
    transform, Pxx: nparray, nparray
        The frequency and fft arrays of the input data.

    """
    maxIdx = np.int(np.floor(20.0 * sampleRate))
    if np.size(data_set) < maxIdx:
        maxIdx = np.size(data_set)

    average_i = np.around(np.mean(data_set[0:maxIdx])).astype(
        dtype=np.int16, copy=False
    )
    transform, Pxx_den = welch(
        data_set[0:maxIdx] - average_i,
        sampleRate,
        nperseg=nperseg,
        noverlap=noverlap,
        average="median",
    )

    return (
        transform.astype(dtype=np.float32, copy=False),
        Pxx_den.astype(dtype=np.float32, copy=False),
    )


def fft_periodogram(data_set, sampleRate):
    """
    Exports filtered and subsampled data from primarly sampling system, as the experiment and runconstants dictate.

    Parameters
    ----------
    data : nparray
        The data to flip into a periodogram.
    sampleRate : float
        Data samplerate

    Returns
    -------
    transform, Pxx: nparray, nparray
        The frequency and periodogram of the data.
    """
    average_i = np.mean(data_se, dtype=np.int16)
    transform, Pxx_den = periodogram(data_set - average_i, sampleRate)
    return np.float32(transform), np.float32(Pxx_den)
