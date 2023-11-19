import numpy as np
from math import factorial
from scipy.signal import (
    filtfilt,
    butter,
    lfilter,
    bessel,
    firwin,
    decimate,
    kaiserord,
    kaiser,
    iirnotch,
    sosfiltfilt,
)
from scipy.ndimage.filters import uniform_filter1d
import logging
import gc
import pywt

logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s %(levelname)s - %(message)s"
)


def baselinePlotDecimate(data, old_rate, d=10):
    """
    Exports filtered and subsampled data from primarly sampling system, as the experiment and runconstants dictate. For plotting only (in eventfinder)

    Parameters
    ----------
    data : nparray
        The data to decimate.

    Returns
    -------
    None.
    """
    new_data = lowpass_resample(data, old_rate, old_rate / d, False, logInfo=False)
    new_data[0] = new_data[
        1
    ]  # BUG? to supress plotting artifact of filter intiaization - tested this on 20191023! original data is sane
    lengthOfData = np.size(new_data)
    new_samplerate = old_rate * lengthOfData / np.size(data)
    time = np.arange(0, lengthOfData, 1) / new_samplerate

    # supressing decimation filter artefact in plot output
    new_data[0] = new_data[2]
    new_data[1] = new_data[2]
    return time[0:-2], new_data[0:-2]


# TODO noticed phase lag of moving average filter, tried to do ultra agressive FIR filter intead of averaging, PROBLEM: out of memory
# Try: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html
def detrend_data(a, window):
    """
    Calculates a forward/backward lowpass with window size. This function is best use to remove drift and baseline fluctuations before starting event finding. If window is set too small it deminishes the events depths.

    Parameters
    ----------
    a : nparray
        The data to detrend.

    window : int
        Window size for filter

    Returns
    -------
    out : nparray
        Filtered data.
    """
    out = uniform_filter1d(a, size=window)
    return out


# @njit(cache=True) TODO breaks with
# ""
# numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
# Untyped global name 'butter': cannot determine Numba type of <class 'function'>
# ""
def butter_lowpass(highcut, fs, order=9, sosFilter=True):
    """
    Depricated, because of low stability.  TODO is it?
    Synthesizes a lowpass Butterworth filter of fixed order.

    Parameters
    ----------
    highcut : float
        The cutoff frequency.
    fs : double
        The sample frequency of the sampled data.
    order : int
        The order of the lowpass Butterworth filter.

    Returns
    -------
    a, b : filter coefficents
        Filter coefficent objects for putting into a filter call.
    """
    nyq = 0.5 * fs
    high = highcut / nyq

    if sosFilter:
        sos = butter(order, high, output="sos")
        return sos
    else:
        b, a = butter(order, high)
        return b, a


def bessel_lowpass(highcut, fs, order=9, sosFilter=True):
    """
    Depricated, because of low stability.  TODO is it?

    Synthesizes a low-pass Bessel filter of fixed order.

    Parameters
    ----------
    highcut : float
           The cutoff frequency.
    fs : double
        The sample frequency of the sampled data.
    order : int
        The order of the lowpass Bessel filter.

    Returns
    -------
    a, b : filter coefficents
        Filter coefficent objects for putting into a filter call.
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    if sosFilter:
        sos = bessel(order, high, output="sos")
        return sos
    else:
        b, a = bessel(order, high, output="ba")
        return b, a


def FIR_lowpass(highcut, fs):
    """
    Synthesizes a low-pass Finite Impule Response filter of fixed order that depends on sample rate and cutoff frequency

    Parameters
    ----------
    highcut : float
           The cutoff frequency.
    fs : double
        The sample frequency of the sampled data.
    order: int
        The order of the filter.

    Returns
    -------
    a, b : filter coefficents
        Filter coefficent objects for putting into a filter call.
    """
    numtaps = int(
        np.floor(
            2.0 / 3.0 * np.log10(1.0 / 10.0 / 1.0e-4 / 1.0e-3) * fs / highcut * 2.0
        )
    )
    logging.info("Applying lowpass - Hamming numtaps are " + str(numtaps))
    taps = firwin(numtaps, highcut, window="hamming", pass_zero=1, fs=fs)
    a = [1.0]

    return taps, a


def kaiser_lowpass(highcut, fs):
    """
    Synthesizes a low-pass Kaiser filter of fixed order that depends on sample rate and cutoff frequency.

    Parameters
    ----------
    highcut : float
        The cutoff frequency.
    fs: double
        The sample frequency of the sampled data.
    order: int
        The order of the filter.

    Returns
    -------
    a, b: filter coefficents
        Filter coefficent objects for putting into a filter call.
    """
    nyq = fs * 0.5
    width = highcut / 15
    att = 10  # in dB
    numtaps, beta = kaiserord(att, width / nyq)
    logging.info("Applying lowpass - Kaiser numtaps are " + str(numtaps))
    taps = firwin(numtaps, highcut, window=("kaiser", beta), scale=False, fs=fs)
    a = [1.0]
    return taps, a


def lowpass_resample(data, oldSampleRate, sampleRate, filteredInt, logInfo=True):
    """
    Inputs data sampled at oldSampleRate, and a new sampleRate
    returns data resampled at sampleRate, downsampled using scipy.signal.decimate
    filteredInt: Flag to set if you want to subsample and return int (default for Chimera data) or whether you want to return float32 (default for smFET)

    loginfo: Flag whether to print out log or not. (the event finder is using this decimation feature often and heavily) and it clutters a lot the printed output.

    Parameters
    ----------
    data : nparray
        The input data
    oldSampleRate: double
        The sample frequency of the sampled data.
    sampleRate : double
        The sample frequency of the sampled data that you'd like to downsample to.
    filteredInt: int
        Flag to set if you want to subsample and return int (default for Chimera data) or whether you want to return float32 (default for smFET)
    logInfo : Bool
        Do you want verbose mode?

    Returns
    -------
    data : nparray
        Resampled data.
    """
    downScale = int(np.floor(oldSampleRate / sampleRate))

    if downScale == 1:
        if logInfo:
            logging.info(
                "No resampling, since data is at target sample rate of "
                + str(oldSampleRate)
            )
        return data

    elif downScale < 1:
        if logInfo:
            logging.info(
                "Resampling not possible, since you have asked for increasing the sample rate, instead of decimating.  Returning original data. original samplerate: {s0}, requested samplerate: {s1}".format(
                    s0=oldSampleRate, s1=sampleRate
                )
            )
        return data

    elif downScale > 10:
        if logInfo:
            logging.info(
                "Resample in two steps with total scale factor " + str(downScale)
            )
        data = decimate(
            data, int(np.floor(np.sqrt(downScale))), ftype="fir", zero_phase=True
        )
        if filteredInt:
            return np.around(
                decimate(
                    data,
                    int(np.floor(np.sqrt(downScale))),
                    ftype="fir",
                    zero_phase=True,
                )
            ).astype(np.int16, copy=False)
        else:
            return decimate(
                data,
                int(np.floor(np.sqrt(downScale))),
                ftype="fir",
                zero_phase=True,
            )
    else:
        if logInfo:
            logging.info(
                "resample in one step with total scale factor " + str(downScale)
            )
        if filteredInt:
            return np.around(
                decimate(data, downScale, ftype="fir", zero_phase=True)
            ).astype(np.int16, copy=False)
        else:
            return decimate(data, downScale, ftype="fir", zero_phase=True)


def lowpass_filter(data, rc, fs, order=9, fType="Butter", resample=4.0):
    """
    Actually runs the lowpass filter on the relevant data, based on one of the input filter types, the data, and the parameters supplied here.

    Note: the first 20 values are overwritten with the mean of the second 20 values, to avoid filter artifacts.

    This function includes a case for filtering down in 2 steps to increase filter robustness and computational speed.

    Additionally the function returns either int parameter (if sampling freq >200000, chimera) or float (if sampling freq < 200000 smFET, heka)


    Parameters
    ----------
    data : nparray
        The input data
    rc : RunConstants
        RunConstants for this run
    fs : double
        The sample frequency of the sampled data that you'd like to downsample to.
    order : int
        The filter order, for Bessel and Butterworth.
    fType : string
        The filter type: Butter, Bessel, FIR, Kaiser
    resample : double
        The downsampling rate

    Returns
    -------
    data : nparray
        Resampled, low-pass filtered data.
    """
    filt_cutoff = rc.args.filter
    filterType = rc.args.filterType

    if fs > 200000:
        filterThreshold = 2.5e4
        filteredInt = True
    else:
        filterThreshold = 250
        filteredInt = False

    if filt_cutoff < fs:
        if filt_cutoff < filterThreshold:

            originalSize = np.size(data)

            data = LP_filter(
                data, filterThreshold, fs, order, filterType, resample, filteredInt
            )
            sampleRateFilt = fs * np.size(data) / originalSize
            logging.info(
                "Done filtering at "
                + str(filterThreshold)
                + "Hz now filtering to "
                + str(filt_cutoff)
                + "Hz"
            )
            data = LP_filter(
                data,
                filt_cutoff,
                sampleRateFilt,
                order,
                filterType,
                resample,
                filteredInt,
            )
        else:
            data = LP_filter(
                data, filt_cutoff, fs, order, filterType, resample, filteredInt
            )

    return data


def LP_filter(
    data, highcut, fs, order=9, fType="Butter", resample=4.0, filteredInt=True
):
    """
    Actually runs the lowpass filter on the relevant data, based on one of the input filter types, the data, and the parameters supplied here.
    Important: to avoid filter onset issues initial 21 values are overwritten with mean of 21-41 values of data array

    Parameters
    ----------
    data : nparray
        The input data
    rc : RunConstants
        RunConstants for this run
    fs : double
        The sample frequency of the sampled data that you'd like to downsample to.
    order : int
        The filter order, for Bessel and Butterworth.
    fType : string
        The filter type: Butter, Bessel, FIR, Kaiser
    resample : double
        The downsampling rate

    Returns
    -------
    data : nparray
        Resampled, low-pass filtered data.
    """
    # applies filter of type "type" options - 'Butter', 'Bessel', 'FIR', 'Kaiser'
    if highcut >= fs / 2.0:
        highcut = 0.49 * fs

    sosFilter = True

    if fType == "Bessel" and sosFilter:
        sosParam = bessel_lowpass(highcut, fs, order=order, sosFilter=sosFilter)
    elif fType == "Butter" and sosFilter:
        sosParam = butter_lowpass(highcut, fs, order=order, sosFilter=sosFilter)
    elif fType == "Bessel":
        b, a = bessel_lowpass(highcut, fs, order=order, sosFilter=sosFilter)
    elif fType == "Butter":
        b, a = butter_lowpass(highcut, fs, order=order, sosFilter=sosFilter)
    elif fType == "FIR":
        b, a = FIR_lowpass(highcut, fs)
    elif fType == "Kaiser":
        b, a = kaiser_lowpass(highcut, fs)

    if filteredInt:
        if sosFilter and (fType in ["Bessel", "Butter"]):
            data = np.around(sosfiltfilt(sosParam, data, padtype="even")).astype(
                np.int16, copy=False
            )
        else:
            data = np.around(
                filtfilt(b, a, data, padlen=3 * max(len(a), len(b)), padtype="even")
            ).astype(np.int16, copy=False)
    else:
        if sosFilter and (fType in ["Bessel", "Butter"]):
            data = sosfiltfilt(sosParam, data, padtype="even")
        else:
            data = filtfilt(b, a, data, padlen=3 * max(len(a), len(b)), padtype="even")

    if sosFilter and (fType in ["Bessel", "Butter"]):
        del sosParam
    else:
        del a, b

    gc.collect()

    if resample > 1.0:
        newSampleRate = highcut * resample
        data = lowpass_resample(data, fs, newSampleRate, filteredInt)
    data[0:20] = np.mean(data[21:41])  # just mark this as something like a BUG...
    return data


def notch_filter(data, w0, Q, sampleRate):
    """
    Basic implementation of analog notch filter. Can be used to remove 50/60Hz hum or similar artifacts. Is activate by keyword --notch but frequency is right now hardcoded

    Parameters
    ----------

    data : nparray
        Sample data.
    stopband: list
        Stop band for the bandstop filter.
    fs : int
        Sample frequency
    fType: string
        What type filter is this?

    Returns
    -------
    a, b : filter coefficents
        Filter coefficent objects for putting into a filter call of the correct type.
    """
    b, a = iirnotch(w0, Q, sampleRate)
    y = filtfilt(b, a, data, padlen=3 * max(len(a), len(b)))
    return y


def bandstop_filter(data, stopband, fs, fType="hamming"):
    """
    Selects and runs one type of bandstop filter, with pre-selected numtaps.

    Parameters
    ----------

    data : nparray
        Sample data.
    Stopband : np.array int
        [start1, end1, start2, end2, start3, ....] of supressed frequencies
    fs : int
        Sample frequency
    fType: string
        What type filter is this?

    Returns
    -------
    a, b: filter coefficents
        Filter coefficent objects for putting into a filter call of the correct type.
    """
    if fType == "hamming":
        b, a = hamming_bandstop(stopband, fs)
    elif fType == "kaiser":
        b, a = kaiser_bandstop(stopband, fs)

    y = filtfilt(b, a, data, padlen=3 * max(len(a), len(b)))
    del data, a, b
    gc.collect()
    return y


def hamming_bandstop(stopband, fs):
    """
    Hamming window implementation of bandstop filter with adaptive window size.

    Parameters
    ----------

    Stopband : np.array int
        [start1, end1, start2, end2, start3, ....] of supressed frequencies
    fs : double
        The sample frequency of the sampled data.

    Returns
    -------
    a, b : filter coefficents
        Filter coefficent objects for putting into a filter call.
    """
    nyq = 0.5 * fs
    numtaps = int(
        np.floor(
            2.0 / 3.0 * np.log10(1.0 / 10.0 / 1.0e-4 / 1.0e-3) * fs / stopband[-1] * 3.0
        )
    )
    if (numtaps % 2) == 0:
        numtaps = numtaps + 1
    logging.info("applying Bandstop - Hamming numtaps are " + str(numtaps))
    taps = firwin(numtaps, stopband, window="hamming", pass_zero="bandstop", fs=fs)
    a = [1.0]
    return taps, a


def kaiser_bandstop(stopband, fs):
    """
    Kaiser window implementation of bandstop filter with adaptive window size.

    Parameters
    ----------

    Stopband : np.array int
        [start1, end1, start2, end2, start3, ....] of supressed frequencies
    fs : double
        The sample frequency of the sampled data.

    Returns
    -------
    a, b : filter coefficents
        Filter coefficent objects for putting into a filter call.
    """
    nyq = 0.5 * fs
    width = 120
    att = 13  # in dB
    numtaps, beta = kaiserord(att, width / nyq)
    if (numtaps % 2) == 0:
        numtaps = numtaps + 1
    logging.info("applying Bandstop - Kaiser numtaps are " + str(numtaps))
    taps = firwin(numtaps, stopband, window=("kaiser", beta), scale=False, fs=fs)
    a = [1.0]
    return taps, a


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Passes data through Savitzky-Golay filter.

    from https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    Parameters
    ----------
    data : nparray
        The data to filter

    window_size: int
        size of filter window

    order: int
        filter order

    deriv: int
        size of filter dervivative

    rate: int
        filter rate

    Returns
    -------
    out : nparray
        Filtered data
    """
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat(
        [[k ** i for i in order_range] for k in range(-half_window, half_window + 1)],
        dtype="float",
    )
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def wavelet_filter(ADCData, rc):
    """
    Wavelet filter implementation

    CAUTION:
    * Time resolution of filtered data unclear. At best it is resultion of subsampled data.
    * Should be compared to standard FIR filter at desired frequency. So run example: -f 1000 AND independently -f 1000 --wavelet rbio1.1
    *

    Parameters
    ----------
    ADCData : nparray
    The data to filter

    rc : rc DATACLASS object
        rc.args.wavelet STR
            STR = default value is 'none' which makes the wavelet filter inactive, available wavelets for pywt : http://wavelets.pybytes.com
        rc.args.waveletthreshold FLOAT
            FLOAT = additional filter threshold parameter should be 1 per default - not really clear what this does
        rc.args.waveletlevel INT
            INT = level of wavelet filter decomposition for smFET 5 is a good value (maximum) and Chimera (maybe 8)

    Returns
    -------
    filteredData : nparray
        Filtered data
    """
    motherWavelet = rc.args.wavelet
    level = rc.args.waveletlevel
    maxlevel = pywt.swt_max_level(len(ADCData))

    if (
        (not hasattr(rc.args, "instrumentType"))
        or ("Chimera" in rc.args.instrumentType)
    ) and rc.args.waveletlevel > maxlevel:
        level = 8
        logging.info(
            "Wavelet filtering attempt with too high level reset to " + str(level)
        )

    elif (
        (not hasattr(rc.args, "instrumentType")) or ("smFET" in rc.args.instrumentType)
    ) and rc.args.waveletlevel > maxlevel:
        level = 5
        logging.info(
            "Wavelet filtering attempt with too high level reset to " + str(level)
        )

    waveletCoefficients = pywt.swt(ADCData, motherWavelet, level=level)
    logging.info(
        "Wavelet filtering of data with wavelet {wav} to level = {lv:d}".format(
            wav=motherWavelet, lv=level
        )
    )

    for i, all_coeff in enumerate(waveletCoefficients):
        detail_coeffs = all_coeff[1]
        detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745 * rc.args.waveletthreshold
        threshold = sigma * np.sqrt(
            2 * np.log10(len(ADCData))
        )  # / np.log10(level - i + 1)
        if i == 0:
            waveletCoefficients[i] = (
                all_coeff[0],
                np.array(
                    pywt.threshold(np.abs(all_coeff[1]), value=threshold, mode="hard")
                ),
            )
        else:
            waveletCoefficients[i] = (
                np.array(
                    pywt.threshold(np.abs(all_coeff[0]), value=threshold, mode="hard")
                ),
                np.array(
                    pywt.threshold(np.abs(all_coeff[1]), value=threshold, mode="hard")
                ),
            )

    filteredData = pywt.iswt(tuple(waveletCoefficients), motherWavelet)
    return filteredData
