import numpy as np
from pathlib import Path
import sys
import math
import copy
import pandas as pd
import gc
from numba import njit
from datetime import datetime
from datetime import timedelta
import os
import scipy.optimize
import csv

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.append(str(Path.cwd() / "source" / "utils"))
sys.path.append(str(Path.cwd() / "source" / "classes"))
sys.path.append(str(Path.cwd() / "source"))
import import_utils
import export_utils
import filter_utils
import plot_utils
import fft_utils
import hmm_utils
import baseline_utils

def get_smFET_gatefile(exp):
    """
    This takes in an Experiment() that's of type _IV_, according to the instructions given by Erik, and gets the gatefile data for this run.

    Probably there's a way better way of doing this, since this reads disk all the time, but that would require refactoring Experiment() to accomodate Gate files.  Using get_data_smfet won't work, as the files seem coded differently.

    Parameters
    ----------
    exp : Experiment()
        This is the current data subject to fitting.

    Returns
    -------
    V : nparray
        The voltages that were applied during the full gate sweep, which means they are listed mutiple times.
    """
    assert "_IV_" in exp.name
    the_file = exp.metaData["DATA_FILES"][0]

    the_dir = the_file.parent
    the_stem = (
        str(the_file.stem).split("Chan")[0] + "Gate.bin"
    )  # not ideal, but about as good as the convention they have...
    the_gate_file = the_dir / the_stem
    gate_values = []

    with open(the_gate_file, "rb") as h:
        for line in h:
            gate_values.append(float(line))

    return np.asarray(gate_values)


# maybe a @njit(cache=True), but check if that fails on arrays...
def get_sigmas(I, V):
    """
    This is annoying -- this takes a a matching array of I and V, and returns standard deviation of the I at fixed V.  I and V have the same length -- every pair (V,I) corresponds to a measurement of I at that V at a different time, and the return is the standard deviation of the I at each V.

    Parameters
    ----------
    I : nparray
        The current of the IV
    V : nparray
        The voltage of the IV

    Returns
    -------
    sigma : list of length 2 of nparrays.
        The standard deviation of the multiple runs of the mean, according to the way Erik wants this.

    """
    assert len(I) == len(V)
    v, i = np.unique(V, return_inverse=True)
    voltages = []
    sigmas = []
    for index, guy in enumerate(
        v
    ):  # There's probably some better way of doing this than low-level for loops...
        scratch = []
        voltages.append(guy)  # grab each unique voltage...
        for idx, thing in enumerate(i):
            if thing == index:  # If the index of the unique voltage matches...
                scratch.append(I[idx])  # ... add the relevant current to the pile...
        sigmas.append(np.std(scratch))  # take the std of the pile and record it
    assert len(voltages) == len(sigmas)
    return [np.asarray(voltages), np.asarray(sigmas)]


def get_smFET_IV(exp):
    """
    This takes in an Experiment() that's of type _IV_, according to the instructions given by Erik, and returns the I and V data points, for plotting or something...

    Parameters
    ----------
    exp : Experiment()
        This is the current data for this experiment.

    Returns
    -------
    I : nparray
        The current of the IV
    V : nparray
        The voltage of the IV
    sigma : float
        The standard deviation of the multiple runs of the mean I

    """
    assert "_IV_" in exp.name
    assert (
        len(exp.metaData["DATA_FILES"]) == 1
    ), "You don't have a way yet of handling multifile smfet data."

    # now start stealing code from import_utils.read_params_smfet(), and you can't use that directly, since only IV's have a measure_time
    logFilePath = exp.metaData["DATA_FILES"][0]
    keepers = [
        "Measure Time (s)",
    ]
    setupParameter = dict.fromkeys(keepers)
    base_path = os.path.dirname(logFilePath)
    thing = logFilePath.parent.rglob("*.param")
    params_files = list(thing)
    assert (
        len(params_files) == 1
    ), "Assertion error in IV measure_time reader: one params file per dir."
    matFilePath = os.path.join(base_path, params_files[0])
    readerDict = import_utils.file_to_dict(matFilePath)
    sample_rate = exp.metaData["SETUP_ADCSAMPLERATE"]  # points per time
    measure_time = float(readerDict[("Measure Time (s)")])
    del readerDict
    block_size = sample_rate * measure_time

    voltages = get_smFET_gatefile(exp)

    exp.load()

    I = []
    for i, v in enumerate(voltages):
        start = int(block_size * i)
        end = int(start + block_size)
        the_current = np.mean(exp.Ia[start:end])
        I.append(the_current)

    exp.unload()

    sigma = get_sigmas(I, voltages)
    # go look at the output of get_sigmas -- out[0] is the voltages, out[1] is a single std of the thing..

    v, i = np.unique(voltages, return_inverse=True)
    currents = []
    voltages = []
    for index, guy in enumerate(v):
        scratch = []
        voltages.append(guy)
        for idx, thing in enumerate(i):
            if thing == index:
                scratch.append(I[idx])
        currents.append(np.mean(scratch))

    return np.asarray(currents), np.asarray(voltages), sigma[1]


def error_function_single(popt, T, I):
    """
    Error function for the current to single exponential function.
    """
    assert len(T) == len(I)
    error = np.linalg.norm(I - fit_target_func_single(T, *popt))
    return error


def error_function_double(popt, T, I):
    """
    Error function for the current to double exponential function.
    """
    assert len(T) == len(I)
    error = np.linalg.norm(I - fit_target_func_double(T, *popt))
    return error


@njit(cache=True)
def fit_target_func_single(t, a, b, c):
    """
    Fits the current to single exponential function.
    """
    return a * np.exp(-t / b) + c


@njit(cache=True)
def fit_target_func_double(t, a, b, c, d, e):
    """
    Fits the current to double exponential function.
    """
    return a * np.exp(-t / b) + c * np.exp(-t / d) + e


def fit_the_current_double(T, I):
    """
    Fits the current to double exponential function.

    Parameters
    ----------
    T : nparray
        This is the current data subject to fitting
    I : nparray
        This is the interpolated fitting output provided by the fitter function.

    Returns
    -------
    params : float
        The list of params for the fitter function.
    """
    A0 = I[1]
    B0 = 10
    C0 = I[100]
    D0 = 10
    E0 = I[-1]
    params = scipy.optimize.fmin(
        error_function_double,
        [A0, B0, C0, D0, E0],
        args=(T, I),
        xtol=1e-3,
        maxiter=1500,
    )
    return params


def fit_the_current_single(T, I):
    """
    Fits the current to single exponential function.

    Parameters
    ----------
    T : nparray
        This is the current data subject to fitting
    I : nparray
        This is the interpolated fitting output provided by the fitter function.

    Returns
    -------
    params : float
        The list of params for the fitter function.
    """
    A0 = I[1]
    B0 = 10
    C0 = I[-1]
    params = scipy.optimize.fmin(
        error_function_single, [A0, B0, C0], args=(T, I), xtol=1e-3, maxiter=1500
    )
    return params


def standard_error_estimate(data, fit):
    """
    Returns the standard error estimate of a fitted dataset.

    Parameters
    ----------
    data : nparray
        This is the current data subject to fitting
    fit : nparray
        This is the interpolated fitting output provided by the fitter function, which corrrespond 1 to 1o with the data points.

    Returns
    -------
    output : float
        This is the single float that is the root of the mean squared error between data and fit.
    """
    assert len(data) == len(fit)
    return np.sqrt(((np.linalg.norm(data - fit)) ** 2) / len(fit))


def readDataFilter(metaData, rc):
    """
    reads the data from the files
    returns filtered data
    if fake events are requested additional fake events are added

    Parameters
    ----------
    metadata : list
        contains experiment list and other
    rc : run constants data class
        contains save paht and arguments for the run

    Returns
    -------
    data_set: np.array
        current data in correct unit with the original sampling frequency (untouched)
    data_set_filtered : np.array
        current data in correct unit with filter applied + subsampling
    sampleRateFilt: float
        new sample rate of filtered data
    """
    sampleRate = metaData["SETUP_ADCSAMPLERATE"]

    if rc.args.instrumentType == "Chimera":
        data_set = import_utils.get_data_Chimera(metaData)

    elif rc.args.instrumentType == "smFET":
        data_set = import_utils.get_data_smFET(metaData)

    elif rc.args.instrumentType == "HEKA":
        data_set = import_utils.get_data_HEKA(metaData)

    elif rc.args.instrumentType == "CNP2":
        data_set = import_utils.get_data_CNP2(metaData)

    elif rc.args.instrumentType == "PCKL":
        data_set, data_set_filtered, sampleRateFilt = import_utils.get_data_PCKL(
            metaData
        )

    if rc.args.fakeevents and not (rc.args.instrumentType == "PCKL"):
        data_set = import_utils.fakeEvents(
            data_set,
            sampleRate,
            depth=1000,
            rate=20,
            numberOfLevels=10,
            filterCutoff=rc.args.filter,
            setupParameter=metaData,
        )

    if rc.args.reversepolarity:
        data_set = np.negative(data_set)

    if not rc.args.instrumentType == "PCKL":
        data_set_filtered, sampleRateFilt = applyFilter(data_set, sampleRate, rc)

    return data_set, data_set_filtered, sampleRateFilt


def applyFilter(data_set, sampleRate, rc):
    """
    filters the data according to rc arguments.
    default filtering is lowpass fir
    can do as well notch filter at 60Hz
    can do as well stopband filter (set to 250-600Hz stopband)

    Parameters
    ----------
    data_set: np.array
        current data in correct unit with the original sampling frequency (untouched)
    sampleRate : float
        sample rate of data_set
    rc : run constants data class
        contains save paht and arguments for the run

    Returns
    -------
    data_set_filtered : np.array
        current data in correct unit with filter applied + subsampling
    sampleRateFilt: float

    """
    # lowpass filter
    filterOrder = 5  # BUG for RC?
    dsSize = np.size(data_set)

    if "none" in rc.args.wavelet:
        resample = 4.0
        data_set = filter_utils.lowpass_filter(
            data_set, rc, sampleRate, filterOrder, resample
        )
        sampleRateFilt = sampleRate * np.size(data_set) / dsSize
    else:
        data_set = filter_utils.wavelet_filter(data_set, rc)
        newSampleRate = rc.args.filter * 4
        filteredInt = False
        if rc.args.instrumentType == "Chimera":
            filteredInt = True
        data_set = filter_utils.lowpass_resample(
            data_set, sampleRate, newSampleRate, filteredInt=filteredInt
        )
        sampleRateFilt = sampleRate * np.size(data_set) / dsSize

    # stopband filter
    if rc.args.stopBand:
        stopband = [250, 600]
        # data_set_filtered = filter_utils.bandstop_filter(data_set_filtered, stopband, sampleRateFilt, fType='hamming')
        data_set = filter_utils.bandstop_filter(
            data_set, stopband, sampleRateFilt, fType="kaiser"
        )

    # notch filter
    if rc.args.notch:
        data_set = filter_utils.notch_filter(data_set, 60.0, 10, sampleRateFilt)

    return data_set, sampleRateFilt


def fixed_transitions(metaData, rc):
    """
    return list of floats for fixed slicing of data using rc.args.sliceTime argument on commandline

    # BUG -- this still screws with transitions as datetime vs as ints.

    Parameters
    ----------
    metadata : list
        contains experiment list and other
    rc : run constants data class
        contains save paht and arguments for the run

    Returns
    -------
    transitions: list
        list of np.array (size(1,2)) with floats for the relative timestamp of each slice start and endpoint.
    """
    totalDuration = metaData["TOTALTIME"]  # [s]
    transitions = []
    transitions.append(list(np.array([0, 1]) * rc.args.sliceTime))
    for i in range(1, int(totalDuration // rc.args.sliceTime)):
        transitions.append(list(np.array([i, i + 1]) * rc.args.sliceTime))
    return transitions


def exp_to_slice(experiment):
    """
    Take an experiment, and return it's trivial slice initializer transition values.

    BUG Blagh -- this does IO, and not insignificant IO -- it's not super-visible on the way we do it, but room for improvement.
    """

    file_length = import_utils.file_length_sec(experiment.metaData["DATA_FILES"][0])

    if experiment.metaData["NUMBEROFFILES"] > 1:
        full_files_time = (experiment.metaData["NUMBEROFFILES"] - 1) * file_length
        final_file_time = import_utils.file_length_sec(
            experiment.metaData["DATA_FILES"][1]
        )
    else:
        full_files_time = file_length
        final_file_time = 0

    start_time = experiment.datetime
    end_time = (
        start_time
        + timedelta(seconds=full_files_time)
        + timedelta(seconds=final_file_time)
    )
    return [[start_time, end_time]]


def align_time(benchvue, timestamp, experiment):
    # note that this does not check for it to be the same DATE BUG but this should be easy...
    benchvue_time = benchvue.time.iloc[0]
    benchvue_end = benchvue.time.iloc[-1]
    lag = (benchvue_time - timestamp).total_seconds()
    length = (benchvue_end - timestamp).total_seconds()
    try:
        assert lag > 0, "Start the Chimera first, then start the Synchrodance."
    except AssertionError:
        logging.error("Lag check fail on \n" + experiment.name)
        logging.error("File list was \n" + str(experiment.metaData["DATA_FILES"]))
        raise
    try:
        assert length > 0, "Chimera should run past synchrodance end."
    except AssertionError:
        logging.error("Lag check fail on \n" + experiment.name)
        logging.error("File list was \n" + str(experiment.metaData["DATA_FILES"]))
        raise
    return lag, length


class rc:
    """
    RunConstants -- things that should never change
    """

    def __init__(
        self,
        npersegPARAM,
        noverlapPARAM,
        args,
        front_crop,
        back_crop,
        VA,
        dataPath,
        resultsPath,
        allowed_time_lag,
    ):
        self.npersegPARAM = npersegPARAM
        self.noverlapPARAM = noverlapPARAM
        self.front_crop = timedelta(seconds=front_crop)
        self.back_crop = timedelta(seconds=back_crop)
        self.args = args
        self.VA = VA
        self.dataPath = dataPath
        self.resultsPath = resultsPath
        self.allowed_time_lag = allowed_time_lag

    def __str__(self):
        data = vars(self)
        out = ""
        for datum in data:
            out = out + datum + " " + str(data[datum]) + "\n"
        return out


class Experiment:
    """
    This class represents an single measurement taken with the instrument, over a series of applied voltages and perhaps concatenating a series of individual measurement files.

    Parameters
    ----------

    metaData : list
        metaData for the measurement.
    rc : RunConstant object
        indicating run constants for the experiment.
    """

    def __init__(self, metaData, rc):
        self.metaData = metaData
        self.rc = rc
        self.name = metaData["NAME"]
        self.SETUP_ADCSAMPLERATE = metaData["SETUP_ADCSAMPLERATE"]
        self.datetime = metaData["mytimestamp"]
        self.VAC = metaData["VAC"]
        if isinstance(metaData["BENCHVUE"], Path):
            benchvue = import_utils.handle_benchvue_import(metaData["BENCHVUE"])
        else:
            benchvue = pd.DataFrame(
                columns=["time", "voltage", "current"], index=["none"]
            )
            benchvue.loc["none"] = pd.Series(
                ({"time": self.datetime, "voltage": 0.0, "current": 0.0})
            )

        self.benchvue = benchvue
        self.Ia = np.nan
        self.IaRAW = np.nan

        if "VG" in metaData.keys():
            """
            OK, a note here on what V_AG and all that is, and why it may be counter-intuitive.

            Typically, folks refer everything to the source, like V_ds, and V_gs.  This is consistent with majority carriers entering the *device* at the source and flowing to the drain (i.e -- the thing being sourced into the device at the source is the majority carriers, and the thing being drained out of the device at the drain is the majority carriers), and the usual Franklin current convention (positive carriers moving down a voltage produce a positive current).

            We assume we don't have majority carriers that dominate (for KCl, we neglect that the mobilities of K and Cl differ by about 2%) and refer to the electrodes by their effect on the current *though the device* -- current enters the device at the anode, and exits the device at the cathode, meaning that for R > 0, V_AC > 0.

            However, the Chimera programming relies on the two-terminal nanopore convention, in which DNA events (negative moving charge) yields a current blockage (lower value) -- therefore, the Chimera convention is that the anode is the terminal where the current *enters* the *measurement device (the headstage)* and the cathode is where the current *leaves the headstage*.  This is the cause of the minus sign in the Chimera programming, and why we reference everything to the anode.  This also relies on the anode being in the front -- top well -- of the experiment, but explais why you're used to seeding V_gs and V_ds, and should now start looking at V_ac and V_ag.

            What's slightly less annoying and actually convenient is that V_AC = V_AG + V_GC, which is the two potential drops a (negtively charged!) DNA will experience going across a single-gate device.

            """

            self.VAG = metaData["VG"]
            self.gate_current = np.nan
        elif isinstance(metaData["BENCHVUE"], Path):
            self.VAG = rc.VA - benchvue.voltage.to_numpy()
            self.gate_current = benchvue.current.to_numpy()
        else:
            self.VAG = np.nan
            self.gate_current = np.nan

        self.gate_time = benchvue.time

        if rc.args.instrumentType == "Chimera":
            self.unit = "pA"
        elif rc.args.instrumentType == "smFET":
            self.unit = "nA"
        elif rc.args.instrumentType == "HEKA":
            self.unit = "pA"
        elif rc.args.instrumentType == "CNP2":
            self.unit = "pA"
        elif rc.args.instrumentType == "PCKL":
            self.unit = metaData["unit"]

    def load(self):
        """
        Loads the current data.

        Parameters
        ----------
        None.

        Returns
        -------
        self.IaRAW : float
                Raw current data
        self.Ia : float
                Filtered and interpolated current data
        """
        if not np.isnan(self.Ia).all():
            logging.warning("You asked me to load a dataset that was already loaded!")
            return

        data_set, data_set_filtered, sampleRateFilt = readDataFilter(
            self.metaData, self.rc
        )
        self.filtersamplerate = sampleRateFilt
        # OK, here's the problem.  By convention, the benchvue is started AFTER the chimera.  Therefore, we have to crop Chimera points here to align the two.  The number of seconds of points to crop is calculated here:
        if isinstance(self.metaData["BENCHVUE"], Path):
            benchvue = import_utils.handle_benchvue_import(self.metaData["BENCHVUE"])
            lag, length = align_time(benchvue, self.datetime, self)
            start_point = math.ceil(lag * self.metaData["SETUP_ADCSAMPLERATE"])
            end_point = math.floor(length * self.metaData["SETUP_ADCSAMPLERATE"])
            self.IaRAW = data_set[start_point:end_point]
            start_point = math.ceil(lag * self.filtersamplerate)
            end_point = math.floor(length * self.filtersamplerate)
            self.Ia = data_set_filtered[start_point:end_point]
        else:
            self.IaRAW = data_set
            self.Ia = data_set_filtered
        # del data_set
        gc.collect()
        return

    def unload(self):
        """
        Unloads the current data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        del self.IaRAW
        del self.Ia
        self.Ia = np.nan
        self.IaRAW = np.nan
        gc.collect()
        return

    def schneiden(self):  # slice is apparently a reserved word, whatever....
        """
        Cuts up experiment into slices, according to the function chosen below.

        Parameters
        ----------
        None.

        Returns
        -------
        slices : list of Slice() objects
                Slice() array coming off this Experiment.
        """
        if "_Gate_" in self.name:
            # We currently don't have a way of handling slices for _IV_ files in smFET data, to treat like a trivial slice.
            transitions = exp_to_slice(self)
        elif self.rc.args.slicefromfile:
            logging.info("Going to go slice this one by file.")
            logFilePath = self.metaData["DATA_FILES"][0].parent
            file_name = self.name + ".txt"
            txt_file = list(logFilePath.rglob(file_name))
            if len(txt_file) == 0:  ## if no slice file, just run with trivial slice.
                transitions = exp_to_slice(self)
            else:
                assert len(txt_file) == 1, "One slice file per Experiment."
                transitions = []
                with open(txt_file[0], newline="") as txt_file:
                    txtreader = csv.reader(txt_file, delimiter="\n")
                    for row in txtreader:
                        assert float(row[0].split(",")[0]) < float(row[0].split(",")[1])
                        assert float(row[0].split(",")[1]) <= self.metaData["TOTALTIME"]
                        start = self.datetime + timedelta(
                            seconds=float(row[0].split(",")[0])
                        )
                        assert start >= self.datetime
                        end = self.datetime + timedelta(
                            seconds=float(row[0].split(",")[1])
                        )
                        transitions.append([start, end])
        elif isinstance(self.metaData["BENCHVUE"], Path):
            benchvue = import_utils.handle_benchvue_import(self.metaData["BENCHVUE"])
            if benchvue.size > 3:
                transitions = import_utils.pull_transitions_from_benchvue(
                    benchvue, self.rc
                )
        #                plot_utils.plot_benchvue(self, benchvue, transitions, self.rc)
        # The above is a nice plotting function, and can be readily uncommented -- however, all the output is now reproduced in plot_experiment, and flipping through it all is annoying.
        elif not math.isnan(self.rc.args.sliceTime):
            transitions = fixed_transitions(self.metaData, self.rc)
        else:
            ## OK, now your only option is the trivial slice.
            transitions = exp_to_slice(self)

        slices = []
        assert len(transitions) > 0

        if self.rc.args.instrumentType == "PCKL":
            assert len(transitions) == 1
            seg = Slice(self, transitions[0], self.metaData["slice_i"], self.rc)
            slices.append(seg)
        else:
            for i, t in enumerate(transitions):
                seg = Slice(self, t, i, self.rc)
                slices.append(seg)
        assert len(slices) > 0, "You did slice something, right?"
        return slices


class Slice:
    """
    Some description TODO.

    Parameters
    ----------
    experiment : Experiment()
        The parent Experiment() to this guy.
    transition : two-list of datetime
        This is the timestamp of the transition, in datetime objects.
    index : int
        The ordered index of the Slice relative to other slices in the parent.
    rc : rc()
        Run-constants.

    Methods
    -------
    init_statsDF:
        fix TODO
    load:
        fix TODO
    fit_transform
    """

    def __init__(self, experiment, transition, index, rc):
        """
        Constructor to initialize a slice.
        """
        self.name = experiment.name + "_" + str(index)
        self.exp = experiment
        self.sample = self.exp.metaData["SAMPLE"]
        self.COMMENTS = self.exp.metaData["COMMENTS"]
        self.transition = transition
        self.i = index
        if not np.isnan(self.exp.rc.args.sliceTime):
            self.datetime = experiment.datetime + timedelta(0, self.exp.rc.args.sliceTime * self.i)
        else:
            self.datetime = experiment.datetime
        self.date = self.datetime.date()
        self.time = self.datetime .time()
        self.rc = rc
        self.VA = rc.VA
        self.VAC = self.exp.VAC
        self.baseline = np.nan
        if not np.isnan(self.exp.VAG).all():
            self.VAG = np.nanmean(self.exp.VAG)
            self.VGC = self.VAC - self.VAG
        else:
            self.VAG = np.nan
            self.VGC = np.nan

        self.salt = self.exp.metaData["electrolyte"]
        self.conc = self.exp.metaData["conc"]
        self.statsDF = self.init_statsDF()
        self.unit = experiment.unit
        self.samplerate = experiment.SETUP_ADCSAMPLERATE

        # Populate states for fit / non fit compatibility
        self.fit_current = np.nan
        self.fit_error = np.nan
        self.fit_current_conv = np.nan
        self.fit_prefactorA = np.nan
        self.fit_prefactorC = np.nan
        self.fit_decay_time1 = np.nan
        self.fit_decay_time2 = np.nan
        self.fit_gate_current = np.nan
        self.fit_gate_error = np.nan
        self.fit_gate_decay_time = np.nan
        self.fit_gate_current_conv = np.nan

    def init_statsDF(self):
        index = [self.dfIndex()]
        theCols = [
            "exp_name",
            "timestamp",
            "slice_i",
            "SAMPLE",
            "COMMENTS",
            "conc",
            "VA",
            "electrolyte",
            "VAC",
            "VAG",
            "CHANNEL",
            "I_gate",
            "I_mean",
            "I_std",
            "fit_cur",
            "fit_time1",
            "fit_time2",
            "fit_error",
            "gate_fit_cur",
            "gate_fit_time",
            "gate_fit_error",
            "CCHIP",
            "RCHIP",
            "I2NOISE-1",
            "I2NOISE0",
            "I2NOISE1",
            "I2NOISE2",
            "unit",
        ]
        return pd.DataFrame(index=index, columns=theCols)

    def write_setupParameter_to_statsDF(self):
        excludeKeys = "SETUP_"
        for key in self.exp.metaData:
            if (
                # BUG -- what are these?
                not key.startswith("SETUP")
                and not key.startswith("DATA_")
                and not key.startswith("NUMBERO")
                and not key.startswith("NAME")
                and not key.startswith("myt")
            ):
                # BUG -- what are we looking for here?
                self.add_to_statsDF(key, self.exp.metaData[key])

    def dfIndex(self):
        return "{}-{}_{}_{}mV".format(
            self.date.strftime("%Y%m%d"),
            self.time.strftime("%H%M%S"),
            self.name,
            self.VAC,
        )

    def add_to_statsDF(self, key, value):
        # tmpDF = self.statsDF
        index = [self.dfIndex()]
        if isinstance(key, list):
            tmp = zip(key, value)
            for i, v in enumerate(tmp):
                self.statsDF.at[index[0], v[0]] = v[1]
        else:
            self.statsDF.at[index[0], key] = value

    def join_statsDF(self, other, rsuffix=""):
        """
        joins other dataframe to self.statsDF
        use rsuffix if colum is already existing.

        Parameters:
        ------------

            self : data class object slice
            other : pd dataframe
                with seg.dfIndex as index and columns only 1D possible
            rsuffix : str

        """
        self.statsDF = self.statsDF.join(other, rsuffix=rsuffix)

    def make_fit_current(self):
        time = np.arange(0, np.size(self.Ia), 1, dtype=np.float32) / self.samplerate
        # BUG justification for the disgusting hack below: when the current is on the rails, the exponential fitting algorithm does not converge.
        fp = [0]
        if self.cur_mean > -16000 and self.cur_mean < 16000:
            # If the current is on the rails, this will only fit the first 50 points throughout
            fp = fit_the_current_single(time[::50], self.Ia[::50])
        if len(fp) == 5:
            self.fit_current = fit_target_func_double(
                time, fp[0], fp[1], fp[2], fp[3], fp[4]
            )
            self.fit_error = standard_error_estimate(
                self.Ia[10::50], self.fit_current[10::50]
            )
            self.fit_current_conv = fp[4]
            self.fit_prefactorA = fp[0]
            self.fit_prefactorC = fp[2]
            self.fit_decay_time1 = fp[1]
            self.fit_decay_time2 = fp[3]
        elif len(fp) == 3:
            self.fit_current = fit_target_func_single(time, fp[0], fp[1], fp[2])
            self.fit_error = standard_error_estimate(
                self.Ia[10::50], self.fit_current[10::50]
            )
            self.fit_current_conv = fp[2]
            self.fit_prefactorA = fp[0]
            self.fit_prefactorC = np.nan
            self.fit_decay_time1 = fp[1]
            self.fit_decay_time2 = np.nan
        else:
            self.fit_current = np.nan
            self.fit_error = np.nan
            self.fit_current_conv = np.nan
            self.fit_prefactorA = np.nan
            self.fit_prefactorC = np.nan
            self.fit_decay_time1 = np.nan
            self.fit_decay_time2 = np.nan
        data = [
            self.fit_current_conv,
            self.fit_decay_time1,
            self.fit_decay_time2,
            self.fit_error,
        ]
        keys = ["fit_cur", "fit_time1", "fit_time2", "fit_error"]
        self.add_to_statsDF(keys, data)

    def psd(self):
        nperseg = int(np.size(self.Ia) // self.rc.npersegPARAM)
        assert self.rc.noverlapPARAM > 1
        noverlap = nperseg // self.rc.noverlapPARAM
        psd_freq, psd_cur = fft_utils.fft_welch(
            self.Ia, self.samplerate, nperseg, noverlap
        )
        self.psd_cur = psd_cur
        self.psd_freq = psd_freq

    def unload_psd(self):
        """
        Unloads the PSD.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        del self.psd_cur
        del self.psd_freq
        gc.collect()

    def unload(self):
        """
        Unloads the current data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        del self.Ia
        del self.gate_current
        del self.VAG
        del self.gate_time
        del self.exp
        del self.IaRAW
        del self.baseline
        gc.collect()

    def load(self):
        """
        Loads the data for the Slice().  Loads the slice VA.  BUG -- entire thing should be re-written to just use transitions in the time domain, and not as ints.

        Parameters
        ----------
        None.

        Returns
        -------
        self.Ia : nparray
            The data points corresponding to this gate current segment.
        self.Ia_raw : nparray
            Same as the above, but unfiltered.
        """
        # Well, this requires the transitions to be [int, int] or [datetime, datetime] -- BUG we should see which way to make this
        assert type(self.transition) == list
        assert len(self.transition) == 2
        print(
            "STILL A BUG type of self.transition for experiment {s1}: {s2}".format(
                s1=self.exp.name, s2=str(type(self.transition[0]))
            )
        )
        self.add_to_statsDF(["VA"], [self.VA])
        ## BUG the below fail sometimes on something relating to Pandas dateimes and datetime.datetimes not being the same thing.
        # assert type(self.transition[0]) == int or type(self.transition[0]) == datetime
        # assert type(self.transition[1]) == int or type(self.transition[1]) == datetime

        if self.rc.args.instrumentType != "PCKL":
            if isinstance(
                self.exp.metaData["BENCHVUE"], Path
            ):  # if there's a benchvue to look at, even if it specs a single slice.
                cropped_bench = self.get_cropped_bench()
                self.load_VAG(cropped_bench)
                self.load_Ig(cropped_bench)
                self.load_Tg(cropped_bench)
                self.load_Ia()

            # TODO This is a mess -- why are we doing the below?  We should generalize here to have a correct self.transition, and then go from there.  This also breaks when transitions are datetimes, as schniden returns them.  When did this break?

            elif not math.isnan(self.rc.args.sliceTime):
                experiment = self.exp
                experiment.load()
                self.samplerate = experiment.filtersamplerate
                self.samplerateRAW = experiment.SETUP_ADCSAMPLERATE

                # get relevant slice from experiment
                startIdx = np.int(np.floor(self.transition[0] * self.samplerate))
                endIdx = np.int(np.floor(self.transition[1] * self.samplerate))
                # if transition is correct, both of these will never be true, so use an assert here...
                if endIdx > len(experiment.Ia):
                    endIdx = len(experiment.Ia) - 1
                if startIdx > len(experiment.Ia) - 2:
                    startIdx = len(experiment.Ia) - 1000

                self.Ia = experiment.Ia[startIdx:endIdx]

                startIdx = np.int(np.floor(self.transition[0] * self.samplerateRAW))
                endIdx = np.int(np.ceil(self.transition[1] * self.samplerateRAW))

                if endIdx > len(experiment.IaRAW):
                    endIdx = len(experiment.IaRAW) - 1
                if startIdx > len(experiment.IaRAW) - 2:
                    startIdx = (
                        len(experiment.IaRAW) - 1000
                    )  # BUG -- well, it's not the file then...

                self.IaRAW = experiment.IaRAW[
                    startIdx:endIdx
                ]  # BUG?  having a benchvue below makes things consistant, but a little ugly sometime
                self.gate_time = self.datetime
                self.gate_current = (
                    np.nan
                )  # TODO Dude, this destroys data!  If you feed this a slice that has a gate voltage, it'll clear it out.
                self.add_to_statsDF("sliceTime", self.rc.args.sliceTime)
                experiment.unload()

            elif (
                self.rc.args.slicefromfile
            ):  # This is a skeleton of what this might want to look like if we just do transitions.
                experiment = self.exp
                experiment.load()

                self.samplerate = experiment.filtersamplerate

                # get relevant slice from experiment
                start_time = (self.transition[0] - self.exp.datetime).total_seconds()
                end_time = (self.transition[1] - self.exp.datetime).total_seconds()
                assert start_time < end_time
                assert start_time >= 0
                startIdx = np.int(np.floor(start_time * self.samplerate))
                endIdx = np.int(np.floor(end_time * self.samplerate))
                self.Ia = experiment.Ia[startIdx:endIdx]

                self.samplerateRAW = experiment.SETUP_ADCSAMPLERATE
                startIdx = np.int(np.floor(start_time * self.samplerateRAW))
                endIdx = np.int(np.floor(end_time * self.samplerateRAW))

                self.IaRAW = experiment.IaRAW[startIdx:endIdx]

                ## below will never execute because of if true than this will be not entered...
                assert (
                    experiment.rc.args.instrumentType == "Chimera"
                    or experiment.rc.args.instrumentType == "smFET"
                )
                self.gate_current = np.nan
                self.VAG = np.nan
                self.gate_time = self.datetime
                experiment.unload()

            else:
                experiment = self.exp
                experiment.load()
                self.Ia = experiment.Ia
                self.IaRAW = experiment.IaRAW
                self.samplerate = experiment.filtersamplerate
                self.samplerateRAW = experiment.SETUP_ADCSAMPLERATE
                self.gate_time = self.datetime
                self.gate_current = np.nan

        else:
            loadedSeg = import_utils.get_segment_PCKL(self.exp.metaData)
            self.Ia = loadedSeg.Ia
            self.IaRAW = (
                loadedSeg.IaRAW
            )  # BUG?  having a benchvue below makes things consistant, but a little ugly sometime
            self.samplerate = loadedSeg.samplerate
            self.samplerateRAW = loadedSeg.samplerateRAW
            self.VAG = loadedSeg.VAG
            self.gate_time = loadedSeg.gate_time
            self.gate_current = loadedSeg.gate_current

        self.cur_mean = np.mean(self.Ia)
        self.cur_std = np.std(self.Ia)
        self.write_setupParameter_to_statsDF()
        if not np.isnan(self.VAG).all():
            self.VGC = self.VAC - np.nanmean(self.VAG)
            data = [
                self.exp.name,
                self.datetime,
                self.i,
                self.VAC,
                np.nanmean(self.VAG),
                self.VGC,
                np.nanmean(self.gate_current),
                self.cur_mean,
                self.cur_std,
                self.unit,
            ]
        else:
            data = [
                self.exp.name,
                self.datetime,
                self.i,
                self.VAC,
                self.VAG,
                self.VGC,
                self.gate_current,
                self.cur_mean,
                self.cur_std,
                self.unit,
            ]
        keys = [
            "exp_name",
            "timestamp",
            "slice_i",
            "VAC",
            "VAG",
            "VGC",
            "I_gate",
            "I_mean",
            "I_std",
            "unit",
        ]
        self.add_to_statsDF(keys, data)
        return

    def load_Ia(self):
        """
        Loads the anode current data.

        Parameters
        ----------
        None.

        Returns
        -------
        self.Ia : nparray
                The data points corresponding to this gate current segment.
        self.Ia_raw : nparray
                Same as the above, but unfiltered.

        """
        # now you know this is multiple files...
        keepers = []
        prev_file = ""
        for fileN in self.exp.metaData["DATA_FILES"]:
            file_time_start = import_utils.read_mat_Chimera(fileN)["mytimestamp"]
            file_length = import_utils.file_length_sec(fileN)
            if file_time_start > self.transition[0]:
                # entered stuff to add to keepers...
                if prev_file != "":  # if this is the first one, add the prev one...
                    keepers.append(prev_file)
                    prev_file = ""
                # you are right of the first transition, add it
                keepers.append(fileN)
                # are you right of the second transition?
                if file_time_start > self.transition[1]:
                    keepers.pop()
            else:  # before start of ones to add
                prev_file = fileN
        assert len(keepers) > 0
        assert (
            import_utils.read_mat_Chimera(keepers[0])["mytimestamp"]
            < self.transition[0]
        ), ("Shenanigans on start for + \n" + str(keepers[0]) + str(self.transition[0]))
        assert (
            import_utils.read_mat_Chimera(keepers[-1])["mytimestamp"]
            <= self.transition[1]
        ), (
            "Shenanigans on end for + \n"
            + str(keepers[0])
            + str(import_utils.read_mat_Chimera(keepers[-1])["mytimestamp"])
            + " "
            + str(self.transition[1])
        )
        assert (
            import_utils.read_mat_Chimera(keepers[-1])["mytimestamp"]
            + timedelta(seconds=file_length)
            >= self.transition[1]
        ), ("Shenanigans + \n" + str(keepers[0]) + str(self.transition))
        localData = copy.deepcopy(self.exp.metaData)
        localData.update({"NUMBEROFFILES": len(keepers)})
        localData.update({"DATA_FILES": keepers})
        data_set, data_set_filtered, sampleRateFilt = readDataFilter(
            localData, self.exp.rc
        )
        self.samplerate = sampleRateFilt
        # OK, this gets you all the data from all the files that you have pulled, but this is too much, on both ends.
        # How much do you crop off?
        first_file_time = import_utils.read_mat_Chimera(keepers[0])["mytimestamp"]
        lag_front = self.transition[0] - first_file_time
        assert lag_front.total_seconds() > 0
        crop_front = self.samplerate * (lag_front.total_seconds())

        last_file_points = import_utils.read_binary_Chimera(
            keepers[-1], self.exp.metaData
        )
        last_file_sec = (
            len(last_file_points) / self.exp.metaData["SETUP_ADCSAMPLERATE"]
        )  # all match to like 7 decimal places
        last_file_time = import_utils.read_mat_Chimera(keepers[-1])["mytimestamp"]
        assert last_file_time + timedelta(seconds=file_length) > self.transition[1]
        assert last_file_time < self.transition[1]
        assert (self.transition[1] - last_file_time).total_seconds() < file_length
        last_file_end = last_file_time + timedelta(seconds=last_file_sec)
        lag_back = self.transition[1] - last_file_end
        assert lag_back.total_seconds() < 0
        crop_back = -self.samplerate * lag_back.total_seconds()

        self.Ia = data_set_filtered[
            math.ceil(crop_front) : math.floor(len(data_set_filtered) - crop_back)
        ]

        # Same story, but with the correct sample frequencies for the raw data
        self.samplerateRAW = self.exp.metaData["SETUP_ADCSAMPLERATE"]
        first_file_time = import_utils.read_mat_Chimera(keepers[0])["mytimestamp"]
        lag_front = self.transition[0] - first_file_time
        assert lag_front.total_seconds() > 0
        crop_front = self.samplerateRAW * (lag_front.total_seconds())

        last_file_points = import_utils.read_binary_Chimera(
            keepers[-1], self.exp.metaData
        )
        last_file_sec = (
            len(last_file_points) / self.samplerateRAW
        )  # all match to like 7 decimal places
        last_file_time = import_utils.read_mat_Chimera(keepers[-1])["mytimestamp"]
        assert last_file_time + timedelta(seconds=file_length) > self.transition[1]
        assert last_file_time < self.transition[1]
        assert (self.transition[1] - last_file_time).total_seconds() < file_length
        last_file_end = last_file_time + timedelta(seconds=last_file_sec)
        lag_back = self.transition[1] - last_file_end
        assert lag_back.total_seconds() < 0
        crop_back = -self.samplerateRAW * lag_back.total_seconds()
        self.IaRAW = data_set[
            math.ceil(crop_front) : math.floor(len(data_set) - crop_back)
        ]

    def get_cropped_bench(self):
        """
        Get the benchvue files.  "get" not "load" since it just gets the file without modifying a self.cropped_bench variable

        Parameters
        ----------
        None.

        Returns
        -------
        out : nparray
                Returns the sections of the benchvue file that correspond to this segment.
        """
        benchvue = self.exp.benchvue
        return benchvue.loc[
            (benchvue.time >= self.transition[0])
            & (benchvue.time <= self.transition[1])
        ]

    def load_VAG(self, cropped_bench):
        """
        Loads the gate voltage data.

        Parameters
        ----------
        cropped_bench : nparray
                The gate time data points

        Returns
        -------
        self.VAG : nparray
                The voltage data points taken by the SR570.
        """
        self.VAG = self.VA - cropped_bench.voltage.to_numpy()

    def load_Ig(self, cropped_bench):
        """
        Loads the gate current data.

        Parameters
        ----------
        cropped_bench : nparray
                The gate current data pointss

        Returns
        -------
        self.gate_current : nparray
                The current data points taken by the SR570.
        """
        self.gate_current = cropped_bench.current.to_numpy()

    def load_Tg(self, cropped_bench):
        """
        Loads the gate time data.

        Parameters
        ----------
        cropped_bench : nparray
                The gate time data points

        Returns
        -------
        self.gate_time : nparray
                The time data points taken by the SR570.
        """
        self.gate_time = cropped_bench.time

    def make_fit_gate_current(self):
        # BUG -- has to be a better way of doing this, but for 100 points, it scales...
        if np.isnan(np.array(self.gate_current, dtype=np.float32)).any():
            self.fit_gate_current = np.nan
            self.fit_gate_error = np.nan
            self.fit_gate_decay_time = np.nan
            self.fit_gate_current_conv = np.nan
        else:
            intervals = []
            start = self.gate_time.iloc[0]
            for i in self.gate_time:
                change = i - start
                intervals.append(change.total_seconds())
                start = i
            time = np.cumsum(intervals)
            current = []
            for i in self.gate_current:
                current.append(i)
            fit_params = fit_the_current_single(time, current)
            self.fit_gate_current = fit_target_func_single(
                time, fit_params[0], fit_params[1], fit_params[2]
            )
            self.fit_gate_error = standard_error_estimate(
                current, self.fit_gate_current
            )
            self.fit_gate_decay_time = fit_params[1]
            self.fit_gate_current_conv = fit_params[2]
        data = [self.fit_gate_decay_time, self.fit_gate_error]
        keys = ["gate_fit_time", "gate_fit_error"]
        self.add_to_statsDF(keys, data)
        self.add_to_statsDF("gate_fit_cur", self.fit_gate_current_conv)
        # separate from others to be able to write np array to one entry
        return

    def hmm_build_and_fit(self):
        """
        builds a hiddem Markov modeling of data,
        depending on rc.args.hmm if 'two' - two state model, if 'hidden' - three state model with one hidden state

        fits hiddem Markov modeling to data with Baum-Welch algorithm
        Parameters
        ----------
        self :
            hmmModel : pomegranate model isinstance

        Returns
        -------
            self.hmmTrace : nparray
            self.hmmState : nparray
            self.hmmLogTransProb : np.array 
                with states x states size
            self.modelDict :
                dict to match state to description and transition labels


        """

        (
            self.hmmTrace,
            self.hmmState,
            self.hmmLogTransProb,
            self.modelDict,
            self.baselineCorrectedData,
            self.hmmModel,
        ) = hmm_utils.build_and_fit_hmm_model(self, self.rc)
        return

    def hmm_analyze(self):
        """
        Analyzes HMM
        computes k_values from transtion probability matrix: self.hmmLogTransProb
        calculates dwell time of each state
        plots ideal trace of fitted HMM
        saves hmm model data to csv file
        saves k values of fit and model to statsDF
        plots histogram of all state dwell times
        
        output
        ----------
        * plots of dwell time fits
        * csv of all states of HMM with kvalue fits

        Parameters
        ----------
            self :


        Returns
        -------
            none
            
        """

        tmpRc = self.rc
        filterCutoff = np.copy(self.rc.args.filter)
        tmpRc.resultsPath = self.rc.resultsPath / "hmm"
        tmpRc.resultsPath.mkdir(parents=True, exist_ok=True)

        # calculate dwell times from trace and save to dataframe
        hmmEventsDF = hmm_utils.get_dwell_times(self, tmpRc)

        # calculate transitions and update dataframe
        hmmEventsDF = hmm_utils.get_hmm_transitions(self, tmpRc, hmmEventsDF)

        # get HMM type and update dataframe
        hmmEventsDF = hmm_utils.get_hmm_type(self, tmpRc, hmmEventsDF)
        hmmEventsDF = hmm_utils.get_hmm_path(self, tmpRc, hmmEventsDF)
        hmmEventsDF = hmm_utils.get_state_probability(self, tmpRc, hmmEventsDF)

        # do k value fitting on trace
        tmpRc.args.filter = filterCutoff
        hmmEventsDF = hmm_utils.state_dwell_time_fit_wrapper(self, tmpRc, hmmEventsDF)

        # plot histogram of dwell times
        hmm_utils.plot_hmm_state_dwellHistogram(self, tmpRc, hmmEventsDF)

        # save dataframe to disk
        hmm_utils.save_hmm_df(self, tmpRc, hmmEventsDF)

        # plot hmm trace and data
        if math.isnan(tmpRc.args.plotidealevents):
            tmpRc.args.plotidealevents = 5.0
        hmm_utils.plot_hmm_trace(self, tmpRc)

        # save stats of hmm to statsDF
        self.hmm_hmmEventsDF_to_statsDF(hmmEventsDF)

        del hmmEventsDF

        # reset path
        self.rc.resultsPath = tmpRc.resultsPath.parent
        return

    def hmm_hmmEventsDF_to_statsDF(self, hmmEventsDF):
        """
        populates statsDF with k values from hidden markov model

        output
        ----------
        * statsDF gets more columns

        Parameters
        ----------
            self :
            hmmEventsDF : pd DataFrame
                contains dwell time of each state and k values for all transitions

        Returns
        -------
            none

        """
        # remove unrequired data

        columns = [
            "STATELABEL",
            "STATESTARTIDX",
            "STATESTOPIDX",
            "STATEDWELLTIME",
            "TRANSITIONLABEL",
            "STATELEVEL",
            "level_0",
        ]
        tmpDF = hmmEventsDF.drop(columns, axis=1, inplace=False)
        tmpDF = tmpDF.loc[0, :].to_frame().T
        tmpDF["index"] = [self.dfIndex()]
        tmpDF = tmpDF.set_index("index")
        self.join_statsDF(tmpDF)
        del tmpDF

        return

    def hmm_unload(self):
        """
        unloads self.hmmTrace, self.hmmState and self.hmmLogTransProb

        Parameters
        ----------
            self :

        """
        del self.hmmTrace
        del self.hmmState
        del self.hmmLogTransProb
        gc.collect()

        return

    def hmm_simulate_save(self):
        """
        simulates a fitted hmm model for the same time and samplerate as seg.
        writes trace information and model json to folder.
        
        Parameters
        ----------
            self : slice object

        Returns
        --------
            none

        """

        hmm_utils.save_hmm_model(self, self.rc, self.hmmModel)

        simulatedTrace = hmm_utils.simulate_hmm_model(self, self.rc, self.hmmModel)

        hmm_utils.save_simulated_hmm_trace(self, self.rc, simulatedTrace)

        del simulatedTrace

        return

    def hmm_normalized_save(self):
        """
        simulates a fitted hmm model for the same time and samplerate as seg.
        writes trace information and model json to folder.
        
        Parameters
        ----------
            self : slice object

        Returns
        --------
            none

        """

        hmm_utils.save_normalized_trace(self, self.rc)

        return

    def hmm_read_simulate_save(self):
        """
        reads hmm model from disk, simulates the imported hmm model for the same time and samplerate as seg. 
        writes trace information and model json to folder.
        
        Parameters
        ----------
            self : slice object

        Returns
        --------
            none

        """

        self.hmmModel = hmm_utils.read_hmm_model(self, self.rc)

        hmm_utils.save_hmm_model(self, self.rc, self.hmmModel)

        simulatedTrace = hmm_utils.simulate_hmm_model(self, self.rc, self.hmmModel)

        hmm_utils.save_simulated_hmm_trace(self, self.rc, simulatedTrace)

        del simulatedTrace

        return

    def correct_baseline(self):
        """
        will correct baseline drift and fluctuations
        
        Parameters
        ----------
            self : slice object

        Returns
        --------
            none

        """

        if not np.isnan(self.rc.args.blcorrectionfreq):
            logging.info("event finder based baseline correction for {s0}_{s1:.0f}mV".format(s0=self.name, s1=self.VAC))
            self.Ia, self.baseline = baseline_utils.event_finder_baseline_correction(self, self.rc)

        if not len(self.rc.args.blcorrectionwhittaker) == 0 and np.isnan(self.rc.args.blcorrectionfreq):
            logging.info("whittaker baseline correction for {s0}_{s1:.0f}mV".format(s0=self.name, s1=self.VAC))
            self.Ia, self.baseline = baseline_utils.whittaker_baseline_correction(self, self.rc)

        return

    def save_preprocessed_data(self):

        preprocessPath = self.exp.metaData['DATA_FILES'][0].parent / "preprocessedData"
        if self.exp.metaData['DATA_FILES'][0].parent.name == "preprocessedData":
            preprocessPath = preprocessPath.parent
            
        preprocessPath.mkdir(parents=True, exist_ok = True)

        logging.info("saving preprocessed data trace to original data folder for {s0}_{s1:.0f}mV".format(s0=self.name, s1=self.VAC))
        self.IaRAW = None

        # del self.IaRaw
        export_utils.save_segment(self, preprocessPath)
        
        rc_name = (
            "runConstants_atruntime_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
        rcFile =  preprocessPath / rc_name
        with open(rcFile, "w") as file_h:
            writer = csv.writer(file_h)
            writer.writerow([str(self.rc)])

        return


