import os
import numpy as np
from pathlib import Path
import math
import random
import logging

logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s %(levelname)s - %(message)s"
)
import pandas as pd
from datetime import datetime
from datetime import timedelta
import scipy.io as sio
import scipy.signal as ssi
import sys
import gc
import csv
import re
import pickle

sys.path.append(str(Path.cwd() / "src" / "utils"))


def get_stem(experiment):
    """
    Given an Experiment, returns the stem of the files.

    Parameters
    ----------
    experiment : Experiment()
        The experiment to consider.

    Returns
    -------
    stem : str
        The intended base for the Chimera-type experiment filenames.
    """
    the_date = experiment.datetime.date().strftime("%Y%m%d")
    the_time = experiment.datetime.time().strftime("%H%M%S")
    stem = str(the_date) + "_" + str(the_time) + "_" + experiment.name
    return stem


def check_dataPath(resultsPath):
    """
    Check whether data file is available and has no blank line.

    Parameters
    ----------
    resultsPath : Path()
        The path to look at.

    Returns
    -------
    out : Bool
        True if files are in there, else False.
    """
    t = sorted(resultsPath.rglob("data.txt"))
    if not t:
        return False
    else:
        return True


## Read file specifying data folder
def get_dataPath(resultsPath):
    """
    Reads data path from data.txt

    Parameters
    ----------
    resultsPath : Path()
        The path to look at.

    Returns
    -------
    dataPath : Path()
        The dataPath in there.
    """
    dataPath = []
    for fileN in sorted(resultsPath.rglob("data.txt")):
        with open(fileN) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if not len(row) == 0:  # if the line is all whitespace, ignore it.
                    dataPath.append(Path(row[0]))
    return dataPath


def create_experiment_list_smFET(rc):
    """
    Returns a list of concatentated experiments on the smFET runs, each of which is to make an Experiment object.

    This now works by walking all the way down to the root of the directory structure here, and then assuming that there's exactly one params file in the end of the directory structure.

    Still an issue with parsing IV-type data; for now, this is RT only BUG.

    Parameters
    ----------
    rc : runConstants
        run constants

    Returns
    -------
    experimentList : list
        List of Experiment() objects.
    """
    logFiles = []
    if isinstance(rc.dataPath, list):
        preLogFilePath = rc.dataPath[0]
        if rc.args.channel != []:
            for j in rc.args.channel:
                for i in range(len(rc.dataPath)):
                    channelInterest = "*_Chan_{s1}*.bin".format(s1=j)
                    logFiles.extend(sorted(rc.dataPath[i].rglob(channelInterest)))
        else:
            for i in range(len(rc.dataPath)):
                logFiles.extend(sorted(rc.dataPath[i].rglob("*.bin")))
    else:
        preLogFilePath = rc.dataPath
        if rc.args.channel != []:
            for j in rc.args.channel:
                channelInterest = "*_Chan_{s1}*.bin".format(s1=j)
                logFiles.extend(sorted(rc.dataPath.rglob(channelInterest)))
        else:
            logFiles.extend(sorted(rc.dataPath.rglob("*.bin")))

    numberOfFiles = len(logFiles)
    assert (
        numberOfFiles >= 1
    ), "You were looking for smFET data, and found no smFET files.  Check the data directory, and check that you're passing the right tool."
    i = 0
    counter = 0
    experimentList = {}
    file_list = []

    for fileN in logFiles:
        if not fileN.stem.split("_")[-1] == "Gate" and not "_IV_" in str(fileN.stem):
            acquisitionParameter = read_params_smFET(fileN)
            name_handle = "_".join(fileN.stem.split("_")[1:])
            file_list.append(fileN)
            acquisitionParameter.update(
                {"NAME": name_handle, "NUMBEROFFILES": 1, "DATA_FILES": file_list}
            )
            experimentList.update({fileN.stem: acquisitionParameter})
        elif not fileN.stem.split("_")[-1] == "Gate" and "_IV_" in str(fileN.stem):
            # now you know for sure you are processing an IV file.
            # For now, this is the same as above, but let's see if we can refactor this later... BUG
            acquisitionParameter = read_params_smFET(fileN)
            name_handle = "_".join(fileN.stem.split("_")[1:])
            file_list.append(fileN)
            acquisitionParameter.update(
                {"NAME": name_handle, "NUMBEROFFILES": 1, "DATA_FILES": file_list}
            )
            experimentList.update({fileN.stem: acquisitionParameter})
        file_list = []

    return experimentList


def get_data_smFET(metadata):
    assert len(metadata["DATA_FILES"]) == metadata["NUMBEROFFILES"]
    chan_num = metadata["CHANNEL"]

    if metadata["NUMBEROFFILES"] == 1:
        with open(str(metadata["DATA_FILES"][0]), "rb") as f:
            logdata = np.fromfile(f, dtype=np.int16)
    else:
        with open(str(metadata["DATA_FILES"][0]), "rb") as f:
            logdata = np.fromfile(f, dtype=np.int16)
        for i in metadata["DATA_FILES"][1:]:
            with open(str(i), "rb") as f:
                logdata = np.append(logdata, np.fromfile(f, dtype=np.int16))

    assert logdata.dtype == np.dtype("int16"), "What went wrong?"

    logdata = (
        (
            (
                np.float32(logdata)
                * metadata["SETUP_ADCVREF"]
                * 2.0
                / (2.0 ** metadata["SETUP_ADCBITS"] - 1)
            )
            - metadata["SETUP_AoffsetV"]
        )
        / (metadata["SETUP_TIAgain"] * metadata["SETUP_preADCgain"])
        * 1e9
    )

    assert logdata.dtype == np.dtype("float32"), "What went wrong?"

    return logdata


def read_params_smFET(logFilePath):
    """
    Load .param file setup parameter; the below parameters were generated by scanning through the codebase, and seeing what we actually needed to keep and to move forward with.  This is a historical anacronism from the use of setupParameter to initialize Chimera readings.

    TODO Now, as amended, this thing requires that there be exactly 1 .params file in each final data directory, which it's gonna be reading all the time, regardless of the filename or coding/tagging/parsing there.

    Parameters
    ----------
    logFilePath : Path
        Path to .params we're looking at.

    Returns
    -------
    setupParameter : dict
        The parameters pulled from the .params file.
    """
    keepers = [
        "mytimestamp",
        "SETUP_ADCSAMPLERATE",
        "VAC",
        "SETUP_ADCBITS",
        "SETUP_ADCVREF",
        "SETUP_AoffsetV",
        "SETUP_TIAgain",
        "SETUP_preADCgain",
        "TOTALTIME",
        "COMMENTS",
        "SAMPLE",
        "VG",
        "CHANNEL",
        "electrolyte",
        "conc",
        "BENCHVUE",
        "Temperature",
    ]
    setupParameter = dict.fromkeys(keepers)
    base_path = os.path.dirname(logFilePath)
    thing = logFilePath.parent.rglob("*.param")
    params_files = list(thing)
    try:
        assert (
            len(params_files) == 1
        ), "Assertion error in reader: one params file per dir."
    except AssertionError as msg:
        logging.error(msg)
        logging.error(str(base_path))
        exit()

    matFilePath = os.path.join(base_path, params_files[0])
    readerDict = file_to_dict(matFilePath)
    channel_name = (
        logFilePath.stem.split("Chan")[1].split("_")[1].lstrip("0")
    )  # half these fucking files have fucking _1 on the end.  WHo came up with this?
    assert isinstance(channel_name, str)
    assert (
        channel_name.isdigit()
    )  # make sure it's a string that can be a positive int...

    # OK, now map setup params to what we have from the params file
    # these below should match 1:1 with the values in the params file
    setupParameter["mytimestamp"] = str(params_files[0].name).split("_")[0]
    setupParameter["SETUP_ADCSAMPLERATE"] = int(25000)
    setupParameter["VAC"] = float(readerDict[("VDS " + channel_name + " value (mV)")])
    setupParameter["SETUP_ADCBITS"] = int(14)
    setupParameter["SETUP_ADCVREF"] = float(readerDict["Default vset (V)"])
    setupParameter["SETUP_TIAgain"] = (
        float(readerDict["Resistor " + channel_name + " gain (Mohm)"]) * 1e6
    )
    setupParameter["SETUP_AoffsetV"] = float(
        readerDict["ADC Offset " + channel_name + " value (V)"]
    )  # gain times offset voltage, and make it picoamps, not nanoamps...
    setupParameter["SETUP_preADCgain"] = float(
        readerDict["ResX " + channel_name + " value"]
    )
    setupParameter["TOTALTIME"] = float(readerDict["Measure Time (s)"])
    setupParameter["COMMENTS"] = readerDict[("Experiment Name")] + (
        readerDict[("Remark")] if "Remark" in readerDict.keys() else ""
    )
    setupParameter["SAMPLE"] = readerDict[("Chip Number")]
    try:
        setupParameter["VG"] = float(readerDict[("Gate (V)")]) - float(
            readerDict["HOLD " + channel_name + " value (V)"]
        )
    except KeyError:
        try:
            setupParameter["VG"] = -float(
                readerDict["HOLD " + channel_name + " value (V)"]
            )
        except KeyError:
            setupParameter[
                "VG"
            ] = np.nan  # if they don't do a gate sweep, they don't record a gate value.

    setupParameter["electrolyte"] = readerDict[("Electrolyte")]
    setupParameter["CHANNEL"] = channel_name
    setupParameter["Temperature"] = (
        readerDict[("Temperature")] if "Temperature" in readerDict.keys() else np.nan
    )

    del readerDict
    setupParameter["mytimestamp"] = datetime_convert(setupParameter["mytimestamp"])
    assert (
        setupParameter["SETUP_ADCBITS"] == 14
    ), "You are pulling in from hardware that Boyan didn't know about.  You're on your own."
    assert (
        setupParameter["SETUP_ADCSAMPLERATE"] == 25000
    ), "You are pulling in from hardware that Boyan didn't know about.  You're on your own."
    return setupParameter


def file_to_dict(filename):
    """
    Take in a filename that is parsed as a text file, split by a single equals sign.

    Return a dict that is the key value pair for that mapping.
    """
    d = {}
    with open(filename) as f:
        for line in f:
            if "=" in line:
                try:
                    (key, val) = line.split("=")
                except ValueError:
                    logging.info(
                        "Are there empty values in the right-hand side of some lines?"
                    )

                d[key.strip()] = val.strip()  # strip newlines and spaces
    return d


def create_experiment_list_Chimera(rc):
    """
    Returns a list of concatentated experiments on the Chimera, each of which is to make an Experiment object.

    Not a garbage can for random ints we're going to forget about...
    """
    benchvueFiles = []
    logFiles = []
    # This assert covers the fact that we rely on the data files being 2 seconds long, below.
    assert rc.args.instrumentType == "Chimera"
    print(rc.dataPath)
    if isinstance(rc.dataPath, list):
        preLogFilePath = rc.dataPath[0]
        for i in range(len(rc.dataPath)):
            logFiles.extend(sorted(rc.dataPath[i].rglob("*.log")))
            benchvueFiles.extend(sorted(rc.dataPath[i].rglob("*_benchvue.csv")))
    else:
        preLogFilePath = rc.dataPath
        logFiles.extend(sorted(rc.dataPath.rglob("*.log")))
        benchvueFiles.extend(sorted(rc.dataPath.rglob("*_benchvue.csv")))

    numberOfFiles = len(logFiles)
    assert (
        numberOfFiles >= 1
    ), "You were looking for Chimera data, and found no Chimera files.  Check the data directory, and check that you're passing the right tool."

    i = 0
    counter = 0
    experimentList = {}
    file_list = []

    allowed_time_lag = (
        rc.allowed_time_lag
    )  # You must start the Benchvue within 20 seconds of starting the Chimera.

    for fileN in logFiles:
        i = i + 1
        # if you are exiting a stream
        if counter >= 1 and (
            fileN.stem[:-16] != preLogFilePath.stem[:-16] or i == numberOfFiles
        ):
            acquisitionParameter.update(
                {
                    "NUMBEROFFILES": counter,
                    "DATA_FILES": file_list,
                    "TOTALTIME": counter * 2.5,
                }  ## TODO this assumes exaclty 2.5 sec per file, so if this changes, change this.
            )
            acquisitionParameter.update(parse_filename(preLogFilePath.stem))
            experimentList.update({preLogFilePath.stem[:-16]: acquisitionParameter})
            file_list = []
            counter = 0
        # if you transitioning into a stream
        if fileN.stem[:-16] != preLogFilePath.stem[:-16]:
            acquisitionParameter = read_mat_Chimera(fileN)
            assert type(acquisitionParameter) == dict
            counter = 1
            file_list.append(fileN)
            preLogFilePath = fileN
            # go check for benchvue
            acquisitionParameter.update({"BENCHVUE": "NULL"})
            acquisitionParameter.update(parse_filename(fileN.stem))
            acquisitionParameter.update(
                {
                    "NAME": fileN.stem[0:-16],
                    "NUMBEROFFILES": counter,
                    "DATA_FILES": file_list,
                    "TOTALTIME": counter
                    * 2.5,  ## TODO this assumes exaclty 2.5 sec per file, so if this changes, change this.
                }
            )
            experimentList.update({fileN.stem[:-16]: acquisitionParameter})
            for bench in benchvueFiles:
                date = bench.stem.split("_")[5]
                time = bench.stem.split("_")[6]
                bench_time = datetime.strptime(date + time, "%Y%m%d%H%M%S")
                if str(bench.stem)[:-25] == fileN.stem[:-16]:
                    logging.info(
                        "There's name-matching benchvues on "
                        + str(fileN)
                        + ".  Now check for time matches."
                    )
                if (
                    str(bench.stem)[:-25] == fileN.stem[:-16]
                    and bench_time - acquisitionParameter["mytimestamp"]
                    > timedelta(seconds=0)
                    and bench_time - acquisitionParameter["mytimestamp"]
                    < allowed_time_lag
                ):
                    # So, rule is to start Chimera first.  So, verify that the acqparam is before the bench_time
                    logging.info(
                        "Benchvue was started after Chimera by "
                        + str(bench_time - acquisitionParameter["mytimestamp"])
                    )
                    logging.info(
                        "Theres a name-matching and time-matching benchvue on "
                        + str(fileN)
                    )
                    acquisitionParameter.update({"BENCHVUE": bench})
                    benchvueFiles.remove(
                        bench
                    )  # benchvue files are unique to an exp, so what's the deal here?
        # if you continue along a stream
        else:
            # only all files if not specified differently by args
            if counter < rc.args.fileNumber:
                file_list.append(fileN)
                counter = counter + 1
        # check that everything that had a pindown has a benchvue TODO fix below
        # if "pindown" in str(fileN):
        #     print(fileN)
        #     print(acquisitionParameter)
        #     assert acquisitionParameter["BENCHVUE"] != 'NULL'

    return experimentList


def get_data_Chimera(setupParameter):
    chLen = 10485760  # one .log file of chimera number of numbers
    logData = np.empty(chLen * setupParameter["NUMBEROFFILES"], dtype="int16")
    i = 0
    for fileN in setupParameter["DATA_FILES"]:
        #        logging.info('Reading ' + str(fileN))
        tmp = read_binary_Chimera(fileN, setupParameter)
        lastShape = tmp.shape[0]
        logData[i * chLen : (i * chLen) + lastShape] = tmp.astype(dtype=np.int16)
        i = i + 1
    logData = logData[0 : ((i - 1) * chLen) + lastShape]
    return logData.astype(dtype=np.int16, copy=False)


def read_binary_Chimera(logFilePath, setupParameter):
    # Read binary and convert to currents
    bitmask = np.uint16(
        (2 ** 16 - 1) - (2 ** (16 - setupParameter["SETUP_ADCBITS"]) - 1)
    )
    with open(str(logFilePath), "rb") as dataFile:
        rawData = np.bitwise_and(np.fromfile(dataFile, dtype=np.uint16), bitmask)
        # read as uint16 and bit wise operation with bitmask
    return convert_current_Chimera(rawData, setupParameter)


def convert_current_Chimera(rawData, setupParameter):
    # scales uint16 to int16 current data in pA
    currentProp = get_current_prop_Chimera(setupParameter)
    currentOffset = get_current_offset_Chimera(setupParameter)
    rawData = np.around((currentProp * rawData) / 100).astype(
        dtype=np.int16, copy=False
    )  # /1000 to scale back after multiplication
    return (rawData + currentOffset).astype(dtype=np.int16, copy=False)


def get_current_prop_Chimera(setupParameter):
    currentProp = np.float32(
        100
        * 1e12
        * (
            np.float32(setupParameter["SETUP_ADCVREF"])
            / np.float32(
                setupParameter["SETUP_TIAgain"] * setupParameter["SETUP_preADCgain"]
            )
        )
        * np.float32(2.0 / (2.0 ** 16))
    )  # times 1000 to maintain digits
    return currentProp


def get_current_offset_Chimera(setupParameter):
    # calculates constants to scale binary to pA
    currentOffset = np.int64(
        1e12
        * (
            -np.float32(setupParameter["SETUP_ADCVREF"])
            / np.float32(
                setupParameter["SETUP_TIAgain"] * setupParameter["SETUP_preADCgain"]
            )
            + np.float32(setupParameter["SETUP_pAoffset"])
        )
    )
    return currentOffset


def read_mat_Chimera(logFilePath):
    # Load mat file setup parameter
    garbage = []
    # the below were generated by scanning through the file below and seeing what we actually need to keep.
    keepers = [
        "mytimestamp",
        "SETUP_ADCSAMPLERATE",
        "bias2value",
        "SETUP_ADCBITS",
        "SETUP_ADCVREF",
        "SETUP_pAoffset",
        "SETUP_TIAgain",
        "SETUP_preADCgain",
    ]
    matFilePath = logFilePath.with_suffix(".mat")
    setupParameter = sio.loadmat(str(matFilePath))
    assert type(setupParameter) == dict
    for i in setupParameter:
        if i not in keepers:
            garbage.append(i)
    for i in garbage:
        del setupParameter[i]
    for i in setupParameter:
        setupParameter[i] = setupParameter[i][0][0]
    setupParameter["VAC"] = setupParameter["bias2value"]
    setupParameter["mytimestamp"] = datetime_mtopy(setupParameter["mytimestamp"])
    setupParameter["VAG"] = np.nan
    del setupParameter["bias2value"]
    assert (
        setupParameter["SETUP_ADCBITS"] == 14
    ), "You are pulling in from hardware that Boyan didn't use.  You're on your own."
    assert (
        setupParameter["SETUP_ADCVREF"] == 2.5
    ), "You are pulling in from hardware that Boyan didn't use.  You're on your own."
    assert (
        setupParameter["SETUP_TIAgain"] == 100000000
    ), "You are pulling in from hardware that Boyan didn't use.  You're on your own."
    return setupParameter


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def file_length_sec(fileN):
    """
    Takes file handle, returns the file length in seconds.
    """
    if fileN.suffix == ".log":
        return fileN.stat().st_size / 2  # assumes dtype is still int16
    elif fileN.suffix == ".pckl":
        return fileN.stat().st_size / 2  # assumes dtype is still int16
    elif fileN.suffix == ".asc":
        return (
            (fileN.stat().st_size - 206.0) / 162000214.0 * 10.0
        )  # might have an error
    elif fileN.suffix == ".bin":
        return fileN.stat().st_size / 2  # assumes dtype is still int16
    else:
        assert (
            (fileN.suffix == ".asc")
            or fileN.suffix == ".log"
            or fileN.suffix == ".bin"
            or fileN.suffix == ".pckl"
        ), "unknow file type"


def datetime_convert(datenum_string):
    """
    Input
    A string that looks like a date.
    Output
    A datetime object
    blagh -- https://strftime.org/
    """
    datetime_object = datetime.strptime(datenum_string, "%Y%m%d-%H%M%S")
    return datetime_object


def datetime_mtopy(datenum):
    """
    Input
    The fractional day count according to datenum datatype in matlab
    Output
    The date and time as a instance of type datetime in python
    Notes on day counting
    matlab: day one is 1 Jan 0000
    python: day one is 1 Jan 0001
    hence a reduction of 366 days, for year 0 AD was a leap year
    """
    ii = datetime.fromordinal(int(datenum) - 366)
    ff = timedelta(days=datenum % 1)
    return ii + ff


# ------------------------------------------------------------
## check event finder and eventplotting being compatible
def check_plot_events_args(rc):
    eventFileList = sorted(rc.resultsPath.rglob("events/*events.csv"))
    if len(eventFileList) != 0:
        logging.info(
            "Updating run constants to match event finding conditions to ensure sanity in plotevents."
        )
        eventFrame = pd.read_csv(eventFileList[-1])
        rc.args.PDF = eventFrame.PDF[0]
        rc.args.PDFreversal = eventFrame.PDFREVERSAL[0]
        rc.args.detrend = eventFrame.DETREND[0]
        rc.args.filter = eventFrame.FILTERCUTOFF[0]
        rc.args.stopBand = eventFrame.STOPBAND[0]
        rc.args.notch = eventFrame.NOTCH[0]
        del eventFrame
    return rc


def parse_filename(fileName):
    """
    Parses filename, validates that it makes sense and returns measurement params.

    Parameters
    ----------
    fileName : str()
        The path to look at.

    Returns
    -------
    measurementParameter : dict
        Measurement params that that then go in the rest of the code.
    """
    bathSalt = "NULL"
    bathConcentration = 0
    backsideSalt = "NULL"
    backsideConc = 0

    fileNameParser = fileName.rsplit(".log", 1)[0].split("_")
    sample = fileNameParser[0]

    if (
        "KCl" in fileNameParser[1]
    ):  # TODO oh god is this fragile against anything but KCl
        COMMENTS = fileNameParser[2]
        if "-" in fileNameParser[1]:  # this is a multi-salt thing
            bathSalt = fileNameParser[1].split("-")[0].split("M", 1)[1]
            backsideSalt = fileNameParser[1].split("-")[1].split("M", 1)[1]
            if "m" in fileNameParser[1].split("-")[0]:
                bathConcentration = (
                    float(fileNameParser[1].split("-")[0].split("m", 1)[0]) * 1.0e-3
                )
            elif "u" in fileNameParser[1]:
                bathConcentration = (
                    float(fileNameParser[1].split("-")[0].split("u", 1)[0]) * 1.0e-6
                )
            else:
                bathConcentration = float(
                    fileNameParser[1].split("-")[0].split("M", 1)[0]
                )
            if "m" in fileNameParser[1].split("-")[1]:
                backsideConc = (
                    float(fileNameParser[1].split("-")[1].split("m", 1)[0]) * 1.0e-3
                )
            elif "u" in fileNameParser[1]:
                backsideConc = (
                    float(fileNameParser[1].split("-")[1].split("u", 1)[0]) * 1.0e-6
                )
            else:
                backsideConc = float(fileNameParser[1].split("-")[1].split("M", 1)[0])
        else:  # single-salt
            bathSalt = fileNameParser[1].split("M", 1)[1]
            if "m" in fileNameParser[1]:
                bathConcentration = float(fileNameParser[1].split("m", 1)[0]) * 1.0e-3
            elif "u" in fileNameParser[1]:
                bathConcentration = float(fileNameParser[1].split("u", 1)[0]) * 1.0e-6
            else:
                bathConcentration = float(fileNameParser[1].split("M", 1)[0])
    else:
        COMMENTS = fileNameParser[1]
        backsideSalt = "N.A"
        backsideConc = np.nan
        bathConcentration = np.nan
        bathSalt = "N.A."

    if len(fileNameParser) > 6:
        COMMENTS = "".join(fileNameParser[2:-4][0])

    measurementParameter = {
        "SAMPLE": sample,
        "electrolyte": bathSalt,  # seriously?
        "conc": bathConcentration,
        "otherelectrolyte": backsideSalt,
        "otherconc": backsideConc,
        "COMMENTS": COMMENTS,
    }

    return measurementParameter


# ------------------------------------------------------------
## add Fake Events with rate and depth levels (numberOfLEvels)
def fakeEvents(
    rawData, sampleRate, depth, rate, numberOfLevels, filterCutoff, setupParameter
):
    dataSetSize = np.size(rawData)
    numberOfEvents = np.int(np.floor(dataSetSize / sampleRate * rate))
    minEventLength = np.int(np.floor(1.0 / filterCutoff * sampleRate))
    maxEventLength = np.int(np.floor(20e-3 * sampleRate))
    individualDepth = np.random.choice(
        np.arange(
            1.0 / numberOfLevels, 1.0 + 1.0 / numberOfLevels, 1.0 / numberOfLevels
        )
        * depth,
        size=numberOfEvents,
        replace=True,
    )
    individualDuration = np.random.random_integers(
        low=minEventLength, high=maxEventLength, size=numberOfEvents
    )
    individualPosition = np.random.random_integers(
        low=0, high=dataSetSize - np.max(individualDuration), size=numberOfEvents
    )
    fakeData = rawData
    alreadyEvent = np.full((dataSetSize, 1), False)
    counter = 0
    for i in range(numberOfEvents):
        if not np.any(
            alreadyEvent[
                individualPosition[i] : individualPosition[i] + individualDuration[i]
            ]
        ):
            fakeData[
                individualPosition[i] : individualPosition[i] + individualDuration[i]
            ] = (
                fakeData[
                    individualPosition[i] : individualPosition[i]
                    + individualDuration[i]
                ]
                + individualDepth[i]
            )
            alreadyEvent[
                individualPosition[i] : individualPosition[i]
                + individualDuration[i]
                + minEventLength
            ] = True
            counter = counter + 1
    logging.info(
        "Added "
        + str(counter)
        + " fake events (off attempted "
        + str(numberOfEvents)
        + ")"
    )
    return fakeData


# ------------------------------------------------------------
## BENCHVUE
def handle_benchvue_import(file_h):
    assert isinstance(
        file_h, Path
    ), "You're trying to import something that's not a file...much less a benchvue file"
    headers = ["time", "voltage", "current"]
    dtypes = {"time": "str", "voltage": "float", "current": "float"}
    parse_dates = ["time"]
    entries = pd.read_csv(
        file_h,
        sep=",",
        usecols=[3, 4, 5],
        skiprows=1,
        header=None,
        names=headers,
        dtype=dtypes,
        parse_dates=parse_dates,
    )
    entries = entries.fillna("None")
    # OK, what's this?  Occasionally, the amplifier puts out random values that are insanely high.  Not clear what this is, and has to get sorted in SynchroDance.  The thing to fix this is to replace just the voltages with None, and this plots cleanly.  I also tried averaging the values,  but that fails with when is happens in the start and end.  So, this is what we do for now..
    entries.loc[entries["voltage"] > 10000, "voltage"] = None  # 10 volts
    entries.current = (
        entries.current * 1000
    )  # convert nA in benchvue to pA in this software.
    return entries


def pull_transitions_from_benchvue(benchvue, rc, thresh=25):
    """
    Returns a list of pairs of entry and exit points for events from the benchvue file, trimmed by rc.front_crop and rc.back_crop.  Checks that this does not mess up the event length

    Parameters
    ----------
    benchvue : dataframe
        The benchvue dataframe
    rc : runConstants
        run constants
    thresh : float
        The minimal threshold that we consider to be different between two voltage levels.  Defaults to 10 mV.

    Returns
    -------
    out : list
        List of pairs of transitions.
    """
    out = []
    # changes = np.gradient(benchvue.voltage, 2)
    # peaks = ssi.find_peaks(abs(changes))
    # the issue with the peaks things is that it eats points here, and this translates to a million points in the Chimera stream
    start = 0
    v = benchvue.iloc[0].voltage
    for i in range(len(benchvue.time)):
        if abs(benchvue.iloc[i].voltage - v) > thresh:  # entered a new phase.
            # handle the old phase, both with crops
            assert benchvue.iloc[start].time <= benchvue.iloc[i - 1].time
            element = [
                benchvue.iloc[start].time + rc.front_crop,
                benchvue.iloc[i - 1].time - rc.back_crop,
            ]
            # now, go check that your crops are not too long
            # and that they don't zero out the data stream length
            try:
                assert element[1] > element[0], "Should move forward in time on crops."
                assert (
                    element[1] > benchvue.iloc[start].time
                ), "Should have end be in the file."
                assert (
                    element[0] < benchvue.iloc[i - 1].time
                ), "Should have the start be in the file."
            except AssertionError as e:
                logging.warning(
                    "There's an issue with the current crops with the length of the slice when you measured.  Not cropping.  Benchvue file is "
                    + str(benchvue)
                )
                element = [benchvue.iloc[start].time, benchvue.iloc[i - 1].time]
                logging.warning(
                    "Uncropped slice length is " + str(element[1] - element[0])
                )
                logging.warning(
                    "Default crops are "
                    + str(rc.front_crop)
                    + " and "
                    + str(rc.back_crop)
                )
            out.append(element)
            start = i  # reset both counters...
            v = benchvue.iloc[i].voltage
    if len(out) == 0:
        out.append(
            [
                benchvue.iloc[start].time + rc.front_crop,
                benchvue.iloc[-1].time - rc.back_crop,
            ]
        )
    return out


def fix_voltage(VA, VAC, vg):
    """
    This takes an Offset Of The Day, a benchvue gate voltage and a VAC, and returns the gate voltage relative to VAC
    """
    if abs(vg - 0) < 1:
        return 0  # if you are within 1 mV of zero, return the zero to preserve zero as pinup.
    else:
        return vg - VA + VAC / 2


def get_VA(dataPath):
    """
    Gets Offset Of The Day for a data dir
    """
    VAFiles = []
    if isinstance(dataPath, list):
        for i in range(len(dataPath)):
            VAFiles.extend(sorted(dataPath[i].rglob("*OOTD.csv")))
    else:
        VAFiles.extend(sorted(dataPath.rglob("*OOTD.csv")))
    VA = np.nan
    for fileN in VAFiles:
        with open(fileN) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            row_count = 0
            for row in csv_reader:
                if row:  # skip empty lines that result from random Windows ^M stuff.
                    row_count += 1
                    VA = int(row[0])
            assert row_count == 1
    return VA


# ------------------------------------------------------------
## HEKA .asc FILE READING
def create_experiment_list_HEKA(rc):
    """
    Returns a list of concatentated experiments on the HEKA, each of which is to make an Experiment object.
    Not a garbage can for random ints we're going to forget about...
    """

    logFiles = []
    if isinstance(rc.dataPath, list):
        preLogFilePath = rc.dataPath[0]
        for i in range(len(rc.dataPath)):
            logFiles.extend(sorted(rc.dataPath[i].rglob("*.asc")))
    else:
        preLogFilePath = rc.dataPath
        logFiles.extend(sorted(rc.dataPath.rglob("*.asc")))

    numberOfFiles = len(logFiles)
    counter = 0
    experimentList = {}
    file_list = []
    i = 0
    for fileN in logFiles:
        i = i + 1
        # if you are exiting a stream
        if counter >= 1 and (
            fileN.stem[:-20] != preLogFilePath.stem[:-20] or i == numberOfFiles
        ):
            acquisitionParameter.update(
                {
                    "NAME": preLogFilePath.stem[:-20],
                    "NUMBEROFFILES": counter,
                    "DATA_FILES": file_list,
                }
            )
            acquisitionParameter.update(parse_filename(preLogFilePath.stem))
            experimentList.update({preLogFilePath.stem[:-20]: acquisitionParameter})
            file_list = []
            counter = 0
        # if you transitioning into a stream
        if fileN.stem[:-20] != preLogFilePath.stem[:-20]:
            acquisitionParameter = read_HEKAHEADER(fileN)
            counter = 1
            file_list.append(fileN)
            preLogFilePath = fileN
            # go check for benchvue
            acquisitionParameter.update({"BENCHVUE": "NULL"})
            acquisitionParameter.update(parse_filename(preLogFilePath.stem))
            acquisitionParameter.update(
                {
                    "NAME": preLogFilePath.stem[:-20],
                    "NUMBEROFFILES": counter,
                    "DATA_FILES": file_list,
                }
            )
            experimentList.update({preLogFilePath.stem[:-20]: acquisitionParameter})

        # if you are coninuing along a stream
        else:
            # only all files if not specified differently by args
            if counter < rc.args.fileNumber:
                file_list.append(fileN)
                counter = counter + 1
    return experimentList


def read_HEKAHEADER(logFilePath):
    # Load header of asc file and create basic experiment entry

    setupParameter = {
        "SETUP_ADCBITS": np.nan,
        "SETUP_ADCVREF": np.nan,
        "SETUP_pAoffset": np.nan,
        "SETUP_TIAgain": np.nan,
        "SETUP_preADCgain": np.nan,
    }

    # read 2nd row and save to timestamp and sweep information
    tmp = np.loadtxt(
        logFilePath.with_suffix(".asc"),
        dtype=str,
        delimiter=",",
        skiprows=1,
        max_rows=1,
    )
    setupParameter["mytimestamp"] = datetime.strptime(tmp[3], " %Y/%m/%d %H:%M:%S.%f")
    setupParameter["HEKAFile"] = tmp[4].replace('"', "").replace(" ", "")
    setupParameter["HEKATrace"] = tmp[0]

    # read 3rd row for COMMENTS
    tmp = np.loadtxt(
        logFilePath.with_suffix(".asc"),
        dtype=str,
        delimiter=",",
        skiprows=2,
        max_rows=1,
    )
    if '"Index"' not in tmp:
        setupParameter["HEKAComment"] = tmp
    else:
        setupParameter["HEKACommsent"] = "NONE"

    # read 5th row for time stamp ADC calculation
    tmp = np.loadtxt(
        logFilePath.with_suffix(".asc"),
        dtype=float,
        delimiter=",",
        skiprows=4,
        max_rows=2,
        usecols=1,
    )
    setupParameter["SETUP_ADCSAMPLERATE"] = np.round(1.0 / (tmp[1] - tmp[0]))

    # read 5th row 5th col for VAC calculation
    try:
        setupParameter["VAC"] = (
            np.mean(
                np.loadtxt(
                    logFilePath.with_suffix(".asc"),
                    dtype=float,
                    delimiter=",",
                    skiprows=4,
                    usecols=4,
                    max_rows=500,
                )
            )
            * 1e3
        )
    except:
        try:
            group = re.search("(_)(-*\d*)(_mV_)", str(logFilePath.stem))
            setupParameter["VAC"] = np.float64(group[2])
        except:
            setupParameter["VAC"] = np.nan

    return setupParameter


def get_data_HEKA(data_dict):
    # Load current from asc file and convert to pA

    # read all files
    logData = np.empty(0, dtype="int16")
    for fileN in data_dict["DATA_FILES"]:
        # check for current in 3 column and unit  = [A]
        skiprow = 2
        tmp = np.loadtxt(
            fileN.with_suffix(".asc"), dtype=str, delimiter=",", skiprows=2, max_rows=1
        )
        if '"Index"' not in tmp:
            skiprow = 3

        tmp = np.loadtxt(
            fileN.with_suffix(".asc"),
            dtype=str,
            delimiter=",",
            skiprows=skiprow,
            max_rows=1,
        )
        assert "[A]" in tmp[2].replace('"', "").replace(
            " ", ""
        ), "3 column in file {s0} contains not current in [A]".format(s0=fileN)

        logData = np.append(
            logData,
            (
                np.loadtxt(
                    fileN.with_suffix(".asc"),
                    dtype=float,
                    delimiter=",",
                    skiprows=skiprow + 1,
                    usecols=2,
                )
                * 1e12
            ).astype(dtype=np.int16, copy=False),
        )

    return logData


# ------------------------------------------------------------
## PCKL

def create_experiment_list_PCKL(rc):
    """

    """
    logFiles = []
    if isinstance(rc.dataPath, list):
        preLogFilePath = rc.dataPath[0]
        for i in range(len(rc.dataPath)):
            logFiles.extend(sorted(rc.dataPath[i].rglob("*.pckl")))
    else:
        preLogFilePath = rc.dataPath
        logFiles.extend(sorted(rc.dataPath.rglob("*.pckl")))

    numberOfFiles = len(logFiles)
    # assert (numberOfFiles >= 1), "You found bupkus."
    i = 0
    counter = 0
    experimentList = {}
    file_list = []

    for fileN in logFiles:
        acquisitionParameter = read_params_PCKL(fileN)
        name_handle = "_".join(fileN.stem.split("_")[1:-3])
        file_list.append(fileN)
        acquisitionParameter.update(
            {"NAME": name_handle, "NUMBEROFFILES": 1, "DATA_FILES": file_list}
        )
        if (rc.args.channel == []):
            experimentList.update({fileN.stem: acquisitionParameter})
        elif (str(acquisitionParameter['CHANNEL']) in rc.args.channel):
            experimentList.update({fileN.stem: acquisitionParameter})
        file_list = []

    return experimentList


def read_params_PCKL(fileN):
    """

    """
    csvFileN = fileN.with_suffix(".csv")
    experimentDF = pd.read_csv(csvFileN, parse_dates=["mytimestamp"])
    name = experimentDF["Unnamed: 0"][0]
    experimentDF = experimentDF.drop(columns="Unnamed: 0")
    experimentDF = experimentDF.set_index(pd.Index([name]))
    expDict = experimentDF.to_dict(orient="records")

    return expDict[0]


def get_data_PCKL(metadata):
    """
    This function pickles segment and saves it to disk
    
    Parameters
    ----------
    seg : dataclass object Slice()
        filtered data
    
    rc : dataclass object rc()
    
    Returns
    ----------
    None.  Will write a byte stream "pickle" file to disk.
    
    """

    readFile = metadata["DATA_FILES"][0]

    with open(readFile, "rb") as f:
        seg = pickle.load(f)

    return seg.IaRAW, seg.Ia, seg.samplerate


def get_segment_PCKL(metadata):
    """
    This function pickles segment and saves it to disk
    
    Parameters
    ----------
    seg : dataclass object Slice()
        filtered data
    
    rc : dataclass object rc()
    
    Returns
    ----------
    None.  Will write a byte stream "pickle" file to disk.
    
    """

    readFile = metadata["DATA_FILES"][0]

    with open(readFile, "rb") as f:
        seg = pickle.load(f)

    return seg
