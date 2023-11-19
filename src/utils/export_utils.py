import numpy as np
import logging
import pickle
from pathlib import Path
from import_utils import get_stem
from scipy.interpolate import interp1d
import pandas as pd

import sys

sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src"))

import data_classes
import event_utils
import filter_utils

logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s %(levelname)s - %(message)s"
)


def export_experiment(experiment):
    """
    Exports filtered and subsampled data from primarly sampling system, as the experiment and runconstants dictate.

    Parameters
    ----------
    experiment : experiment
        The experiment whose primary sampling system current data you'd like to export.

    Returns
    -------
    None.
    """
    logging.info("Now exporting csv of " + experiment.name)
    savePath = experiment.rc.resultsPath / (get_stem(experiment) + "_Ia.csv")
    filtTime = (
        np.arange(0, np.size(experiment.Ia), 1, dtype=np.float32)
        / experiment.filtersamplerate
    )
    np.savetxt(
        str(savePath),
        np.array([filtTime, experiment.Ia]).T,
        delimiter=",",
        fmt=["%.2e", "%.d"],
    )
    return


def export_slice(sliver, args, rc):
    """
    Exports filtered and subsampled data from a Slice of the primarly sampling system, as the experiment and runconstants dictate.

    Parameters
    ----------
    sliver : Slice()
        The Slice whose primary sampling system current data you'd like to export.

    Returns
    -------
    None.
    """
    logging.info("Now exporting csv of " + sliver.name)
    savePath = rc.resultsPath / (str(sliver.name) + "_Ia.csv")
    filtTime = np.arange(0, np.size(sliver.Ia), 1, dtype=np.float32) / sliver.samplerate
    np.savetxt(
        str(savePath),
        np.array([filtTime, sliver.Ia]).T,
        delimiter=",",
        fmt=["%.2e", "%.d"],
    )
    return


def export_experiment_danube(experiment):
    """
    Exports filtered and subsampled data from primarly sampling system, as the experiment and runconstants dictate, to the DAnube csv format, which asks for the point count as the first line.

    Parameters
    ----------
    experiment : experiment
        The experiment whose primary sampling system current data you'd like to export to Danube's format, which requries the first line contain the point count.

    Returns
    -------
    None.
    """
    logging.info("Now exporting Danube-compatible csv of " + experiment.name)
    savePath = experiment.rc.resultsPath / (get_stem(experiment) + "_Ia.csv")
    with open(savePath, "w") as out:
        out.write("{0}".format(len(experiment.Ia)))
        out.write("\n")
    with open(savePath, "a") as the_file:
        np.savetxt(the_file, np.array(experiment.Ia).T, fmt="%.2f")
    return


def export_event_trace_chunks(time, data, eventTime, eventValues, writeFile):
    """
    This function saves individual .csv files for each rc.plotidealevent duration with the original trace and the idealized event trace and saves the file to the results folder. This was requested from Yoonhee for plotting the data with another program.

    Output: .csv files with 3 columns, col1: Timestamp [s], current or raw data [unit], current of idealized event trace data [unit]

    Parameters
    ----------
    time : np.array with floats
        time in seconds
    data : np.array floats
        filtered data
    eventTime : np.array
        time stamps of last baseline value before event, first event point time stamp, last event point time stamp, first new baseline point time stamp (4 points per event)
    eventValues : np.array
        current value of last baseline value before event, first event point current value, last event point current value, first new baseline point current value (4 points per event)
    writeFile : posixPath
        path and file name to save trace .csv


    Returns
    -------
    None.
    """
    logging.info("Save ideal event trace to file " + str(writeFile))
    f = interp1d(eventTime, eventValues)
    np.savetxt(
        str(writeFile),
        np.array([time, data, f(time)]).T,
        delimiter=",",
        fmt=["%e", "%.4e", "%.4e"],
    )
    return


def save_segment(seg, savePath):

    """
    This function pickles segment and saves it to disk


    Parameters
    ----------
    seg : dataclass object Slice()
        filtered data
    savePath : dataclass object rc()

    Returns
    ----------
    None.  Will write a byte stream "pickle" file to disk.

    """

    # write segment
    name = "{s1}_savedTrace.pckl".format(s1=seg.dfIndex())
    writeFile = savePath / name
    writeFile.parent.mkdir(parents=True, exist_ok=True)

    with open(writeFile, "wb") as f:
        pickle.dump(seg, f)

    # write metaData as csv
    seg.exp.metaData["mytimestamp"] = (
        seg.statsDF["timestamp"].values[0].strftime("%Y-%m-%d, %H:%M:%S")
    )
    seg.exp.metaData["slice_i"] = seg.statsDF["slice_i"].values[0]
    seg.exp.metaData["unit"] = seg.statsDF["unit"].values[0]
    exp_frame = pd.Series(seg.exp.metaData).to_frame().T
    # exp_frame = pd.DataFrame.from_dict(seg.exp.metaData)  ## use keys as rows...
    name = "{s1}_savedTrace.csv".format(s1=seg.dfIndex())
    writeFile = savePath / name
    exp_frame.to_csv(writeFile, sep=",", encoding="utf-8")

    return
