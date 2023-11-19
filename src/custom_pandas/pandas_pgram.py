import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import multiprocessing

import sys
import distutils.dir_util
import shutil
from pathlib import Path

from numba import njit


@njit(cache=True)
def get_data_density(
    event_starts, event_stops, measurement_length, sample_rate, avg_time
):
    """
    Calculates the density, in events per second, of the event flags in data.  The thing is backward looking, looking at the last avg_time seconds before the point.  If an event spans a start or end boundary, it counts as half an event for that slice, up to a maximum of two halves.

    I did try this with DF comprehensions, and it's insanely slow (like overnight vs immediate).

    Parameters
    ----------
    event_starts :  nparray
        Sorted array of event start indicies.
    event_stops :  nparray
        Sorted array oof event top indicies.
    measurement_length :  float
        How many points are here?
    sample_rate :  float
        Sample rate for this data
    avg_time :  float
        How many backward-looking seconds to average over for event count

    Returns
    -------
    out : nparray()
        An array of length len(data) that returns the event frequency in units of events/sec
    """
    data_dens = np.zeros(int(measurement_length))
    assert (
        avg_time * sample_rate < measurement_length
    ), "You're putting in too many seconds to do backwards averaging."
    for i in range(int(measurement_length)):
        start = max(
            i - int(avg_time * sample_rate), 0
        )  # just do it up to the start point, don't look out of the array...
        end = i
        # OK, look for the start index in the event_starts, and see if you're starting on a start or on an end
        start_in_starts = np.searchsorted(event_starts, start)
        start_in_ends = np.searchsorted(event_stops, start)
        if start_in_starts == start_in_ends + 1 or start_in_starts == start_in_ends - 1:
            data_dens[i] += 0.5
        # Same deal for the end index
        end_in_starts = np.searchsorted(event_starts, end)
        end_in_ends = np.searchsorted(event_stops, end)
        if end_in_starts == end_in_ends + 1 or end_in_starts == end_in_ends - 1:
            data_dens[i] += 0.5
        # Then, having fixed the side bits, do the main density.
        length = min(end_in_starts, end_in_ends) - min(start_in_starts, start_in_ends)
        data_dens[i] += length
    return data_dens


def plot_pgram(time, datasum, datadensity, experiment, filename):
    """
    Plots the pgram for the time and data variables here

    Parameters
    ----------
    time :  nparray
        The time axis
    datasum :  nparray
        The data cumsum
    datadensity : nparray
        The data density of the points per second.
    experiment :  string
        The name of the experiment which is being plotted
    filename :  PosixPath()
        dir to write the png to

    Returns
    -------
    None
        Will write a file.
    """
    fig = plt.figure(figsize=(8, 8))
    gs = matplotlib.gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(
        time,
        datasum,
        label="Cumulative event time",
        marker=",",
        linestyle="-",
        color="darkblue",
    )
    ax1.grid()
    ax1.set_title(r"{}".format(experiment.replace("_", "\_")))
    ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    Limits = [0, time[-1]]
    ax1.set_xlim(Limits)
    ax1.set_ylim(Limits)
    ax1.set_xlabel(r"Time [$s$]")
    ax1.set_ylabel(r"Cumulative event time [$s$] ")
    ax2 = ax1.twinx()
    ax2.plot(
        time,
        datadensity,
        label="Event density",
        marker=",",
        linestyle="-",
        color="green",
    )
    ax2.set_ylabel(r"Event density [$1/sec$]")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    return


def processing_function(handle):
    """
    Takes a filename of an event file, then plots the p-gram for the events found there.

    Parameters
    ----------
    handle :  List of length 2
        The filename of the events output CSV file, and the path to which to save the pngs

    Returns
    -------
    None
        Will write a file.
    """
    assert (
        len(handle) == 2
    ), "Misplaced experiment handle passed to processing_function, should be: [csvfile, path]"
    the_file = handle[0]
    the_path = handle[1]
    DF = pd.read_csv(the_file)
    sample_rate = DF["SAMPLERATE"][0]
    total_wrong_time = DF["TOTALTRACETIME"][
        0
    ]  # hardcoded microseconds!  BUG what's the deal with the CSV output of the eventfinder?
    total_time = total_wrong_time / 1e6
    experiment_name = DF["index"][0]
    measurement_length = sample_rate * total_time
    data = np.zeros(int(measurement_length))
    for index, row in DF.iterrows():
        data[row["EVENTSTARTIDX"] : row["EVENTSTOPIDX"]] = 1.0 / sample_rate
    filename = the_path / (experiment_name + "_pgram.png")
    time = np.arange(measurement_length) / float(sample_rate)
    event_starts = DF["EVENTSTARTIDX"].to_numpy()
    event_stops = DF["EVENTSTOPIDX"].to_numpy()
    data_density = get_data_density(
        event_starts, event_stops, measurement_length, sample_rate, 1
    )
    plot_pgram(time, np.cumsum(data), data_density, experiment_name, filename)
    return


if __name__ == "__main__":
    resultsPath = Path.cwd().parent / "DataAndResults/results"
    files_for_frame = sorted(resultsPath.rglob("*events.csv"))

    handles = []
    for thing in files_for_frame:
        handles.append([thing, resultsPath])

    # This is the thing you want to uncomment to run in serial mode and get actual error messages if this looks wierd.
    #    for thing in handles:
    #        processing_function(thing)

    mp = multiprocessing.get_context("spawn")
    master_pool = mp.Pool(
        processes=multiprocessing.cpu_count(), maxtasksperchild=1
    )  # BUG should fix that to not have hardcoded threadcount.
    slices = master_pool.imap_unordered(processing_function, handles, chunksize=1)
    master_pool.close()
    master_pool.join()
