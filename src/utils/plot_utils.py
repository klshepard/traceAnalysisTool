import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import gc
import math
import pandas as pd
import seaborn as sns
import math
import datetime
from datetime import timedelta
import os
from scipy import integrate
from textwrap import wrap
import logging

logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s %(levelname)s - %(message)s"
)

# plt.rc('text', usetex=True)
plt.rcParams["agg.path.chunksize"] = 100000
plt.rcParams["path.simplify_threshold"] = 0.1
plt.rcParams["path.simplify"] = True
plt.rcParams["figure.constrained_layout.use"] = False

import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))
import import_utils
import filter_utils
import export_utils
import event_utils

matplotlib.use("Agg")


def plot_benchvue(experiment, benchvue, out, rc):
    """
    Plots the benchvue file for the Chimera experiment.

    Parameters
    ----------
    experiment : Experiment()
        The experiment to plot.
    benchvue : list
        benchvue data structure.
    out : string
        where to write to
    rc :  runConstants
        RC element.

    Returns
    -------
    None
        Will write a file.

    """
    fig, ax1 = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(14)
    benchvue.plot(kind="line", x="time", y="voltage", ax=ax1)
    ax1.grid()
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"Run time")
    ax1.set_ylabel(r"$V_{AG}$ [$mV$]")
    seconds = mdates.SecondLocator(interval=5)
    ax1.xaxis.set_minor_locator(seconds)
    #    ax2 = ax1.twinx()
    #    ax2.plot(benchvue.time, benchvue.voltage, color='red')
    for step in out:
        plt.axvspan(step[0], step[1], alpha=0.25)
    plt.savefig(
        rc.resultsPath / (import_utils.get_stem(experiment) + "_segments_voltage.png")
    )
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    gc.collect()

    fig, ax1 = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(12)
    benchvue.plot(kind="line", x="time", y="current", ax=ax1)
    for step in out:
        plt.axvspan(step[0], step[1], alpha=0.25)
    ax1.grid()
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"Run time")
    ax1.set_ylabel(r"$I_{G}$ " + "[{sunit}]".format(sunit=experiment.unit))
    seconds = mdates.SecondLocator(interval=5)
    ax1.xaxis.set_minor_locator(seconds)
    plt.savefig(
        rc.resultsPath / (import_utils.get_stem(experiment) + "_segments_current.png")
    )
    # Garbage collecion
    fig.clear()
    plt.close(fig)

    fig, ax1 = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax1.plot(benchvue.time, benchvue.voltage, color="blue")
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"Run time")
    ax1.set_ylabel(r"$V_{AG}$ [$mV$]")
    seconds = mdates.SecondLocator(interval=5)
    ax1.xaxis.set_minor_locator(seconds)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$I_{G}$ " + "[{sunit}]".format(sunit=experiment.unit))
    ax2.plot(benchvue.time, benchvue.current, linestyle="", color="red", marker=".")
    ax2.grid()
    for step in out:
        plt.axvspan(step[0], step[1], alpha=0.3)
    plt.savefig(rc.resultsPath / (import_utils.get_stem(experiment) + "_segments.png"))
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    return


def plot_experiment(experiment, rc):
    """
    Default plot for an Experiment().  This will load the experimental data, eating RAM.

    Parameters
    ----------
    experiment : Experiment()
        The experiment to plot.
    rc :  runConstants
        RC element.

    Returns
    -------
    None
        Will write a file.

    """
    stem = import_utils.get_stem(experiment)
    logging.info("Generating plot of " + experiment.name)

    totalTime = np.size(experiment.Ia) / experiment.filtersamplerate
    filtTime = (
        np.arange(0, np.size(experiment.Ia), 1, dtype=np.float32)
        / np.size(experiment.Ia)
        * totalTime
    )

    fig = plt.figure(figsize=(14, 8))
    gs = matplotlib.gridspec.GridSpec(2, 1)

    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(
        filtTime,
        experiment.Ia,
        label="Filt",
        marker=",",
        linestyle="",
        color="darkblue",
    )

    del filtTime
    gc.collect()

    ax1.grid()
    ax1.set_title(r"{}".format(experiment.name.replace("_", "\_")))
    #    ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    #    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"T [$s$]")
    ax1.set_ylabel(r"$I_{A}$ " + "[{sunit}]".format(sunit=experiment.unit))
    ax1.annotate(
        "$V_{AC}$ = " + str(experiment.VAC) + " mV",
        xy=(0.9, 1.15),
        xycoords="axes fraction",
    )
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(
        experiment.benchvue.time,
        experiment.benchvue.current,
        linestyle="",
        marker=".",
        color="red",
        label="I",
    )

    ax3 = ax2.twinx()
    ax3.plot(
        experiment.benchvue.time,
        rc.VA - experiment.benchvue.voltage,
        color="blue",
        label="V",
    )

    ax2.grid()
    ax2.set_xlabel(r"Run time")
    ax3.set_ylabel(r"$V_{AG}$ [$mV$]")
    ax2.set_ylabel(r"$I_{G}$ " + "[{sunit}]".format(sunit=experiment.unit))
    if (
        experiment.benchvue.size > 3
    ):  # everything has a benchvue, just if no gate it's zeros...
        seconds = mdates.SecondLocator(interval=5)
        ax3.xaxis.set_minor_locator(seconds)
        out = import_utils.pull_transitions_from_benchvue(
            experiment.benchvue, experiment.rc
        )
        for step in out:
            plt.axvspan(step[0], step[1], alpha=0.3)
    #   fig.tight_layout()
    plt.savefig(rc.resultsPath / (stem + "_trace.png"))
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    gc.collect()
    return


def plot_smFET_IV(rc, experiment, I, V, sigma):
    """
    Plots smFET IVs.

    Parameters
    ----------
    rc : RunConstants()
        This is the run constants
    experiment : Experiment()
        This is the experiment
    I : nparray
        The current of the IV
    V : nparray
        The voltage of the IV
    sigma : nparray
        Whatever error bar values you want, one up, one down...

    Returns
    -------
    None.  Will write a image file.
    """
    stem = import_utils.get_stem(experiment)
    logging.info("Generating IV plot of " + experiment.name)
    fig = plt.figure(figsize=(6, 6))
    gs = matplotlib.gridspec.GridSpec(1, 1)

    ax1 = plt.subplot(gs[0, 0])
    ax1.errorbar(
        V,
        I,
        yerr=2 * sigma,
        capsize=3,
        label="IV",
        marker="o",
        linestyle=":",
        color="darkblue",
    )
    ax1.grid()
    ax1.set_title(r"{}".format(experiment.name.replace("_", "\_")))
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"$V_G$ [V]")
    ax1.set_ylabel(r"$I$ " + "[{sunit}]".format(sunit=experiment.unit))
    ax1.annotate(
        "$V_{AC}$ = " + str(experiment.VAC) + " mV",
        xy=(0.8, 1.1),
        xycoords="axes fraction",
    )
    #   fig.tight_layout()
    plt.savefig(rc.resultsPath / (stem + "_smFET_IV.png"))
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    return


def plot_smFET_IV_overlay(rc, experiment, I, V, sigma, old_I, old_V, old_sigma):
    """
    Plots smFET IVs for two experiments, overlays them.

    Parameters
    ----------
    rc : RunConstants()
        This is the run constants
    experiment : Experiment()
        This is the experiment
    I : nparray
        The current of the IV
    V : nparray
        The voltage of the IV
    sigma : nparray
        Whatever error bar values you want, one up, one down...
    old_I : nparray
        The current of the old_IV
    old_V : nparray
        The voltage of the old IV
    old_sigma : nparray
        Whatever error bar values you want for the old one, one up, one down...

    Returns
    -------
    None.  Will write a image file.
    """
    stem = import_utils.get_stem(experiment)
    logging.info("Generating IV plot overlay of " + experiment.name)
    fig = plt.figure(figsize=(6, 6))
    gs = matplotlib.gridspec.GridSpec(1, 1)

    ax1 = plt.subplot(gs[0, 0])
    ax1.errorbar(
        old_V,
        old_I,
        yerr=2 * old_sigma,
        capsize=3,
        marker="o",
        linestyle=":",
        color="darkblue",
        label="PreAttack",
    )
    ax1.errorbar(
        V,
        I,
        yerr=2 * sigma,
        capsize=3,
        marker="o",
        linestyle=":",
        color="darkred",
        label="PostAttack",
    )
    ax1.grid()
    ax1.legend(loc="lower left")
    ax1.set_title(r"{}".format(experiment.name.replace("_", "\_")))
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"$V_G$ [V]")
    ax1.set_ylabel(r"$I$ " + "[{sunit}]".format(sunit=experiment.unit))
    ax1.annotate(
        "$V_{AC}$ = " + str(experiment.VAC) + " mV",
        xy=(0.8, 1.075),
        xycoords="axes fraction",
    )
    plt.savefig(rc.resultsPath / (stem + "_smFET_IV_overlay.png"))
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    return


def plot_smFET_IV_multipleoverlay(rc, experiment, data):
    """
    Plots smFET IVs for arbitrary number of  experiments, overlays them.

    Parameters
    ----------
    rc : RunConstants()
        This is the run constants
    experiment : Experiment()
        This is the experiment
    data : list of lists
        [V, I, sigma, name]

    Returns
    -------
    None.  Will write a image file.
    """
    stem = import_utils.get_stem(experiment)
    fig = plt.figure(figsize=(6, 6))
    gs = matplotlib.gridspec.GridSpec(1, 1)

    ax1 = plt.subplot(gs[0, 0])
    for thing in data:
        ax1.errorbar(
            thing[0],
            thing[1],
            yerr=2 * thing[2],
            capsize=3,
            marker="o",
            linestyle=":",
            label=thing[3],
        )
    ax1.grid()
    ax1.legend(loc="lower left")
    ax1.set_title(r"{}".format(experiment.name.replace("_", "\_")))
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    ax1.set_xlabel(r"$V_G$ [V]")
    ax1.set_ylabel(r"$I$ " + "[{sunit}]".format(sunit=experiment.unit))
    ax1.annotate(
        "$V_{AC}$ = " + str(experiment.VAC) + " mV",
        xy=(0.8, 1.075),
        xycoords="axes fraction",
    )
    names = "-".join(thing[3] for thing in data)
    plt.savefig(rc.resultsPath / (stem + names + "_smFET_IV_multipleoverlay.png"))
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    return


def plot_slice(seg, rc):
    """
    Default plot for an Slice().

    Parameters
    ----------
    experiment : Slice()
        The experiment to plot.
    rc :  runConstants
        RC element.

    Returns
    -------
    None
        Will write a file.

    """
    logging.info("Generating plot of " + seg.name)
    mean = seg.cur_mean
    dev = seg.cur_std
    sampleRate = seg.samplerate
    totalTime = len(seg.Ia) * sampleRate
    stem = import_utils.get_stem(seg)
    plotmegaFileName = "{s1}_{s3:.0f}mV_both.png".format(s1=stem, s3=seg.VAC)
    megasavepath = rc.resultsPath / plotmegaFileName
    fig = plt.figure(figsize=(14, 8), constrained_layout=False)
    gs = matplotlib.gridspec.GridSpec(7, 6, hspace=1.2, wspace=0.7, figure=fig)

    if not np.isnan(np.array(seg.gate_current, dtype=np.float32)).any():
        ax_data = plt.subplot(gs[0:2, 0:])
    else:
        ax_data = plt.subplot(gs[0:3, 0:])
    ax_data.set_title(r"{}".format(seg.name.replace("_", "\_")))
    filtTime = np.arange(0, np.size(seg.Ia), 1, dtype=np.float32) / seg.samplerate
    assert (
        len(filtTime) > 0
    )  # gets effed when we have very small segments with the 1e40 thing.
    ax_data.plot(
        filtTime, seg.Ia, label="Filt", marker=",", linestyle="", color="darkblue"
    )
    ax_data.axhline((mean - dev), color="r", alpha=0.5)
    ax_data.axhline((mean + dev), color="r", alpha=0.5)
    ax_data.axhline(mean, color="r")
    ax_data.grid(True, which="both", ls="-", color="0.65")
    ax_data.set_xlabel(r"T [$s$]")
    ax_data.set_ylabel(r"$I_{0}$" + " [{sunit}]".format(sunit=seg.unit))
    ax_data.set_xlim(filtTime[0], filtTime[-1])

    if rc.args.fitTransients:
        try:
            ax_data.plot(
                filtTime, seg.fit_current, "g-", linewidth=2, label="Fitted Curve"
            )
            ax_data.annotate(
                "fit end value: {sx:.3f}{sunit}".format(
                    sunit=seg.unit, sx=seg.fit_current_conv
                ),
                xy=(0, 1.2),
                xycoords="axes fraction",
            )
            seed = "fit decay: "
            seed = seed + r" ${0:.3f}$ s;".format(seg.fit_decay_time1)
            seed = seed + r" ${0:.3f}$ s;".format(seg.fit_decay_time2)
            ax_data.annotate(seed, xy=(0.225, 1.2), xycoords="axes fraction")
            del seed
            seed = "fit prefactors: "
            seed = seed + r" ${s0:.3f}$ {sunit}".format(
                sunit=seg.unit, s0=seg.fit_prefactorA
            )
            seed = seed + r" ${s0:.3f}$ {sunit}".format(
                sunit=seg.unit, s0=seg.fit_prefactorC
            )
            ax_data.annotate(seed, xy=(0.225, 1.5), xycoords="axes fraction")
            ax_data.annotate(
                "fit standard error: {sx:.3f}{sunit}".format(
                    sunit=seg.unit, sx=seg.fit_error
                ),
                xy=(0.45, 1.2),
                xycoords="axes fraction",
            )
            ax_data.annotate(
                "mean: {smean:.3f} $\pm {sdev:.3f}$ {sunit}".format(
                    smean=mean, sdev=dev, sunit=seg.unit
                ),
                xy=(0.65, 1.2),
                xycoords="axes fraction",
            )
            ax_data.annotate(
                "$V_{AC}$ = " + str(seg.VAC) + " mV",
                xy=(0.9, 1.2),
                xycoords="axes fraction",
            )
        except ValueError as e:
            logging.warning("No fit convergence on first attempt.")
            print(e)

    if isinstance(seg.baseline, np.ndarray):
        ax_data.plot(filtTime, seg.baseline, label="baseline", marker="", linestyle="-", linewidth=0.5, color="k")

    if not np.isnan(np.array(seg.gate_current, dtype=np.float32)).any():
        ax_gate_cur = plt.subplot(gs[2:4, 0:])
        ax_gate_cur.grid(True, which="both", ls="-", color="0.65")
        ax_gate_cur.set_ylabel(r"$I_{G}$ " + "[{sunit}]".format(sunit=seg.unit))
        ax_gate_cur.plot(
            seg.gate_time,
            seg.gate_current,
            linestyle="",
            marker=".",
            color="red",
            label="I",
        )
        try:
            ax_gate_cur.plot(
                seg.gate_time,
                seg.fit_gate_current,
                "g-",
                linewidth=2,
                label="Fitted Curve",
            )
            ax_gate_cur.annotate(
                "fit end value: {sx:.3f}{sunit}".format(
                    sunit=seg.unit, sx=seg.fit_gate_current_conv
                ),
                xy=(0, 1.15),
                xycoords="axes fraction",
            )
            seed = "fit decay: "
            seed = seed + r" ${0:.3f}$ s;".format(seg.fit_gate_decay_time)
            ax_gate_cur.annotate(seed, xy=(0.225, 1.15), xycoords="axes fraction")
            ax_gate_cur.annotate(
                "fit standard error: {sx:.3f}{sunit}".format(
                    sunit=seg.unit, sx=seg.fit_gate_error
                ),
                xy=(0.65, 1.15),
                xycoords="axes fraction",
            )
        except ValueError as e:
            logging.warning("No fit convergence on second attempt.")
            print(e)

        ax_gate = ax_gate_cur.twinx()
        ax_gate.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useOffset=False)
        )
        ax_gate.set_ylabel(r"$V_{AG}$ [$mV$]")
        ax_gate.plot(seg.gate_time, seg.VAG, color="blue", label="V")
    # Below crashes run when fit dosen't converge.  Don't know why...
    #        ax_gate.set_xlim(seg.gate_time[0], seg.gate_time[-1])
    if not np.isnan(np.array(seg.gate_current, dtype=np.float32)).any():
        ax_psd = plt.subplot(gs[4:, 0:-2])
        ax_noise = plt.subplot(gs[4:, -2:])
    else:
        ax_psd = plt.subplot(gs[3:, 0:-2])
        ax_noise = plt.subplot(gs[3:, -2:])
    ax_psd.loglog(seg.psd_freq, seg.psd_cur, "-", color="darkblue", linewidth=0.5)
    ax_psd.set_ylim([1e-4, 1.5e1])
    ax_psd.set_xlim(1e1, np.ceil(seg.samplerate / 4.0))
    ax_psd.set_xlabel(r"Frequency [$Hz$]")
    ax_psd.set_ylabel(r"PSD [${sunit}^2/Hz$]".format(sunit=seg.unit))
    ax_psd.grid(True, which="both", ls="-", color="0.65")

    # IRMS cummulative noise plot
    ax_noise.loglog(
        seg.psd_freq,
        np.sqrt(integrate.cumtrapz(seg.psd_cur, seg.psd_freq, initial=0)),
        "-",
        color="darkblue",
        linewidth=1.0,
    )
    ax_noise.set_xlabel(r"F [$Hz$]")
    ax_noise.set_ylabel(r"$I_{RMS}$ " + "[{sunit}]".format(sunit=seg.unit))
    ax_noise.set_xlim(1e1, np.ceil(seg.samplerate / 4.0))
    ax_noise.grid(True, which="both", ls="-", color="0.65")
    ax_noise.xaxis.set_major_locator(LogLocator(base=10))

    # Format figure and save
    fig.savefig(megasavepath, dpi=600, bbox_inches="tight")
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    gc.collect()
    return


def plot_overview(seg, rc):
    """
    Default plot for an Slice().

    Parameters
    ----------
    experiment : Slice()
    The experiment to plot.
    rc :  runConstants
    RC element.

    Returns
    -------
    None
    Will write a file.
        
    """
    logging.info("Generating overviewplot of " + seg.name)
    mean = seg.cur_mean
    dev = seg.cur_std
    sampleRate = seg.samplerate
    totalTime = len(seg.Ia) * sampleRate
    stem = import_utils.get_stem(seg)
    plotmegaFileName = "{s1}_{s3:.0f}mV_overview.png".format(s1=stem, s3=seg.vac)
    megasavepath = rc.resultsPath / plotmegaFileName
    fig = plt.figure(figsize=(12, 4), constrained_layout=False)
    gs = matplotlib.gridspec.GridSpec(1, 1, hspace=1.2, wspace=0.7, figure=fig)
    ax_data = plt.subplot(gs[0:, 0:])
    ax_data.set_title(r"{}".format(seg.name))
    filtTime = np.arange(0, np.size(seg.Ia), 1, dtype=np.float32) / seg.samplerate
    ax_data.plot(
        filtTime, seg.Ia, label="Filt", marker=",", linestyle="", color="darkblue"
    )
    ax_data.axhline((mean - dev), color="r", alpha=0.5)
    ax_data.axhline((mean + dev), color="r", alpha=0.5)
    ax_data.axhline(mean, color="r")
    ax_data.grid(True, which="both", ls="-", color="0.65")
    ax_data.set_xlabel(r"T [$s$]")
    ax_data.set_ylabel(r"$I_{0}$" + " [{sunit}]".format(sunit=seg.unit))
    ax_data.set_xlim(filtTime[0], filtTime[-1])

    # Format figure and save
    fig.savefig(megasavepath, dpi=300, bbox_inches="tight")
    # Garbage collecion
    fig.clear()
    plt.close(fig)
    return


def plot_single_events(events_frame, seg, rc):
    """
    Default plot for a single event, marked with means deviations, and actual event width.

    Parameters
    ----------
    events_frame : list
        the events_frame to look at.
    seg : Slice()
        The Slice() we care about.
    rc :  runConstants
        RC element.

    Returns
    -------
    None
        Will write a file.

    """
    # plots single event traces, cutoff and baseline values for each event found.
    time = np.arange(0, np.size(seg.Ia), 1) / seg.samplerate
    pairs = []
    for i in events_frame.itertuples():
        pairs.append([int(i.EVENTSTARTIDX), int(i.EVENTSTOPIDX)])
    numberOfEvents = len(pairs)

    data = seg.Ia
    if not math.isnan(rc.args.detrend):
        data = (
            data
            - filter_utils.detrend_data(
                seg.Ia, int(np.floor(seg.samplerate * rc.args.detrend))
            )
            + seg.cur_mean
        )

    f = plt.figure(figsize=(8, 6))
    gs = matplotlib.gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    for index, event_index in enumerate(pairs):

        overhang = int(np.floor((event_index[1] - event_index[0]) / 4.0))
        # check whether index out of bounds
        if event_index[0] - overhang < 0:
            startIdx = 0
        else:
            startIdx = event_index[0] - overhang

        if event_index[1] + overhang > np.size(seg.Ia) - 1:
            endIdx = -1
        else:
            endIdx = event_index[1] + overhang
        ax = plt.subplot(gs[0, 0])
        ax.plot(time[startIdx:endIdx], data[startIdx:endIdx], linewidth=0.7, alpha=0.6)
        ax.axvspan(
            time[event_index[0]],
            time[event_index[1]],
            ymin=0,
            ymax=1,
            color="r",
            alpha=0.2,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index],
            xmin=time[startIdx],
            xmax=time[event_index[0]],
            color="r",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index]
            - float(events_frame.EVENTTHRESHOLD[index]),
            xmin=time[startIdx],
            xmax=time[event_index[0]],
            color="r",
            linestyle=":",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index]
            + float(events_frame.EVENTTHRESHOLD[index]),
            xmin=time[startIdx],
            xmax=time[event_index[0]],
            color="r",
            linestyle=":",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index],
            xmin=time[event_index[1]],
            xmax=time[endIdx],
            color="r",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index]
            - float(events_frame.EVENTTHRESHOLD[index]),
            xmin=time[event_index[1]],
            xmax=time[endIdx],
            color="r",
            linestyle=":",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index]
            + float(events_frame.EVENTTHRESHOLD[index]),
            xmin=time[event_index[1]],
            xmax=time[endIdx],
            color="r",
            linestyle=":",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index],
            xmin=time[event_index[0]],
            xmax=time[event_index[1]],
            color="g",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index]
            - (
                float(
                    events_frame.EVENTTHRESHOLD[index]
                    / events_frame.PDF[index]
                    * events_frame.PDFREVERSAL[index]
                )
            ),
            xmin=time[event_index[0]],
            xmax=time[event_index[1]],
            color="g",
            linestyle=":",
            alpha=0.6,
        )
        ax.hlines(
            y=events_frame.EVENTBASELINE[index]
            + (
                float(
                    events_frame.EVENTTHRESHOLD[index]
                    / events_frame.PDF[index]
                    * events_frame.PDFREVERSAL[index]
                )
            ),
            xmin=time[event_index[0]],
            xmax=time[event_index[1]],
            color="g",
            linestyle=":",
            alpha=0.6,
        )
        name = "{s1}_event{s4:0{width}}_{s5}".format(
            s1=seg.dfIndex(),
            s4=index,
            s5=events_frame.EVENTTYPE[index],
            width=int(math.ceil(math.log(numberOfEvents + 1, 10))),
        )
        ax.set_title("\n".join(wrap(str(name))))
        ax.set_ylabel(r"$I_{A}$ " + "[{sunit}]".format(sunit=seg.unit))
        ax.set_xlabel(r"$T$ [$s$]")
        ax.annotate(
            "even dwell time {s0:.1e}us, event depth {s1:.1f}{sunit}".format(
                sunit=seg.unit,
                s0=events_frame.EVENTDWELLTIME[index],
                s1=events_frame.EVENTDEPTH[index],
            ),
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )
        plotFileName = rc.resultsPath / "events/singleEventPlots" / (name + ".png")
        f.savefig(plotFileName, bbox_inches="tight")
        f.clear()
    plt.close(f)
    return


def plot_events(events_frame, seg, rc):
    """
    This function slices the data into rc.plotidealevents [s] second long slices and draws an idealized event / data trace of all events in the respective slice.

    Parameters
    ----------
    events_frame : Pandas DF.
        The events in the segment
    seg : Segment()
        The segment these supposedly came from
    rc :  RunConstants()
        RunConstants object

    Returns
    -------
    None
        Will write a file.

    """
    # This function slices the data into rc.plotidealevents [s] second long slices and draws an idealized event / data trace of all events in the respective slice.
    time = np.arange(0, np.size(seg.Ia), 1) / seg.samplerate
    stem = import_utils.get_stem(seg)
    data = seg.Ia
    if not math.isnan(rc.args.detrend):
        data = (
            data
            - filter_utils.detrend_data(
                seg.Ia, int(np.floor(seg.samplerate * rc.args.detrend))
            )
            + seg.cur_mean
        )

    plotLengthSec = rc.args.plotidealevents
    plotLength = int(np.floor(seg.samplerate * plotLengthSec))
    dataLength = len(data)
    numOfPlots = np.int(np.ceil(dataLength / plotLength))
    minData = np.min(data)
    maxData = np.max(data)
    yLimits = [minData - (maxData - minData) * 0.1, np.max(data) + (maxData - minData) * 0.1]

    logging.info("Generating ideal event plots of " + seg.name)
    for i in range(0, numOfPlots):
        fig, ax = plt.subplots(figsize=(30, 6))
        startIdx = plotLength * i
        endIdx = plotLength * (i + 1)
        if endIdx >= dataLength:
            endIdx = dataLength - 1
            logging.info(
                "reset length for {s1}_idealEvent_{s5:0{width}}".format(
                    s1=seg.dfIndex(),
                    s5=i,
                    width=int(math.ceil(math.log(numOfPlots + 1, 10))),
                )
            )
            if (endIdx - startIdx) < 100:
                logging.info(
                    "skip last part less than 100 points {s1}_idealEvent_{s5:0{width}}".format(
                        s1=seg.dfIndex(),
                        s5=i,
                        width=int(math.ceil(math.log(numOfPlots + 1, 10))),
                    )
                )
                return

        # plot data
        ax.plot(
            time[startIdx:endIdx],
            data[startIdx:endIdx],
            label="data",
            marker=",",
            linewidth=0.1,
            markeredgecolor="darkblue",
            color="blue",
        )

        ax.grid(True, which="both", ls="-", color="0.65")
        ax.set_ylabel(r"$I_{A}$ " + "[{sunit}]".format(sunit=seg.unit))
        ax.set_xlabel(r"$T$ [$s$]")
        ax.set_xlim(time[startIdx], time[endIdx])
        ax.set_ylim(yLimits)

        eventIdxs = [startIdx]
        eventValues = [0]
        j = 0
        lastBaseline = 0
        # add events to trace
        for k in events_frame[
            events_frame.EVENTSTARTIDX.between(startIdx, endIdx)
        ].itertuples():
            lastBaseline = k.EVENTBASELINE
            if k.EVENTSTARTIDX != 0:
                eventIdxs.extend(
                    [
                        int(k.EVENTSTARTIDX) - 1,
                        int(k.EVENTSTARTIDX),
                        int(k.EVENTSTOPIDX),
                        int(k.EVENTSTOPIDX) + 1,
                    ]
                )
                if j == 0:
                    eventValues[0] = k.EVENTBASELINE
                eventValues.extend(
                    [
                        k.EVENTBASELINE,
                        k.EVENTBASELINE + k.EVENTDEPTH,
                        k.EVENTBASELINE + k.EVENTDEPTH,
                        k.EVENTBASELINE,
                    ]
                )
            else:
                eventIdxs.extend(
                    [
                        int(k.EVENTSTARTIDX),
                        int(k.EVENTSTARTIDX),
                        int(k.EVENTSTOPIDX),
                        int(k.EVENTSTOPIDX) + 1,
                    ]
                )
                if j == 0:
                    eventValues[0] = k.EVENTBASELINE + k.EVENTDEPTH
                eventValues.extend(
                    [
                        k.EVENTBASELINE + k.EVENTDEPTH,
                        k.EVENTBASELINE + k.EVENTDEPTH,
                        k.EVENTBASELINE + k.EVENTDEPTH,
                        k.EVENTBASELINE,
                    ]
                )
            j = j + 1
        eventIdxs.extend([endIdx])
        eventValues.extend([lastBaseline])
        if eventValues[0] != 0:
            ax.plot(
                time[eventIdxs],
                eventValues,
                label="idealized event data",
                marker=",",
                linewidth=1,
                markeredgecolor="red",
                color="red",
            )
        name = "{s1}_idealEvent_{s5:0{width}}".format(
            s1=seg.dfIndex(), s5=i, width=int(math.ceil(math.log(numOfPlots + 1, 10)))
        )
        ax.set_title(str(name))
        fig.legend()
        fig.savefig(
            rc.resultsPath / "events" / (name + ".png"),
            dpi=70,
            format="png",
            bbox_inches="tight",
        )
        ax.clear()
        fig.clear()
        plt.close(fig)
        if eventValues[0] != [0] and rc.args.exportEventTrace:
            tracePath = rc.resultsPath / "events" / "traceData"
            tracePath.mkdir(parents=True, exist_ok=True)
            export_utils.export_event_trace_chunks(
                time[startIdx:endIdx],
                data[startIdx:endIdx],
                time[eventIdxs],
                eventValues,
                tracePath / (name + ".csv"),
            )

    return


def plot_events_scatterplot(allEventLog, writeFile, seg):
    """
    Plot event statistics -- this is the thing that was originally inlined up in the main loop.

    Parameters
    ----------
    allEventLog : Pandas DF.
        the events to scatterplot
    writeFile : Path
        Where to write this.
    seg :  Segment()
        the segment this came from

    Returns
    -------
    None
        Will write a file.

    """
    sns.set()
    sns.set_style("whitegrid")

    figsns, axsns = plt.subplots(figsize=(8, 6))
    seaborn_plot = sns.scatterplot(
        x="EVENTDWELLTIME", y="EVENTDEPTH", hue="EVENTTYPE", data=allEventLog, ax=axsns
    )
    seaborn_plot.set_xlim(auto=True)
    seaborn_plot.set_ylim(auto=True)
    seaborn_plot.set_xscale("log")
    seaborn_plot.set(
        xlabel="Event duration [$\mu sec$]",
        ylabel="Event depth  [{sunit}]".format(sunit=seg.unit),
    )
    seaborn_plot.set_title("\n".join(wrap(str(writeFile.stem))))
    seaborn_plot.annotate(
        "Up: {s0}, Down: {s1} events".format(
            s0=allEventLog[allEventLog.EVENTTYPE == "up"].EVENTDWELLTIME.count(),
            s1=allEventLog[allEventLog.EVENTTYPE == "down"].EVENTDWELLTIME.count(),
        ),
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    figsns.savefig(str(writeFile) + "_eventsplot.png", dpi=600)
    figsns.clear()
    plt.close(figsns)
    return


# plot of plotLengthSec s raw data (filtered but not filter_utils.baselinePlotDecimated) to check real trace
def data_plot(seg, rc, avg, sigmaValue, PDF, dataSetMinLengthAv, dataSet):
    """
    Jakob?  TODO does anybody use this?

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
    None
        Will write a file.

    """
    # event finder undecimated plot
    plotLengthSec = 1.0
    plotLength = int(np.floor(seg.samplerate * plotLengthSec))
    dataLength = len(dataSet)
    numOfPlots = 0
    if rc.args.plotundecimated:
        numOfPlots = np.int(np.floor(dataLength / seg.samplerate))

    yLimits = [np.min(dataSet), np.max(dataSet)]

    for i in range(0, numOfPlots):
        fig, ax = plt.subplots(figsize=(30, 6))
        startIdx = plotLength * i
        endIdx = plotLength * (i + 1)
        if endIdx > dataLength:
            endIdx = dataLength
        time = np.arange(startIdx, endIdx, 1) / seg.samplerate

        # plot data
        ax.plot(
            time,
            dataSet[startIdx:endIdx],
            label="data",
            marker=",",
            linewidth=0.1,
            markeredgecolor="darkblue",
            color="blue",
        )

        # plot std
        ax.fill_between(
            time,
            avg[startIdx:endIdx] - (sigmaValue * PDF),
            avg[startIdx:endIdx] + (sigmaValue * PDF),
            facecolor="orange",
            alpha=(0.4),
            label="event threshold",
        )

        # plot baseline average
        ax.plot(
            time,
            avg[startIdx:endIdx],
            label="baseline average",
            linewidth=0.8,
            color="red",
        )

        # plot forward moving average (min event length)
        ax.plot(
            time,
            dataSetMinLengthAv[startIdx:endIdx],
            label="FW moving average",
            linewidth=0.5,
            color="green",
        )

        ax.grid(True, which="both", ls="-", color="0.65")
        ax.set_ylabel(r"$I_{A}$ " + "[{sunit}]".format(sunit=seg.unit))
        ax.set_xlabel(r"$T$ [$s$]")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(yLimits)
        name = "{s0}_{s4:.0E}HzData_{s5:0{width}}".format(
            s0=seg.dfIndex(),
            s4=rc.args.filter,
            s5=i,
            width=int(math.ceil(math.log(numOfPlots + 1, 10))),
        )
        ax.set_title(str(name))
        fig.legend()
        fig.savefig(
            rc.resultsPath / "events" / (name + ".png"),
            dpi=200,
            format="png",
            bbox_inches="tight",
        )
        ax.clear()
        fig.clear()
        plt.close(fig)

    return


def plot_detrend(ax0_longtrace, seg, decer, stride, rc):
    """
    Jakob?

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
    None
        Will write a file.

    """
    # event finder detrend line plot
    time, data = filter_utils.baselinePlotDecimate(
        filter_utils.detrend_data(
            seg.Ia, int(np.floor(seg.samplerate * rc.args.detrend))
        ),
        seg.samplerate,
        decer,
    )
    label_name = "low pass detrend"
    ax0_longtrace.plot(
        time,
        data,
        label=label_name,
        marker=".",
        linestyle="-",
        color="black",
        linewidth=0.8,
        alpha=(0.7),
        markersize=1,
    )
    del time, data
    return ax0_longtrace


def plot_endpass(ax0_longtrace, passes, dataSet, dataRunner, samplerate, decer):
    """
    Jakob?

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
    None
        Will write a file.

    """
    # event finder baseline plot - last pass
    ax0_longtrace.annotate(
        "passes: " + r"${0:d}$".format(passes + 1),
        xy=(0.05, 1.01),
        xycoords="axes fraction",
    )
    time, data = filter_utils.baselinePlotDecimate(dataRunner, samplerate, decer)
    label_name = "baseline" + str(passes)
    ax0_longtrace.plot(
        time, data, label=label_name, marker=",", linewidth=0.3, color="salmon"
    )
    time, data = filter_utils.baselinePlotDecimate(dataSet, samplerate, decer)
    label_name = "dataSet" + str(passes)
    ax0_longtrace.plot(
        time,
        data,
        label=label_name,
        marker=".",
        linestyle="None",
        color="red",
        alpha=(0.1),
        markersize=1,
    )
    ax0_longtrace.set_xlim(time[0], time[-1])
    del time, data
    return ax0_longtrace


def plot_pass(
    ax0_longtrace, passes, avg, sigmaValue, passcount, samplerate, decer, PDF, seg
):
    """
    Jakob?

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
    None
        Will write a file.

    """
    # event finder baseline plot - normal pass
    ax0_longtrace.annotate(
        "P{s0:d} std: {s2:.2e}[{sunit}]".format(
            s0=passes, s2=np.mean(sigmaValue), sunit=seg.unit
        ),
        xy=(0.25 + passes * 0.7 / passcount, 0.01),
        xycoords="axes fraction",
    )
    time, data = filter_utils.baselinePlotDecimate(avg, samplerate, decer)
    ax0_longtrace.plot(
        time, data, label="average" + str(passes), marker=",", linewidth=0.5
    )
    ax0_longtrace.fill_between(
        time,
        data - (sigmaValue * PDF),
        data + (sigmaValue * PDF),
        facecolor="blue",
        alpha=(0.07),
    )
    del time, data
    return ax0_longtrace
