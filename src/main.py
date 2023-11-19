import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
import distutils.dir_util
import shutil
from pathlib import Path
import gc
import csv
import os
import psutil
import random
import multiprocessing
from datetime import datetime
from datetime import timedelta
import time

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# import custom utils
sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))
import import_utils
import plot_utils
import filter_utils
import export_utils
import event_utils
import setENV_utils
import poreAnalysis_utils
from data_classes import Experiment
from data_classes import rc
import data_classes
import parser_utils

# set up run constants
npersegPARAM = 240  # cut the whole thing in to this many segments
noverlapPARAM = 1.2  # 1.1 indicates 10% overlap
seg_front_crop = 5  # seconds to cut off front TODO, ok , what the fuck.  This slicer is very sensitive to this
seg_back_crop = 1  # seconds to cut off back
allowed_time_lag = timedelta(
    seconds=60
)  # You must start the Benchvue within this many seconds of starting the Chimera.

resultsPath = Path.cwd().parent / "DataAndResults/results"
resultsPath.mkdir(parents=True, exist_ok=True)
dataPath = Path()

if import_utils.check_dataPath(resultsPath.parent):
    dataPath = import_utils.get_dataPath(resultsPath.parent)
else:
    # BUG -- this is ripe for fuckery.  We should just error out on no data.txt
    logging.warning(
        "No data.txt file to read found in {s1}. Assuming data stored in {s1}/dump folder.".format(
            s1=str(resultsPath.parent)
        )
    )
    dataPath = resultsPath.parent / "dump"


def get_exp_RAM(metaData):
    """
    How much RAM do you need? Estimates the amount of RAM I need to launch this experiment read. Estimates, crudely for now, the amount of RAM I need to launch this experiment read, returns bytes  BUG -- -this is way off for smFET data, since we were looking at Chimera data for this, so this needs to be fixed.

    Parameters
    ----------
    experiment : Exp()
    Segment to guess RAM use of.

    Returns
    -------
    buffer : float
        How much RAM you need for this Experiment(), in bytes
    """
    fudge_factor = (
        2  # this can vary to about 1.2 to 2 -- 2 is safest, based on looking at runs.
    )
    return fudge_factor * 20 * 1024 * 1024 * len(metaData["DATA_FILES"])


def get_seg_RAM(seg):
    """
    How much RAM do you need? Estimates the amount of RAM I need to launch this seg read.

    Parameters
    ----------
    seg : slice()
    Segment to guess RAM use of.

    Returns
    -------
    buffer : float
        How much RAM you need for this segment, in bytes.
    """
    file_length = import_utils.file_length_sec(seg.exp.metaData["DATA_FILES"][0])
    bytes_per_point = 2  # uint is 16 bits
    fudge_factor = 2
    # fudge_factor * kilobytes per megabyte * megabytes per gigabyte * bytes_per_point * points_per_file * len(files)
    return (
        fudge_factor
        * bytes_per_point
        * file_length
        * len(seg.exp.metaData["DATA_FILES"])
    )


def get_seg_RAM_pedantically(seg):
    """
    How much RAM do you need? Estimates RAM I need to load the segment.  Slow, because it does IO on reading the files explicitly.

    Parameters
    ----------
    seg : slice()
    Segment to guess RAM use of.

    Returns
    -------
    buffer : float
        How much RAM you need for this segment, in bytes.
    """
    keepers = []
    prev_file = ""
    for fileN in seg.exp.metaData["DATA_FILES"]:
        file_time_start = import_utils.read_mat_Chimera(fileN)["mytimestamp"]
        file_length = import_utils.file_length_sec(fileN)
        if file_time_start > seg.transition[0]:
            # entered stuff to add to keepers...
            if prev_file != "":  # if this is the first one, add the prev one...
                keepers.append(prev_file)
                prev_file = ""
            # you are right of the first transition, add it
            keepers.append(fileN)
            # are you right of the second transition?
            if file_time_start > seg.transition[1]:
                keepers.pop()
        else:  # before start of ones to add
            prev_file = fileN
    fudge_factor = 1
    return fudge_factor * 20 * 1024 * 1024 * len(keepers)


def ram_buffer(avaliable, needed):
    """
    How much RAM do you need?

    Parameters
    ----------
    availiable : float
        How much RAM is avaliable on this machine.
    needed : float
        How much RAM is you need to finish this computation

    Returns
    -------
    buffer : float
        How much RAM you would have free, should you continue.
    """
    return avaliable - 1.2 * needed


def handle_postattack_iv(experiment, exList, rc):
    """
    Handles post-attack IV names here, and plots the overlay.  The thing to do is to match the names, but ignoring the date timestamps in the front; this is annoying, and somewhat cumbersome...

    This entire situation has the following assumptions:
    The file name designators are _PostAttack_ and _PreAttack_
    The IV file name designator is _IV_Experiment

    Parameters
    ----------
    experiment : Experiment()
        The name of the post-attack experiment
    exList : list
        The list of experiments to check.
    rc : rc()
        RunConstants

    Returns
    -------
    None.  Will write an image file.
    """
    assert "PostAttack" in experiment.name
    old_name = experiment.name.replace("PostAttack", "PreAttack").split("Batch")[1]
    for key in exList:
        if key.split("Batch")[1] == old_name:
            old_exp = Experiment(exList[key], rc)
    old_I, old_V, old_sigma = data_classes.get_smFET_IV(old_exp)
    del old_exp
    I, V, sigma = data_classes.get_smFET_IV(experiment)
    plot_utils.plot_smFET_IV_overlay(
        rc, experiment, I, V, sigma, old_I, old_V, old_sigma
    )


def handle_lynalls_iv(experiment, exList, rc):
    """
    Handles post-attack IV names here, and plots the overlay.  The thing to do is to match the names, but ignoring the date timestamps in the front; this is annoying, and somewhat cumbersome...

    This entire situation has the following assumptions:
    The file name designators are pre_side_wall, post_side_wall and post_probe_attachment
    The IV file name designator is _IV_Experiment

    Parameters
    ----------
    experiment : Experiment()
        The name of the first experiment
    exList : list
        The list of experiments to check.
    rc : rc()
        RunConstants

    Returns
    -------
    None.  Will write an image file.
    """
    name1 = experiment.name.split("Batch")[1]
    name2 = experiment.name.replace("pre_side_wall", "post_side_wall").split("Batch")[1]
    name3 = experiment.name.replace("pre_side_wall", "post_probe_attachment").split(
        "Batch"
    )[1]
    # BUG, if lynall changes his coding tagging scheme, these will all have to change.
    listoflists = []
    old_I, old_V, old_sigma = data_classes.get_smFET_IV(experiment)
    listoflists.append([old_V, old_I, old_sigma, "pre_side_wall"])
    plot_utils.plot_smFET_IV_multipleoverlay(rc, experiment, listoflists)
    for key in exList:
        if key.split("Batch")[1] == name2:
            exp2 = Experiment(exList[key], rc)
            old_I, old_V, old_sigma = data_classes.get_smFET_IV(exp2)
            listoflists.append([old_V, old_I, old_sigma, "post_side_wall"])
    plot_utils.plot_smFET_IV_multipleoverlay(rc, experiment, listoflists)
    listoflists = []
    old_I, old_V, old_sigma = data_classes.get_smFET_IV(experiment)
    listoflists.append([old_V, old_I, old_sigma, "pre_side_wall"])
    for key in exList:
        if key.split("Batch")[1] == name2:
            exp2 = Experiment(exList[key], rc)
            old_I, old_V, old_sigma = data_classes.get_smFET_IV(exp2)
            listoflists.append([old_V, old_I, old_sigma, "post_side_wall"])
        if key.split("Batch")[1] == name3:
            exp3 = Experiment(exList[key], rc)
            old_I, old_V, old_sigma = data_classes.get_smFET_IV(exp3)
            listoflists.append([old_V, old_I, old_sigma, "post_probe_attachment"])
    plot_utils.plot_smFET_IV_multipleoverlay(rc, experiment, listoflists)

    listoflists = []
    for key in exList:
        if key.split("Batch")[1] == name2:
            exp2 = Experiment(exList[key], rc)
            old_I, old_V, old_sigma = data_classes.get_smFET_IV(exp2)
            listoflists.append([old_V, old_I, old_sigma, "post_side_wall"])
    for key in exList:
        if key.split("Batch")[1] == name3:
            exp3 = Experiment(exList[key], rc)
            old_I, old_V, old_sigma = data_classes.get_smFET_IV(exp3)
            listoflists.append([old_V, old_I, old_sigma, "post_probe_attachment"])
    plot_utils.plot_smFET_IV_multipleoverlay(rc, experiment, listoflists)


def exp_master_func(master_arg):
    """
    This is the master function that pulls in a Experiment() and returns Slices()

    Parameters
    ----------
    master_arg : list() of length 2
        [metaData array for this run, statsDF output variable]

    Returns
    -------
    None
        This will call it's own Slice()'s master funcs, and output the results to the shared variable statsDF.
    """
    assert len(master_arg) == 2
    metaData = master_arg[0]
    statsDF = master_arg[1]
    try:
        setENV_utils.setENV_threads(int(1))
        assert len(metaData) == 5
        exList = metaData[0]
        ex = metaData[1]
        rc = metaData[2]
        args = metaData[3]
        semaphore = metaData[4]
        experiment = Experiment(exList[ex], rc)
        needed_RAM = get_exp_RAM(exList[ex])
        assert (
            psutil.virtual_memory().total - needed_RAM > 0
        ), "You may be running this on a machine that does not have enough RAM under any conditons to do this."

        if args.ramcheck:
            free_RAM = psutil.virtual_memory().available
            ram_check = ram_buffer(free_RAM, needed_RAM)
            logging.info(
                "Before I did a thing, estimated free ram remaining would be "
                + str(ram_check)
            )

            while ram_check < 0 or psutil.virtual_memory().percent > 70:
                logging.info("Slept because of RAM on exp_master on " + experiment.name)
                logging.info("Overflow is " + str(ram_check))
                logging.info("RAM load is " + str(psutil.virtual_memory().percent))
                time.sleep(
                    5 * random.random() + 10
                )  # sleep for some random time between 10 and 15 seconds
                gc.collect()
                free_RAM = psutil.virtual_memory().available
                ram_check = ram_buffer(free_RAM, needed_RAM)
            else:
                RAM_before = psutil.virtual_memory().used
                RAM_after = 0
        semaphore.acquire()
        logging.info("Start experiment master for " + experiment.name)
        if "_IV_Experiment" in experiment.name:
            I, V, sigma = data_classes.get_smFET_IV(experiment)
            plot_utils.plot_smFET_IV(rc, experiment, I, V, sigma)
            if "_PostAttack_" in experiment.name:
                logging.info(
                    "Doing the IV plot overlay thing on " + str(experiment.name)
                )
                handle_postattack_iv(experiment, exList, rc)
            if "pre_side_wall" in experiment.name:
                logging.info(
                    "Doing the IV plot overlay thing on " + str(experiment.name)
                )
                handle_lynalls_iv(experiment, exList, rc)
        if args.expplot:
            experiment.load()
            plot_utils.plot_experiment(experiment, rc)
            if args.ramcheck:
                RAM_after = max(RAM_after, psutil.virtual_memory().used)
            gc.collect()
        if args.export:
            experiment.load()
            if args.ramcheck:
                RAM_after = max(RAM_after, psutil.virtual_memory().used)
            export_utils.export_experiment(experiment)
            if args.ramcheck:
                RAM_after = max(RAM_after, psutil.virtual_memory().used)
            gc.collect()
        slices = experiment.schneiden()
        if not (args.segplot or args.findevents) or isinstance(
            experiment.metaData["BENCHVUE"], Path
        ):
            experiment.unload()
        semaphore.release()
        gc.collect()
        if args.ramcheck:
            logging.info("Experiment was " + experiment.name)
            logging.info(
                "Max RAM was "
                + str((RAM_after - RAM_before) / (1024 * 1024 * 1024))
                + " Gb"
            )
            # BUG -- this is not actually super telling, since it incorporates RAM usage from ALL processes that are running, and not just this execution of master_func.
            logging.info(
                "RAM guess was " + str(needed_RAM / (1024 * 1024 * 1024)) + " Gb"
            )
        if args.serial:
            for a_slice in slices:
                statsDF.append(slice_master_func([a_slice, args, rc]))
        else:
            out = []
            for a_slice in slices:
                out.append([a_slice, args, rc])
            jobs = []
            for work_item in out:
                slice_proc = multiprocessing.Process(
                    target=single_slice_master_func_wrapper,
                    args=(
                        work_item,
                        statsDF,
                        semaphore,
                    ),
                )
                slice_proc.start()
                jobs.append(slice_proc)
            for j in jobs:
                j.join()
        return
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
        exit()


def slice_master_func(metaData):
    """
    This is the master function that pulls in a Slice() and processes is to return Slice()-wide statistics, like noise, mean current, events, etc.

    Parameters
    ----------
    metaData : list
        metaData array for this run.

    Returns
    -------
    statsDF : dataFrame

        The line for the pandas dataframe that contains the selected stats on this run.
        BUG These are the columns of statsDF; where is the physical meaning of these documented?

        exp_name; Name of experiment

        timestamp; acquisition time

        slice_i; slice number

        SAMPLE; parsed from file name

        COMMENTS; Padres from filename

        conc; concentration of electrolyte

        electrolyte; electrolyte type

        VAC; applied anode/cathode voltage difference, for smFET Differential bias across tube

        vg; gate voltage

        CHANNEL; channel number (smFET only)

        I_gate; gate current (benchvue only)

        I_mean; average of measured anode/cathode current over entire time trace of segment

        I_std; standard deviation of anode/cathode current over entire time trace of segment

        fit_cur; settling Value of anode/cathode current from exponential fit t->inf

        fit_time1; Fast time constant anode/cathode current Fit

        Fit_time2; slow time constant anode/cathode current Fit

        fit_error; fitting error sum of residuals

        gate_fit_cur; settling Value of gate current from exponential fit t->inf

        gate_fit_time; time constant of exponential fit for gate current

        gate_fit_error; summ of residuals of gate current fitting

        CCHIP; appearant capacitance of Device From power spectrum noise model fit (F^2 noise)

        RCHIP;  appearant device resistance from powe spectrum noise model fit (thermal noise, constant term)

        I2NOISE-1; fitted ~f^-1 Shot noise coefficient in power spectrum itted Shot noise coefficient in power spectrum

        I2NOISE0; fitted ~f^0constant noise coefficient in power spectrum

        I2NOISE1; fitted ~f^1 Dielectric noise coefficient in power spectrum

        I2NOISE2;  fitted f^2 Capacitative noise coefficient in power spectrum

        unit;  unit of current [pA] or[nA]

        TOTALTIME; total duration of slice recording

        BENCHVUE; path to be benchvue file

        Temperature; temperature parsed from .param (smFET only)

        STRIDE; window size of moving average for event finder, in array size integer

        MINLENGTH; minimum event duration in [us]

        PDF; factor of trace standard deviation for event start detection

        PDFREVERSAL; factor of trace standard deviation for end start detection

        FILTERCUTOFF; low pass filter cutoff frequency

        SAMPLERATE; samplerate of current

        BASELINESTD; standard deviation of anode/cathode current of entire segment after event detection excluding events

        AVEVENTSTD; average standard deviation of anode/cathode current during events

        TOTALUPEVENTTIME; total duration spend in UP events (sum of dwell times UP)

        TOTALDOWNEVENTTIME; total duration per segment spend in down events (sum of dwell times DOWN)

        TOTALTRACETIME; total duration of segment

        P_UP; fraction of time spend in UP events (TOTALUPEVENTTIME / TOTALTRACETIME)

        P_DOWN; fraction of time spend in DOWN events (TOTALDOWNEVENTTIME / TOTALTRACETIME)

        DETREND; Flag for detrend option active (needed for event ploting)

        STOPBAND; Flag for stopband option active (needed for event ploting)

        NOTCH; Flag for notch filter active (needed for event ploting)

        NUMBEROFEVENTS; Total number of events found

        K_OFF_single; single exponential fitting coefficient of cumulative EVENT dwell time histogram N( tau <= t ) = A_OFF * exp( - t * K_OFF ), unit: [1/s]

        K_OFF_rchi_single; reduced chi value for single exponential fitting of cumulative EVENT dwell time histogram N( tau <= t ) = A_OFF * exp( - t * K_OFF )

        A_OFF; amplitude of single exponential fitting of cumulative EVENT dwell time histogram N( tau <= t ) = A_OFF * exp( - t * K_OFF ), unit: number

        K_OFF_single_1simga; error (1sigma confidence interval) of single exponential fitting coefficient for EVENT dwell time histogram N( tau <= t ) = A_OFF * exp( - t * K_OFF ), unit: [1/s]

        K_OFF_double_slow; slow exponential coefficient of double exponential fitting cumulative EVENT dwell time histogram N( tau <= t ) = AdF_OFF * exp( - t * K_OFF_double_fast ) + AdS_OFF * exp( - t * K_OFF_double_slow ), unit: [1/s]

        K_OFF_double_fast; fast exponential coefficient of double exponential fitting cumulative EVENT dwell time histogram N( tau <= t ) = AdF_OFF * exp( - t * K_OFF_double_fast ) + AdS_OFF * exp( - t * K_OFF_double_slow ), unit: [1/s]

        K_OFF_rchi_double; reduce chi value for double exponential fitting cumulative EVENT dwell time histogram N( tau <= t ) = AdF_OFF * exp( - t * K_OFF_double_fast ) + AdS_OFF * exp( - t * K_OFF_double_slow )

        AdS_OFF; amplitude of slow exponential of double exponential fitting cumulative EVENT dwell time histogram N( tau <= t ) = AdF_OFF * exp( - t * K_OFF_double_fast ) + AdS_OFF * exp( - t * K_OFF_double_slow ), unit: #

        AdF_OFF; amplitude of slow exponential of double exponential fitting cumulative EVENT dwell time histogram N( tau <= t ) = AdF_OFF * exp( - t * K_OFF_double_fast ) + AdS_OFF * exp( - t * K_OFF_double_slow ), unit: #

        K_OFF_double_slow_1simga; 1 sigma error / uncertainity for fitted value K_OFF_double_slow

        K_OFF_double_fast_1simga; 1 sigma error / uncertainity for fitted value K_OFF_double_fast

        K_OFF_stretchted; stretched exponential fitting (Weibull distribution) coefficient of cumulative EVENT dwell time histogram N( tau <= t ) = A_stretchted_OFF * exp( - (t * K_OFF_stretchted) ^ ALPHA), unit: [1/s]

        A_OFF_stretchted; amplitude of stretched exponential fitting (Weibull distribution) coefficient of cumulative EVENT dwell time histogram N( tau <= t ) = A_stretchted_OFF * exp( - (t * K_OFF_stretchted) ^ ALPHA), unit: #

        ALPHA_OFF_stretchted; alpha parameter of stretched exponential fitting (Weibull distribution) coefficient of cumulative EVENT dwell time histogram N( tau <= t ) = A_stretchted_OFF * exp( - (t * K_OFF_stretchted) ^ ALPHA), unit: none

        K_OFF_stretchted_rchi; reduce chi value for stretched exponential fitting (Weibull distribution) coefficient of cumulative EVENT dwell time histogram N( tau <= t ) = A_stretchted_OFF * exp( - (t * K_OFF_stretchted) ^ ALPHA),

        K_OFF_stretchted_1sigma; 1 sigma error / uncertainity for fitted value K_OFF_stretchted;

        K_OFF_stretched_apparent; apparent K value for  stretched exponential fitting (Weibull distribution) (Mean time of event duration) according to formula K_OFF_stretched_apparent = K_OFF_stretchted / gamma(1.0 + 1.0 / ALPHA)

        for *_ON_* the corresponding dwell time histogramm is based on the duration in between two subsequent events.

        for *_RISINGEDGE_* the corresponding dwell time histogramm is based on the duration from the start of one event to the start of the next event.

    """
    try:
        assert len(metaData) == 3
        seg = metaData[
            0
        ]  # well, here's a design error -- slice is a keyword in python...
        args = metaData[1]
        rc = metaData[2]
        setENV_utils.setENV_threads(int(1))
        if args.ramcheck:
            needed_RAM = get_seg_RAM(seg)
            free_RAM = psutil.virtual_memory().available
            if psutil.virtual_memory().total - needed_RAM > 0:
                logging.warning(
                    "You may be running this on a machine that does not have enough RAM under any condtions to do this."
                )
            logging.info(
                "Before I did a thing, estimated free ram remaining would be "
                + str(ram_buffer(free_RAM, needed_RAM))
            )
            ram_check = ram_buffer(free_RAM, needed_RAM)
            while ram_check < 0 or psutil.virtual_memory().percent > 70:
                logging.info("Slept becuase of RAM on slice_master on " + seg.name)
                logging.info("Overflow is " + str(ram_check))
                logging.info("RAM load is " + str(psutil.virtual_memory().percent))
                time.sleep(
                    5 * random.random() + 10
                )  # don't ask again for at least 10 seconds, no more than 15
                gc.collect()
                free_RAM = psutil.virtual_memory().available
                ram_check = ram_buffer(free_RAM, needed_RAM)
            else:  # ok, you're clear, and now you're running...
                RAM_before = psutil.virtual_memory().used
                RAM_after = 0
        logging.info("Start slice master for " + seg.name)
        seg.load()

        if args.preprocess:
            seg.correct_baseline()
            seg.save_preprocessed_data()

        if args.fitTransients:
            seg.make_fit_current()
            seg.make_fit_gate_current()

        # BUG: the issue here is again the convergence of the fitter for cases where the current is saturating.
        # Same -16000 in data_classes
        if args.fitPore and (seg.cur_mean > -16000 and seg.cur_mean < 16000):
            poreAnalysis_utils.calculateDeviceParameter(seg, rc)

        if args.segplot:
            seg.psd()
            plot_utils.plot_slice(seg, rc)

            logging.info("Segplot was " + seg.name)
            if args.ramcheck:
                RAM_after = max(RAM_after, psutil.virtual_memory().used)

            seg.unload_psd()

        if args.overviewplot:
            plot_utils.plot_overview(seg, rc)

        if args.export_slice:
            export_utils.export_slice(seg, args, rc)

        if args.findevents:
            if (
                not "_IV_Experiment_" in seg.name
            ):  # eventfinding on IV files from smfetdata is wrong since the biases change
                event_utils.find_events(seg, rc)

        if "none" not in args.hmm:
            seg.hmm_build_and_fit()
            seg.hmm_analyze()
            if args.saveHMM:
                seg.hmm_simulate_save()

            if args.readHMM:
                seg.hmm_read_simulate_save()

            if args.normalizeHMM:
                seg.hmm_normalized_save()

        if args.plotevents or not math.isnan(args.plotidealevents):
            plotFileName = seg.dfIndex() + "_events.csv"
            eventsPath = rc.resultsPath / "events" / plotFileName
            try:
                event_frame = pd.read_csv(eventsPath)
                if not math.isnan(args.plotidealevents):
                    plot_utils.plot_events(event_frame, seg, rc)

                elif args.plotevents:
                    plot_utils.plot_single_events(event_frame, seg, rc)

                if args.ramcheck:
                    RAM_after = max(RAM_after, psutil.virtual_memory().used)

            except IOError:
                logging.warning("No events log for " + seg.name)

        if args.kvalueanalysis:
            eventFileName = seg.dfIndex() + "_events.csv"
            eventsPath = rc.resultsPath / "events" / eventFileName
            try:
                event_frame = pd.read_csv(eventsPath)
                event_utils.event_dwell_time_analysis(event_frame, seg, rc)
                if args.ramcheck:
                    RAM_after = max(RAM_after, psutil.virtual_memory().used)
            except IOError:
                logging.warning("No events log for " + seg.name)

        if args.picklesegment:
            export_utils.save_segment(seg, rc.resultsPath / "pickledData")

        if "none" not in args.hmm:
            seg.hmm_unload()

        if args.ramcheck:
            RAM_after = max(RAM_after, psutil.virtual_memory().used)
        ##        assert (
        ##            import_utils.get_size(seg) < 4294967296
        ##        ), "If this is in a thread, it will fail on the 32-bit, 4Gb serializability thing.  TODO We may not need to do this after going away from Pool()s."
        if args.ramcheck:
            logging.info(
                "Max RAM was " + str((RAM_after - RAM_before) / (1024 * 1024)) + " Gb"
            )
            logging.info("RAM guess was " + str(needed_RAM / (1024 * 1024)) + " Gb")

        logging.info("Slice_master run complete, with stats: \n" + str(seg.statsDF))
        seg.unload()

        return seg.statsDF
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
        exit()


def assemble_data(experimentList, the_manager, the_semaphore, rc, args):
    """
    Assembles the data list argument that gets passed to master; just a single place to have the assembly done, since we test against this, and want to pull it in to pytest.

    Parameters
    ----------
    experimentList : Exp()
        the output of create_experimentList for the relevant hardware architecture.
    the_manager : mp.Manager()
        the managed multiproccessing object that handles the Semaphore.
    the_sempahore : mp.Semaphore()
        The Sempahore we pass around.

    Returns
    -------
    data : List of lists
        The input that master_func requires.
    """
    data = []

    for k in experimentList.keys():
        data.append([experimentList, k, rc, args, the_semaphore])

    return data


def run_main(rc, args, the_manager, the_semaphore):
    """
    This is the main loop for running the multithreaded data reads.  It should only be called once, by the if name == main() stanza.

    Architecturally, it does the following:

    * Archives old datasets to resultsArchive
    * synthesizes list of data files in the directory you asked for.
    * Runs a bunch of threads over ExperimentMaster, which moves us from files to Experiment()s
    * Each of the above spawns a Process() which takes in Experiments and operates on their Slices.
    * Writes a dataframe as output.

    Parameters
    ----------
    rc : runConstants
        RunConstants object for this run.
    args : ArgParse
        Parsed args for this run.

    Returns
    -------
    None.
    """
    start_time = time.time()
    t = rc.args.threads
    setENV_utils.setENV_threads(int(1))
    # Clear previous results and make results folder ready
    resultsArchivePath = rc.resultsPath.parent / "resultsArchive"
    resultsArchivePath.mkdir(parents=True, exist_ok=True)
    outputPlace = "AnalysisOutput_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.plotevents:
        args.keep = True
        singleEventsPath = rc.resultsPath / "events/singleEventPlots"
        singleEventsPath.mkdir(parents=True, exist_ok=True)
        rc = import_utils.check_plot_events_args(rc)

    if not args.keep:
        shutil.move(rc.resultsPath, str(resultsArchivePath / outputPlace))
        resultsPath.mkdir(parents=True)
        eventsPath = rc.resultsPath / "events"
        eventsPath.mkdir(parents=True)
    else:
        distutils.dir_util.copy_tree(
            rc.resultsPath, str(resultsArchivePath / outputPlace)
        )

    # save run constants
    rc_name = (
        "runConstants_atruntime_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    )
    rcFile = rc.resultsPath / rc_name
    with open(rcFile, "w") as file_h:
        writer = csv.writer(file_h)
        writer.writerow([str(rc)])

    # Find log files
    if rc.args.loadpreprocessed: 
        rc.args.instrumentType = "PCKL"
    if rc.args.instrumentType == "Chimera":
        experimentList = import_utils.create_experiment_list_Chimera(rc)
    elif rc.args.instrumentType == "smFET":
        experimentList = import_utils.create_experiment_list_smFET(rc)
    elif rc.args.instrumentType == "HEKA":
        experimentList = import_utils.create_experiment_list_HEKA(rc)
    elif rc.args.instrumentType == "CNP2":
        experimentList = import_utils.create_experiment_list_CNP2(rc)
    elif rc.args.instrumentType == "PCKL":
        experimentList = import_utils.create_experiment_list_PCKL(rc)

    exp_frame = pd.DataFrame.from_dict(
        experimentList, orient="Index"
    )  ## use keys as rows...
    sf_name = (
        "experimentList_atruntime_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    )
    statFile = rc.resultsPath / sf_name
    exp_frame.to_csv(statFile, sep=",", encoding="utf-8")

    data = assemble_data(experimentList, the_manager, the_semaphore, rc, args)
    statsDF = the_manager.list()

    if args.serial:
        logging.info("Serial run, probably to read debug messages...")
        slices = []
        for data_point in data:
            slices.append(exp_master_func([data_point, statsDF]))
    # Multicore reading
    else:
        flat_data = []
        for data_point in data:
            flat_data.append([data_point, statsDF])
        master_jobs = []
        for data_element in flat_data:
            master_proc = multiprocessing.Process(
                target=exp_master_func, args=(data_element,), daemon=False
            )
            master_proc.start()
            master_jobs.append(master_proc)

        del flat_data

        for job_proc in master_jobs:
            job_proc.join()

    ## Now the parallelism is over, and I have the statsDF from the slices...
    results_frame = pd.concat(statsDF)
    del statsDF

    logging.info("Dataframe looks like: \n" + results_frame.to_string())

    dataframe_name = (
        "dataframe_atruntime_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    )
    df_name = rc.resultsPath / dataframe_name
    results_frame.to_csv(df_name, encoding="utf-8")

    logging.info("--- Runtime was %s seconds ---" % (time.time() - start_time))

    return 0


def single_slice_master_func_wrapper(data, statsDF, semaphore):
    """
    This wrapper mediates access to running the slice_master_func, while respecting the semaphore.

    Parameters
    ----------
    data : List() of length 3
        the dataset list for this run; look at the start of the slice_master_func for the slice, args, rc variable
    statsDF : managed List()
        the shared variable we append to
    semaphore : multiprocessing.Semaphore()
        the managed semaphore for this run

    Returns
    -------
    None
        Will modify statsDF, with an append()
    """
    semaphore.acquire()
    out = slice_master_func(data)
    statsDF.append(out)
    semaphore.release()
    return


#### Main loop
if __name__ == "__main__":
    parser = parser_utils.cmd_parser()
    args = parser.parse_args()

    if args.data:
        logging.info("The data files I will run on are: \n" + str(dataPath))
    else:
        VA = import_utils.get_VA(dataPath)
        rc = rc(
            npersegPARAM,
            noverlapPARAM,
            args,
            seg_front_crop,
            seg_back_crop,
            VA,
            dataPath,
            resultsPath,
            allowed_time_lag,
        )

        the_manager = multiprocessing.Manager()
        the_semaphore = the_manager.Semaphore(args.threads)

        setENV_utils.setENV_threads(int(1))
        ctx = multiprocessing.get_context("spawn")
        gc.enable()

        run_main(rc, args, the_manager, the_semaphore)
