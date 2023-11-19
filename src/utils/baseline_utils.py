#######################################
## author: Jakob Buchheim,
## company: Columbia University
##
## version history:
##      * created: 20220223
##
## description:
## *
## * baseline drift and fluctuation correction
#######################################

import numpy as np
import logging
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
import os
import sys
import math
from textwrap import wrap
import pomegranate as pgm
from sklearn.mixture import GaussianMixture
import json
import pybaselines

from pathlib import Path

sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))

import event_utils
import plot_utils
import filter_utils
import data_classes
import export_utils

# plt.rc('text', usetex=True)
matplotlib.use("Agg")
plt.rcParams["agg.path.chunksize"] = 20000
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 0.5
plt.rcParams["axes.titlesize"] = 10

logging.basicConfig(
    level=logging.DEBUG, format=" %(asctime)s %(levelname)s - %(message)s"
)


def event_finder_baseline_correction(seg, rc):
    """

    this runs iterative event finding Pesa Dekker algorithm to get baseline drift.... 
    filters data to 50Hz (hardcoded)
    runs eventfinder which returns baseline after event detection
    substracts baseline from seg.Ia
    
    Parameters
    ----------
    dataSet : np.array of current trace
    hmmModel : fitted hmm model instance (pomegranate)

    Returns
    -------
    seg.Ia : np.array
        current trace which is corrected for baseline fluctuations ((iaStore - baselineRunner + baselineSeg.cur_mean))

    """
    # save and modify rc values
    iaStore = np.array(seg.Ia, copy=True)
    eventRC = copy.deepcopy(rc)
    eventRC.args.filter = rc.args.blcorrectionfreq

    # do filtering
    if eventRC.args.filter <= seg.samplerate:
        
        seg.Ia = filter_utils.lowpass_filter(
                seg.Ia, eventRC, seg.samplerate, order = 8, fType = "Butter", resample = 0.9)
    else:
        logging.info("requested lowpass frequency for baseline correction is higher than signal frequency cutoff".format(s0=self.name, s1=self.VAC))

    # run event finder
    baselineRunner = event_utils.find_events(seg, eventRC)

    # revert detrend on filtered baseline from event finder...
    if not math.isnan(eventRC.args.detrend):
        tmp = filter_utils.detrend_data(
            seg.Ia, int(np.floor(seg.samplerate * eventRC.args.detrend))
        )
        baselineRunner = baselineRunner + tmp - seg.cur_mean
        del tmp

    # smooth baseline before substraction of data
    if not len(eventRC.args.blcorrectionwhittaker) == 0:
        lam, p ,lamdev, exp = get_whittaker_baseline_params(eventRC)
        if p<0:
            baselineRunner = - baselineRunner + 2 * seg.cur_mean
        if exp:
            seg.baseline, param = pybaselines.whittaker.psalsa(baselineRunner, lam=lam, p=np.abs(p), k = lamdev)
        else:
            seg.baseline, param = pybaselines.whittaker.iasls(baselineRunner, lam=lam, p=np.abs(p), lam_1 = lamdev)
        if p<0:
            seg.baseline = - seg.baseline + 2 * seg.cur_mean
    else:
        seg.baseline = filter_utils.detrend_data(baselineRunner, int(np.floor(seg.samplerate / rc.args.blcorrectionfreq * 20)))

    blMean = np.mean(seg.baseline)

    # modify current trace
    seg.Ia = iaStore - seg.baseline + blMean

    # reset rc and clear data
    del eventRC, iaStore

    return seg.Ia, seg.baseline

def get_whittaker_baseline_params(rc):

    if len(rc.args.blcorrectionwhittaker) == 1:
        lam = np.float32(rc.args.blcorrectionwhittaker[0])
        p = 0.85
        lamdev = 1e-6
        exp = False
    if len(rc.args.blcorrectionwhittaker) == 2:
        lam = np.float32(rc.args.blcorrectionwhittaker[0])
        p = np.float32(rc.args.blcorrectionwhittaker[1])
        lamdev = 1e-6
        exp = False
    if len(rc.args.blcorrectionwhittaker) == 3:
        lam = np.float32(rc.args.blcorrectionwhittaker[0])
        p = np.float32(rc.args.blcorrectionwhittaker[1])
        lamdev = np.float32(rc.args.blcorrectionwhittaker[2])
        exp = False
    if len(rc.args.blcorrectionwhittaker) == 4:
        lam = np.float32(rc.args.blcorrectionwhittaker[0])
        p = np.float32(rc.args.blcorrectionwhittaker[1])
        lamdev = np.float32(rc.args.blcorrectionwhittaker[2])
        exp = False
        if rc.args.blcorrectionwhittaker[3] == 'exp':
            exp = True


    return lam,p ,lamdev, exp

def whittaker_baseline_correction(seg, rc):

    lam,p ,lamdev = get_whittaker_baseline_params(rc)

    if not len(rc.args.blcorrectionwhittaker) == 0:
        seg.baseline, param = pybaselines.whittaker.iasls(seg.Ia,lam=lam, p=p, lam_1=lamdev)
        blMean = np.mean(seg.baseline)
        seg.Ia = seg.Ia - seg.baseline + blMean
    else:
        seg.baseLine = None

    return seg.Ia, seg.baseline