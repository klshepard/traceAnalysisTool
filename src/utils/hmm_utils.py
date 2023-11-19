#######################################
## author: Jakob Buchheim, Boyan Penkov
## company: Columbia University
##
## version history:
##      * created: 20200917
##
## description:
## *
## * hidden markov model for schnellstapel
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


def build_hmm_model(seg, rc):
    """

    this function builds the hmm model. There are four different models currently avaialable. rc.args.hmm defines which model is returned.
    rc.args.hmm == 'two': two state model with transition back and forth
    rc.args.hmm == 'three': three state model with one hidden state (same emission as baseline) allowing transition between all states
    rc.args.hmm == 'four': four state model with two hidden states (same emission as baseline and event line) allowing transition between all states
    rc.args.hmm == 'linera': three state model with two hidden states (same emission as baseline and event line), no direct transition between one baseline state and the event state

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data

    Returns
    -------
    hmmModel : pomegranate hmm model

    """

    # Define states ---------------------------------------

    # state offset
    allowedEventType = event_utils.get_allowed_event(seg, rc)
    if allowedEventType == "up":
        stateOffset = -rc.args.mineventdepth
    else:
        stateOffset = rc.args.mineventdepth

    # UB - not bound not observed state = STATE0
    meanUB = (np.max(seg.Ia) + seg.cur_mean) * 0.5 + stateOffset * 0.2
    stdUB = seg.cur_std * 1.5
    UBDist = pgm.NormalDistribution(meanUB, stdUB)
    UB = pgm.State(UBDist, name="UB")

    # BO - bound observed state = STATE1
    meanBO = (
        np.min(seg.Ia) + seg.cur_mean
    ) * 0.5 - stateOffset * 0.8  # np.min(seg.Ia)#seg.cur_mean - stateOffset
    stdBO = seg.cur_std
    BODist = pgm.NormalDistribution(meanBO, stdBO)
    BO = pgm.State(BODist, name="BO")

    # rate constants - initial value
    k1on = 10.0 / seg.samplerate
    k1off = 5.0 / seg.samplerate

    if "three" in rc.args.hmm:
        ## Three state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO)

        # Define transitions ---------------------------------------

        k2on = 25.0 / seg.samplerate
        k2off = 25.0 / seg.samplerate
        k3on = 50.0 / seg.samplerate
        k3off = 5.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(UB, UB, 0.999, 100)
        hmmModel.add_transition(BO, BO, 0.999, 100)
        hmmModel.add_transition(BNO, BNO, 0.9, 100)
        hmmModel.add_transition(UB, BO, k1on, 0)
        hmmModel.add_transition(BO, UB, k1off, 0)
        hmmModel.add_transition(BNO, BO, k3on, 0)
        hmmModel.add_transition(BO, BNO, k3off, 0)
        hmmModel.add_transition(UB, BNO, k2on, 0)
        hmmModel.add_transition(BNO, UB, k2off, 0)

    elif "slowandfast" in rc.args.hmm:
        ## Three state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO)

        # Define transitions ---------------------------------------

        k2on = 25.0 / seg.samplerate
        k2off = 25.0 / seg.samplerate
        k3on = 50.0 / seg.samplerate
        k3off = 5.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(UB, UB, 0.999, 100)
        hmmModel.add_transition(BO, BO, 0.999, 100)
        hmmModel.add_transition(BNO, BNO, 0.9, 100)
        hmmModel.add_transition(UB, BO, k1on, 0)
        hmmModel.add_transition(BO, UB, k1off, 0)
        hmmModel.add_transition(BNO, BO, k3on, 0)
        hmmModel.add_transition(BO, BNO, k3off, 0)

    elif "linear" in rc.args.hmm:
        ## Three state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO)

        # Define transitions ---------------------------------------

        k2on = 20.0 / seg.samplerate
        k2off = 100.0 / seg.samplerate
        k3on = 50.0 / seg.samplerate
        k3off = 5.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 0.5, 1)
        hmmModel.add_transition(hmmModel.start, BO, 0.5, 1)
        hmmModel.add_transition(UB, UB, 0.999, 100)
        hmmModel.add_transition(BO, BO, 0.999, 100)
        hmmModel.add_transition(BNO, BNO, 0.9, 100)
        hmmModel.add_transition(BNO, BO, k3on, 0)
        hmmModel.add_transition(BO, BNO, k3off, 0)
        hmmModel.add_transition(UB, BNO, k2on, 0)
        hmmModel.add_transition(BNO, UB, k2off, 0)

    elif "triang" in rc.args.hmm:
        ## Three state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO)

        # Define transitions ---------------------------------------

        k2on = 1.0 / seg.samplerate
        k2off = 25.0 / seg.samplerate
        k3on = 50.0 / seg.samplerate
        k3off = 5.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 0.5, 1)
        hmmModel.add_transition(hmmModel.start, BO, 0.5, 1)
        hmmModel.add_transition(UB, UB, 0.99, 100)
        hmmModel.add_transition(BO, BO, 0.999, 100)
        hmmModel.add_transition(BNO, BNO, 0.999, 100)
        hmmModel.add_transition(BNO, BO, k3on, 0)
        hmmModel.add_transition(BO, BNO, k3off, 0)
        hmmModel.add_transition(UB, BNO, k2on, 0)
        hmmModel.add_transition(BNO, UB, k2off, 0)

    elif "two" in rc.args.hmm:
        ## basic two state model

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BO)

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1)
        hmmModel.add_transition(UB, UB, 0.999, 100)
        hmmModel.add_transition(BO, BO, 0.999, 100)
        hmmModel.add_transition(UB, BO, k1on, 0)
        hmmModel.add_transition(BO, UB, k1off, 0)

    elif "four" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define two additional states with same distribution as original ones ---------------------------------------
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO)

        # Define transitions ---------------------------------------

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.9731450751183339, 100)
        hmmModel.add_transition(UBO, UBO, 0.9614117926973504, 100)
        hmmModel.add_transition(UB, UB, 0.985363846751346, 100)
        hmmModel.add_transition(BO, BO, 0.9477213634247093, 100)
        hmmModel.add_transition(UB, UBO, 0.03837042477866877, 0)
        hmmModel.add_transition(UBO, UB, 0.012497073086702535, 0)
        hmmModel.add_transition(BNO, UB, 0.012147158707664191, 0)
        hmmModel.add_transition(UB, BNO, 0.010799110770787034, 0)
        hmmModel.add_transition(BO, BNO, 0.0522786365752908, 0)
        hmmModel.add_transition(BNO, BO, 0.0014707766174002008, 0)
        hmmModel.add_transition(UBO, BO, 0.0038113931770136423, 0)
        hmmModel.add_transition(BO, UBO, 0.003609113421594702, 0)

    elif "4state" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO)

        # Define transitions ---------------------------------------

        k2on = 25.0 / seg.samplerate
        k2off = 25.0 / seg.samplerate
        k3on = 50.0 / seg.samplerate
        k3off = 5.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.9731450751183339, 100)
        hmmModel.add_transition(UBO, UBO, 0.9614117926973504, 100)
        hmmModel.add_transition(UB, UB, 0.985363846751346, 100)
        hmmModel.add_transition(BO, BO, 0.9477213634247093, 100)
        hmmModel.add_transition(UB, UBO, 0.003837042477866877, 0)
        hmmModel.add_transition(UBO, UB, 0.012497073086702535, 0)
        hmmModel.add_transition(BNO, UB, 0.012147158707664191, 0)
        hmmModel.add_transition(UB, BNO, 0.010799110770787034, 0)
        hmmModel.add_transition(BO, BNO, 0.0522786365752908, 0)
        hmmModel.add_transition(BNO, BO, 0.014707766174002008, 0)
        # hmmModel.add_transition(UBO, BO, 0.0038113931770136423, 0)
        # hmmModel.add_transition(BO, UBO, 0.003609113421594702, 0)


    elif "new5" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(UBDist, name="UBO")
        UBD = pgm.State(BODist, name="UBD")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO, UBD)

        # Define transitions ---------------------------------------

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.9731450751183339, 100)
        hmmModel.add_transition(UBO, UBO, 0.9614117926973504, 100)
        hmmModel.add_transition(UB, UB, 0.985363846751346, 100)
        hmmModel.add_transition(BO, BO, 0.9477213634247093, 100)
        hmmModel.add_transition(UBD, UBD, 0.9818860682298637, 100)
        hmmModel.add_transition(UB, UBO, 0.003837042477866877, 0)
        hmmModel.add_transition(UBO, UB, 0.012497073086702535, 0)
        hmmModel.add_transition(BNO, UB, 0.012147158707664191, 0)
        hmmModel.add_transition(UB, BNO, 0.010799110770787034, 0)
        hmmModel.add_transition(BO, BNO, 0.0522786365752908, 0)
        hmmModel.add_transition(BNO, BO, 0.014707766174002008, 0)
        hmmModel.add_transition(UBD, UBO, 0.018113931770136423, 0)
        hmmModel.add_transition(UBO, UBD, 0.02609113421594702, 0)
        hmmModel.add_transition(UBD, BO, 0.0038113931770136423, 0)
        hmmModel.add_transition(BO, UBD, 0.003609113421594702, 0)

    elif "five" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")
        D2 = pgm.State(BODist, name="D2")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO, D2)

        # Define transitions ---------------------------------------

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.98489, 100)
        hmmModel.add_transition(UBO, UBO, 0.929119, 100)
        hmmModel.add_transition(UB, UB, 0.933867, 100)
        hmmModel.add_transition(BO, BO, 0.972821, 100)
        hmmModel.add_transition(D2, D2, 0.932441, 100)
        # hmmModel.add_transition(UB, BO, k1on, 0)
        # hmmModel.add_transition(BO, UB, k1off, 0)
        hmmModel.add_transition(UB, UBO, 0.061575, 0)
        hmmModel.add_transition(UBO, UB, 0.092351, 0)
        hmmModel.add_transition(BNO, UB, 0.012512, 0)
        hmmModel.add_transition(UB, BNO, 0.014558, 0)
        # hmmModel.add_transition(BO, UBO, 0.023944, 0)
        # hmmModel.add_transition(UBO, BO, 0.015738, 0)
        hmmModel.add_transition(BO, BNO, 0.003235, 0)
        hmmModel.add_transition(BNO, BO, 0.002597, 0)
        hmmModel.add_transition(D2, BO, 0.002792, 0)
        hmmModel.add_transition(BO, D2, 0.07559, 0)
        # hmmModel.add_transition(UBO, BNO, k2on, 0)
        # hmmModel.add_transition(BNO, UBO, k2on, 0)

    elif "longli" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO)

        # Define transitions ---------------------------------------

        k2on = 25.0 / seg.samplerate
        k2off = 25.0 / seg.samplerate
        k3on = 50.0 / seg.samplerate
        k3off = 5.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.9731450751183339, 100)
        hmmModel.add_transition(UBO, UBO, 0.9614117926973504, 100)
        hmmModel.add_transition(UB, UB, 0.985363846751346, 100)
        hmmModel.add_transition(BO, BO, 0.9477213634247093, 100)
        hmmModel.add_transition(BNO, UB, 0.012147158707664191, 0)
        hmmModel.add_transition(UB, BNO, 0.010799110770787034, 0)
        hmmModel.add_transition(BO, BNO, 0.0522786365752908, 0)
        hmmModel.add_transition(BNO, BO, 0.014707766174002008, 0)
        hmmModel.add_transition(UBO, BO, 0.0038113931770136423, 0)
        hmmModel.add_transition(BO, UBO, 0.003609113421594702, 0)
    
    elif "tri4" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define two additional hidden state with same distribution as the two original -
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO)

        # Define transitions ---------------------------------------

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.9731450751183339, 100)
        hmmModel.add_transition(UBO, UBO, 0.9614117926973504, 100)
        hmmModel.add_transition(UB, UB, 0.985363846751346, 100)
        hmmModel.add_transition(BO, BO, 0.9477213634247093, 100)
        hmmModel.add_transition(UB, UBO, 0.003837042477866877, 0)
        hmmModel.add_transition(UBO, UB, 0.012497073086702535, 0)
        hmmModel.add_transition(BNO, UB, 0.012147158707664191, 0)
        hmmModel.add_transition(UB, BNO, 0.010799110770787034, 0)
        hmmModel.add_transition(BO, BNO, 0.0522786365752908, 0)
        hmmModel.add_transition(BNO, BO, 0.014707766174002008, 0)
        hmmModel.add_transition(UB, BO, 0.01113931770136423, 0)
        hmmModel.add_transition(BO, UB, 0.0003609113421594702, 0)

    elif "new4" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define two additional hidden state with same distribution as the two original -
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO)

        # Define transitions ---------------------------------------

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(BNO, BNO, 0.9731450751183339, 100)
        hmmModel.add_transition(UBO, UBO, 0.9614117926973504, 100)
        hmmModel.add_transition(UB, UB, 0.985363846751346, 100)
        hmmModel.add_transition(BO, BO, 0.9477213634247093, 100)
        hmmModel.add_transition(UB, UBO, 0.003837042477866877, 0)
        hmmModel.add_transition(UBO, UB, 0.012497073086702535, 0)
        hmmModel.add_transition(BNO, UB, 0.012147158707664191, 0)
        hmmModel.add_transition(UB, BNO, 0.010799110770787034, 0)
        hmmModel.add_transition(BO, BNO, 0.0522786365752908, 0)
        hmmModel.add_transition(BNO, BO, 0.014707766174002008, 0)
        hmmModel.add_transition(UBO, BO, 0.0038113931770136423, 0)
        hmmModel.add_transition(BO, UBO, 0.003609113421594702, 0)
        hmmModel.add_transition(UB, BO, 0.01113931770136423, 0)
        hmmModel.add_transition(BO, UB, 0.0003609113421594702, 0)
        hmmModel.add_transition(UBO, BNO, 0.02522786365752908, 0)
        hmmModel.add_transition(BNO, UBO, 0.0014707766174002008, 0)

    elif "sixstate" in rc.args.hmm:
        ## four state model with one hidden in baseline

        # Define third state with same distribution as first ---------------------------------------
        # BNO - bound not observed state = STATE3
        # meanBNO = 0
        # stdBNO = 4
        # BNODist = NormalDistribution(meanBNO,stdBNO)
        BNO = pgm.State(UBDist, name="BNO")
        UBO = pgm.State(BODist, name="UBO")
        SBO = pgm.State(UBDist, name="SBO")
        SUB = pgm.State(UBDist, name="SUB")

        # Define model ---------------------------------------
        hmmModel = pgm.HiddenMarkovModel()
        hmmModel.add_states(UB, BNO, BO, UBO, SBO, SUB)

        # Define transitions ---------------------------------------

        k1on = 1.0 / seg.samplerate
        k1off = 15.0 / seg.samplerate
        k2on = 50.0 / seg.samplerate
        k2off = 15.0 / seg.samplerate
        k3on = 5.0 / seg.samplerate
        k3off = 50.0 / seg.samplerate

        # model definitions possible state transitions (edges)
        hmmModel.add_transition(hmmModel.start, UB, 1, 1)
        hmmModel.add_transition(hmmModel.start, BO, 1, 1)
        hmmModel.add_transition(UB, UB, 0.999, 100)
        hmmModel.add_transition(BO, BO, 0.999, 100)
        hmmModel.add_transition(BNO, BNO, 0.9, 100)
        hmmModel.add_transition(UBO, UBO, 0.9, 100)
        hmmModel.add_transition(SUB, SUB, 0.9, 100)
        hmmModel.add_transition(SBO, SBO, 0.9, 100)

        hmmModel.add_transition(BNO, BO, k1on, 0)
        hmmModel.add_transition(BO, BNO, k1off, 0)

        hmmModel.add_transition(UBO, UB, k3off, 0)
        hmmModel.add_transition(UB, UBO, k3on, 0)

        hmmModel.add_transition(SBO, BO, k2on * 10, 0)
        hmmModel.add_transition(BO, SBO, k2off * 10, 0)

        hmmModel.add_transition(UBO, SUB, k3off * 5, 0)
        hmmModel.add_transition(SUB, UBO, k3on * 5, 0)

        hmmModel.add_transition(UBO, BO, k2on, 0)
        hmmModel.add_transition(BO, UBO, k2off, 0)

        hmmModel.add_transition(SBO, SUB, k3off, 0)
        hmmModel.add_transition(SUB, SBO, k3on, 0)

        hmmModel.add_transition(UB, BNO, k3off * 0.2, 0)
        hmmModel.add_transition(BNO, UB, k3on * 0.2, 0)

    # finalize model ---------------------------------------
    hmmModel.bake()

    modelDict = get_model_dict(hmmModel)
    return hmmModel, modelDict


def get_model_dict(hmmModel):
    """
    connects state number and state name as well asigns transition value to a human readable transition 

    Parameters
    ----------
    hmmModel : pomegranate model

    Returns
    -------
    modelDict : dict
        containing {states: {stateNumber:stateName}, transitions:{transitionNumber:startstate-2-targetstatename}, transitionsMidx:{transitionMatrixIndex:startstate-2-targetstatename}}

    """
    modelJson = hmmModel.to_json()
    tmpDict = json.loads(modelJson)

    stateDict = {
        np.str(i): tmpDict["states"][i]["name"]
        for i in range(0, len(tmpDict["states"]))
    }
    transitionNameDict = {
        np.str(tmpDict["edges"][i][1] ** 2 - tmpDict["edges"][i][0] ** 2)
        if tmpDict["edges"][i][1] != tmpDict["edges"][i][0]
        else np.int(
            np.str(tmpDict["edges"][i][0]) + np.str(tmpDict["edges"][i][0])
        ): stateDict[np.str(tmpDict["edges"][i][0])]
        + "-2-"
        + stateDict[np.str(tmpDict["edges"][i][1])]
        for i in range(0, len(tmpDict["edges"]))
    }
    transitionNumberDict = {
        np.str(tmpDict["edges"][i][0:2]): stateDict[np.str(tmpDict["edges"][i][0])]
        + "-2-"
        + stateDict[np.str(tmpDict["edges"][i][1])]
        for i in range(0, len(tmpDict["edges"]))
    }
    modelDict = {
        "states": stateDict,
        "transitions": transitionNameDict,
        "transitionsMidx": transitionNumberDict,
    }

    return modelDict


def initialize_hmm_model(seg, rc, downSampledDataSet, hmmModel):
    """
    initialize model with low strongly subsampled data to get distribution means better. Updates priors!

    Parameters
    ----------
        seg : segment class data
        rc : run constant class data
        dataSet : np.array
            currnentdata which is detrended already
        hmmModel : pomegranate model

    Returns
    -------
        hmmModel : pomegranate model
            initialized hmm model with initial fit. 
    

    """
    logging.info(
        "initializing HMM model {s0}_{s1:.0f}mV \t".format(s0=seg.name, s1=seg.VAC)
    )

    hmmModel.fit(
        [downSampledDataSet[0 : evaluate_hmm_index(downSampledDataSet, rc)]],
        algorithm="baum-welch",
        min_iterations=40,
        max_iterations=80,
        lr_decay=0.75,
        use_pseudocount=True,
        emission_pseudocount=1,
        n_jobs=5,
    )

    logging.info(
        "DONE: initializing HMM model {s0}_{s1:.0f}mV \t".format(
            s0=seg.name, s1=seg.VAC
        )
    )

    # hmmModel.fit([downSampledDataSet[0:evaluate_hmm_index(downSampledDataSet, rc)]], algorithm = 'viterbi', min_iterations = 80, lr_decay = 0.75, use_pseudocount = True, emission_pseudocount = 1, n_jobs = 5)

    return hmmModel


def evaluate_hmm_index(dataSet, rc):
    """
    returns index of dataSet up to which the data is used to fit the hmm model and from which onwards the data is used to evaluate the hmm model
    to deterimine the index the input argument rc.args.fitfraction is used. 


    Parameters
    ----------
        rc : run constant class data
        dataSet : np.array
            currnentdata which is detrended already

    Returns
    -------
        evaluateHmmIndex : int
            index up to which model is fitted
    
    """

    if rc.args.fitfraction <= 1.0:
        evaluateHmmIndex = np.int(len(dataSet) * rc.args.fitfraction)
    else:
        evaluateHmmIndex = np.int(len(dataSet))

    # prevent too short fitting region
    if (evaluateHmmIndex >= len(dataSet) * 0.9) and not (
        evaluateHmmIndex == np.int(len(dataSet))
    ):
        evaluateHmmIndex = np.int(len(dataSet) * 0.9)
        logging.info(
            "reset requested to hmm evaluation data fraction to 10% of the data"
        )

    return evaluateHmmIndex


def lowpass_data(seg, rc, dataSet):
    """

    this function is a wrapper for lowpassing data without downsampling it 

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    dataSet : np.array
        currnentdata which is detrended already

    Returns
    -------
    lowpassedDataSet np.array
        current which is lowpassed but NOT downsampled
    """

    originalFilter = np.copy(rc.args.filter)
    tmpRc = rc

    tmpRc.args.filter = 20
    try: 
        lowpassedDataSet = filter_utils.LP_filter(
            dataSet,
            tmpRc.args.filter,
            seg.samplerate,
            9,
            fType="FIR",
            resample=0.9,
            filteredInt=False,
        )
    except:
        logging.info(
        "twostephmm option failed because lowpass filter further is not possible on {s0}_{s1:.0f}mV".format(s0=seg.name, s1=seg.VAC)
        )
        lowpassedDataSet = dataSet
    rc.args.filter = originalFilter

    return lowpassedDataSet

def hmm_gaussian_classifier(dataSet, seg, rc, allowedEventType):

    """
    calculates baseline value and standard deviation of data based on gaussian mixture model fit, fits 2 gaussian peaks - usually works nicely
    Problem when data in saturation, no Histogram fit possible if single value.

    Output: plot of histogram with peak fitting

    Parameters
    ----------
    dataSet : np.array 
        of data for which to calculate the standard deviation
    seg : segment class data
    rc : run constant class data
    allowedEventType : str
        passes allowed event type to be able to return the correct baseline value

    Returns
    -------
    baselineStateMean : np.float
        baseline average value: for cases looking for up and down events: highest histogram peak; for cases looking for up events: lowest current peak; for cases looking for down events: highest current peak; 
    baselineStateStd : np.float
        standard deviation of baseline gaussian peak
    eventStateMean : np.float 
        event average value: 
    eventStateStd : np.float
        standard deviation of event gaussian peak

    """
    name = seg.dfIndex() + "_Histogram.png"
    writeFile = rc.resultsPath / "hmm" / name
    writeFile.parent.mkdir(parents=True, exist_ok=True)

    # make histogram over full trace
    numberOfBins = int(np.abs(np.max(dataSet) - np.min(dataSet)) / 1.0)
    if numberOfBins < 1:
        return seg.cur_mean, 1000, np.nan
    elif numberOfBins < 40:
        numberOfBins = 40
        # logging.info("issues with bin size - too small set to 20")

    logging.info(
        "initializing data for GMM fit {s0}_{s1:.0f}mV".format(s0=seg.name, s1=seg.VAC)
    )

    rdmSize = np.int(5e4)
    if rdmSize > len(dataSet):
        rdmSize = np.int(len(dataSet))

    smallDataSet = np.random.choice(dataSet, rdmSize, replace=False)
    hist, bins = np.histogram(smallDataSet, bins=numberOfBins, density=True)

    # initialize gaussian mixture model, use to many to find all peaks
    numberOfGaussians = 2  # for PCP data with high concentration 5 is correct
    logging.info(
        "start fitting GMM for HMM current trace of experiment {s0}_{s1:.0f}mV".format(
            s0=seg.name, s1=seg.VAC
        )
    )

    fittedHist = GaussianMixture(
        numberOfGaussians, covariance_type="spherical", tol=1e-6, max_iter=500
    )

    # fit the data to the model
    fittedHist.fit(X=smallDataSet.reshape(-1, 1))

    # create dataframe from fitting parameter to better handle the fitted stuff
    statsData = pd.DataFrame(
        data=list(
            zip(
                np.array(fittedHist.means_).flatten(),
                np.sqrt(fittedHist.covariances_.flatten()),
                np.exp(fittedHist.score_samples(fittedHist.means_)),
                fittedHist.predict(fittedHist.means_),
            )
        ),
        columns=["MEAN", "STANDARDDEVIATION", "P", "LABEL"],
    )
    # statsData.sort_values(by = 'P', ascending = False, inplace = True)
    logging.info(
        "Fitted GMM for HMM current trace of experiment {s0}_{s1:.0f}mV \t Results: \n {s2}".format(
            s0=seg.name, s1=seg.VAC, s2=statsData
        )
    )

    # reference plot for comparison
    fig2, ax = plt.subplots()
    ax.grid(True, which="both", ls="-", color="0.65")
    ax.set_xlabel(r"$I_{A}$ [$pA$]")
    ax.set_ylabel(r"$probability$ ")
    ax.set_title("\n".join(wrap(name)))

    ax.bar((bins[:-1] + bins[1:]) / 2, hist, (bins[0] - bins[1]) * 0.8)
    fittedHist_y = np.exp(fittedHist.score_samples(bins.reshape(-1, 1)))
    ax.plot(bins, fittedHist_y, "-r")

    responsibilities = fittedHist.predict_proba(bins.reshape(-1, 1))
    fittedHist_y_individual = responsibilities * fittedHist_y[:, np.newaxis]
    ax.plot(bins, fittedHist_y_individual, "--r")

    fig2.savefig(writeFile, dpi=100, format="png")
    fig2.clear()
    plt.close(fig2)

    # assembling function output (return values for highest peak)
    outputIdx = 0
    if allowedEventType == "up" and not statsData.MEAN.empty:
        outputIdx = (
            statsData.MEAN.idxmin()
        )  # largest current (baseline) value if probability is not too low
        eventOutputIdx = statsData["MEAN"].idxmax()
    elif allowedEventType == "down" and not statsData.MEAN.empty:
        outputIdx = statsData.MEAN.idxmax()
        eventOutputIdx = statsData["MEAN"].idxmin()
    else:
        outputIdx = statsData.P.idxmax()  # highest peak value
        eventOutputIdx = statsData["MEAN"].idxmin()

    baselineStateMean = statsData.iloc[outputIdx].MEAN
    baselineStateStd = statsData.iloc[outputIdx].STANDARDDEVIATION

    eventStateMean = statsData.iloc[eventOutputIdx].MEAN
    eventStateStd = statsData.iloc[eventOutputIdx].STANDARDDEVIATION

    del hist, bins, statsData, fig2, ax, fittedHist_y

    logging.info(
        "Baseline / Event mean value based on GMM fit of data fit of: {s1} - {s2:.2f}   {s3:.2f}".format(
            s1=seg.dfIndex(), s2=baselineStateMean, s3=eventStateMean
        )
    )

    if abs(eventStateMean - baselineStateMean) < rc.args.mineventdepth:
        eventStateMean = baselineStateMean - rc.args.mineventdepth
        logging.info(
            "reset event mean (depth {s2:.2f} ) to baseline mean - min event Depth {s1}".format(
                s1=seg.dfIndex(), s2=abs(eventStateMean - baselineStateMean)
            )
        )

    logging.info(
        "baseline and event std {s1}, : baselineStateSTD = {s2:.2f} eventStateSTD = {s3:.2f} ".format(
            s1=seg.dfIndex(), s2=baselineStateStd, s3=eventStateStd
        )
    )
    if eventStateStd > baselineStateStd:
        eventStateStd = baselineStateStd * rc.args.PDFreversal / rc.args.PDF
        logging.info(
            "reset event std {s1}, eventStateSTD = {s3:.2f} ".format(
                s1=seg.dfIndex(), s3=eventStateStd
            )
        )

    return baselineStateStd, eventStateStd, baselineStateMean, eventStateMean


def calculate_baseline_std(seg, rc, downSampledDataSet, dataSet, hmmModel, modelDict):
    """
    calculates baseline value and standard deviation of data based on found events in intialization step of hmm model with low sample rate hmm model.

    Parameters
    ----------
    dataSet : np.array 
        of data for which to calculate the standard deviation
    seg : segment class data
    rc : run constant class data
    downSampledDataSet : np.array
    dataSet : np.array
    hmmModel : pomegranate model
    modelDict : dict

    Returns
    -------
    baselineStateMean : np.float
        baseline average value: for cases looking for up and down events: highest histogram peak; for cases looking for up events: lowest current peak; for cases looking for down events: highest current peak; 
    baselineStateStd : np.float
        standard deviation of baseline gaussian peak
    eventStateMean : np.float 
        event average value: 
    eventStateStd : np.float
        standard deviation of event gaussian peak

    """

    trace, state = get_model_trace(downSampledDataSet, hmmModel)
    logging.info(
        "Calculating distribution values based on low sample rate HMM data fit of: {s1}".format(
            s1=seg.dfIndex()
        )
    )

    BNOstate = -100
    UBOstate = -100
    SBOstate = -100
    SUBstate = -100
    D2state = -100
    UBDstate = -100
    for i in range(0, len(modelDict["states"])):
        if modelDict["states"][np.str(i)] == "UB":
            UBstate = i
        if modelDict["states"][np.str(i)] == "BO":
            BOstate = i
        if modelDict["states"][np.str(i)] == "BNO":
            BNOstate = i
        if modelDict["states"][np.str(i)] == "UBO":
            UBOstate = i
        if modelDict["states"][np.str(i)] == "SBO":
            SBOstate = i
        if modelDict["states"][np.str(i)] == "SUB":
            SUBstate = i
        if modelDict["states"][np.str(i)] == "D2":
            D2state = i
        if modelDict["states"][np.str(i)] == "UBD":
            UBDstate = i

    if "five" in rc.args.hmm:
        if D2state != -100:
            state[state == D2state] = BOstate
    # elif "new4" in rc.args.hmm:
    #     if BNOstate != -100:
    #         state[state == BNOstate] = UBstate
    #     if UBOstate != -100:
    #         state[state == UBOstate] = BOstate
    elif "new5" in rc.args.hmm:
        if UBDstate != -100:
            state[state == UBDtate] = BOstate
    elif "sixstate" in rc.args.hmm:
        if BNOstate != -100:
            state[state == BNOstate] = UBstate
        if UBOstate != -100:
            state[state == UBOstate] = BOstate
        if SBOstate != -100:
            state[state == SBOstate] = UBstate
        if SUBstate != -100:
            state[state == SUBstate] = UBstate
    else:
        if BNOstate != -100:
            state[state == BNOstate] = UBstate
        if UBOstate != -100:
            state[state == UBOstate] = BOstate

    boundState = state == BOstate
    startTrue = 0
    startTrueMod = False
    shortEventsThreshold = 30.0 / rc.args.filter * seg.samplerate
    minEventLevel = np.nanmean(dataSet[boundState == False])
    for i in np.arange(0, len(boundState)):
        if boundState[i] and not startTrueMod:
            startTrue = i
            startTrueMod = True
        elif startTrueMod and not boundState[i]:
            lenState = i - startTrue
            if lenState < shortEventsThreshold:
                boundState[startTrue:i] = False
            else:
                tmpEventLevel = np.mean(dataSet[startTrue : i - 1])
                if tmpEventLevel < minEventLevel:
                    minEventLevel = tmpEventLevel

            startTrueMod = False

    baselineStateStd = np.nanstd(dataSet[boundState == False]).tolist() * rc.args.PDF
    baselineStateMean = np.nanmean(dataSet[boundState == False]).tolist()

    if not boundState.any():
        eventStateMean = baselineStateMean - rc.args.mineventdepth
    else:
        eventStateMean = np.nanmean(dataSet[boundState]).tolist()

    logging.info(
        "Baseline / Event mean value based on low sample rate HMM data fit of: {s1} - {s2:.2f}   {s3:.2f}".format(
            s1=seg.dfIndex(), s2=baselineStateMean, s3=eventStateMean
        )
    )

    if abs(eventStateMean - baselineStateMean) < rc.args.mineventdepth:
        eventStateMean = baselineStateMean - rc.args.mineventdepth
        logging.info(
            "reset event mean (depth {s2:.2f} ) to baseline mean - min event Depth {s1}".format(
                s1=seg.dfIndex(), s2=abs(eventStateMean - baselineStateMean)
            )
        )

    if not boundState.any():
        eventStateStd = baselineStateStd * rc.args.PDFreversal / rc.args.PDF
    else:
        eventStateStd = (
            np.sqrt(
                np.sum(
                    np.multiply(
                        dataSet[boundState] - eventStateMean,
                        dataSet[boundState] - eventStateMean,
                    )
                )
                / (len(dataSet[boundState]) - 1)
            )
            * rc.args.PDFreversal
        )

    logging.info(
        "baseline and event std {s1}, : baselineStateSTD = {s2:.2f} eventStateSTD = {s3:.2f} ".format(
            s1=seg.dfIndex(), s2=baselineStateStd, s3=eventStateStd
        )
    )
    if eventStateStd > baselineStateStd:
        eventStateStd = baselineStateStd * rc.args.PDFreversal / rc.args.PDF
        logging.info(
            "reset event std {s1}, eventStateSTD = {s3:.2f} ".format(
                s1=seg.dfIndex(), s3=eventStateStd
            )
        )

    return baselineStateStd, eventStateStd, baselineStateMean, eventStateMean


def filter_spikes_avg(seg, rc, dataSet, model, modelDict):
    """
    removes very short spike like hmm states (events shorter than: 2.0 / rc.args.filter) if not deep enough. Depth required to be considered real: >=0.8 of HMM step size
    replaces current trace spike values with mean value around the spike (+- np.int(shortEventsThreshold * 1e-6 * seg.samplerate)

    Parameters
    ----------
    dataSet : np.array 
        of data for which to calculate the standard deviation
    seg : segment class data
    rc : run constant class data
    model : pomegranate model
    modelDict : dict

    Returns
    -------
    dataSet : np.array
        current data without spikes

    """
    seg.hmmTrace, seg.hmmState = get_model_trace(dataSet, model)
    seg.modelDict = modelDict

    hmmEventDF = get_dwell_times(seg, rc)

    shortEventsThreshold = 4.0 / rc.args.filter * 1e6  # seg.samplerate

    logging.info(
        "Spike filtering HMM events length {s2:.2f} of {s1}".format(
            s1=seg.dfIndex(), s2=shortEventsThreshold
        )
    )

    UBOstate = 4
    BNOstate = 4
    SUBstate = 4
    SBOstate = 4
    D2state = 4
    UBDstate = 4
    for i in range(0, len(modelDict["states"])):
        if modelDict["states"][np.str(i)] == "BO":
            BOstate = i
            muBO = hmmEventDF[hmmEventDF["STATELABEL"] == BOstate][
                "STATELEVEL"
            ].unique()
        if modelDict["states"][np.str(i)] == "UB":
            UBstate = i
            muUB = hmmEventDF[hmmEventDF["STATELABEL"] == UBstate][
                "STATELEVEL"
            ].unique()
        if modelDict["states"][np.str(i)] == "BNO":
            BNOstate = i
        if modelDict["states"][np.str(i)] == "UBO":
            UBOstate = i
        if modelDict["states"][np.str(i)] == "SBO":
            SBOstate = i
        if modelDict["states"][np.str(i)] == "SUB":
            SUBstate = i
        if modelDict["states"][np.str(i)] == "D2":
            D2state = i
        if modelDict["states"][np.str(i)] == "UBD":
            UBDstate = i

    # DOWN spikes
    if "OLDnew4" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                )
            )
        ]
    elif "tir4" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                )
            )
        ]
    elif "five" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                )
            )
        ]
    elif "new5" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBD-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                )
            )
        ]
    elif "sixstate" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-SUB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-SBO")
                )
            )
        ]
    else:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-BNO")
                )
            )
        ]
    staIdx = shortEvents["STATESTARTIDX"].values
    stoIdx = shortEvents["STATESTOPIDX"].values

    overlap = np.int(shortEventsThreshold * 1e-6 * seg.samplerate)
    # for sta, sto in zip(staIdx,stoIdx):
    #     if np.mean(dataSet[sta:sto]) > muUB - 0.8 * np.abs(muUB - muBO):
    #         if (sta > overlap) & (sto < len(dataSet) - overlap):
    #             dataSet[sta:sto] = (np.mean(dataSet[sta - overlap : sto + overlap]))
    #         else:
    #             dataSet[sta:sto] = muUB

    for sta, sto in zip(staIdx, stoIdx):
        if np.mean(dataSet[sta:sto]) > muUB - 0.7 * np.abs(muUB - muBO):
            if (sta > overlap) & (sto < len(dataSet) - overlap):
                dataSet[sta:sto] = 0.5 * (
                    (np.mean(dataSet[sta - overlap : sto + overlap])) + dataSet[sta:sto]
                )
            else:
                dataSet[sta:sto] = muUB

    # filter false end of events events
    shortEvents = hmmEventDF[
        (
            (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
            & (hmmEventDF["DELTALEVEL"] != 0)
            & (
                (hmmEventDF["STATELABEL"] == UBstate)
                | (hmmEventDF["STATELABEL"] == BNOstate)
            )
        )
    ]
    staIdx = shortEvents["STATESTARTIDX"].values
    stoIdx = shortEvents["STATESTOPIDX"].values

    # for sta, sto in zip(staIdx,stoIdx):
    #     if np.mean(dataSet[sta:sto]) < muBO + 0.8 * np.abs(muUB - muBO):
    #         if (sta > 10) & (sto < len(dataSet) - overlap):
    #             dataSet[sta:sto] = (np.mean(dataSet[sta - overlap : sto + overlap]))
    #         else:
    #             dataSet[sta:sto] = muBO

    for sta, sto in zip(staIdx, stoIdx):
        if np.mean(dataSet[sta:sto]) < muBO + 0.7 * np.abs(muUB - muBO):
            if (sta > overlap) & (sto < len(dataSet) - overlap):
                dataSet[sta:sto] = 0.5 * (
                    (np.mean(dataSet[sta - overlap : sto + overlap])) + dataSet[sta:sto]
                )
            else:
                dataSet[sta:sto] = muBO

    return dataSet


def filter_spikes_ramp(seg, rc, dataSet, model, modelDict):
    """
    removes very short spike like hmm states (events shorter than: 15.0 / rc.args.filter) if not deep enough. Depth required to be considered real: >=0.8 of HMM step size
    replaces current trace spike and overhang with ramped values of the spike to originating baseline deviation.

    Parameters
    ----------
    dataSet : np.array 
        of data for which to calculate the standard deviation
    seg : segment class data
    rc : run constant class data
    model : pomegranate model
    modelDict : dict

    Returns
    -------
    dataSet : np.array
        current data without spikes

    """
    seg.hmmTrace, seg.hmmState = get_model_trace(dataSet, model)
    seg.modelDict = modelDict

    hmmEventDF = get_dwell_times(seg, rc)

    shortEventsThreshold = 4.0 / rc.args.filter * 1e6  # seg.samplerate
    overlap = np.int(shortEventsThreshold * 1e-6 * seg.samplerate * 5)
    eventDepthFraction = 0.7
    baseLevelFraction = 0.3
    offsetP = 2.0
    logging.info(
        "Spike filtering HMM events length {s2:.2f} of {s1}".format(
            s1=seg.dfIndex(), s2=shortEventsThreshold
        )
    )

    UBOstate = 4
    BNOstate = 4
    SUBstate = 4
    SBOstate = 4
    D2state = 2
    UBDstate = 4
    for i in range(0, len(modelDict["states"])):
        if modelDict["states"][np.str(i)] == "BO":
            BOstate = i
            muBO = hmmEventDF[hmmEventDF["STATELABEL"] == BOstate][
                "STATELEVEL"
            ].unique()
        if modelDict["states"][np.str(i)] == "UB":
            UBstate = i
            muUB = hmmEventDF[hmmEventDF["STATELABEL"] == UBstate][
                "STATELEVEL"
            ].unique()
        if modelDict["states"][np.str(i)] == "BNO":
            BNOstate = i
        if modelDict["states"][np.str(i)] == "UBO":
            UBOstate = i
        if modelDict["states"][np.str(i)] == "SBO":
            SBOstate = i
        if modelDict["states"][np.str(i)] == "SUB":
            SUBstate = i
        if modelDict["states"][np.str(i)] == "D2":
            D2state = i
        if modelDict["states"][np.str(i)] == "UBD":
            UBDstate = i

    # DOWN spikes
    if "OLDnew4" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                )
            )
        ]
    elif "five" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                )
            )
        ]
    elif "tri4" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                )
            )
        ]
    elif "new5" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBD-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                )
            )
        ]
    elif "sixstate" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-SUB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-SBO")
                )
            )
        ]
    else:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BO-2-BNO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UB")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-BNO")
                )
            )
        ]
    staIdx = shortEvents["STATESTARTIDX"].values
    stoIdx = shortEvents["STATESTOPIDX"].values

    for sta, sto in zip(staIdx, stoIdx):
        if np.mean(dataSet[sta:sto]) > muUB - eventDepthFraction * np.abs(muUB - muBO):
            overlap = np.int((sto - sta) * 2)
            if (sta > 2 * overlap) & (sto < len(dataSet) - 2 * overlap):
                if np.mean(
                    dataSet[sta - 2 * overlap : sto + 2 * overlap]
                ) > muUB - baseLevelFraction * np.abs(muUB - muBO):
                    offset = offsetP * np.abs(
                        np.mean(dataSet[sta - overlap : sto + overlap]) - muUB
                    )
                    tmpArray = (
                        np.arange(overlap + (sto - sta) // 2)
                        * offset
                        / (overlap + (sto - sta) // 2)
                    )
                    if (
                        len(dataSet[sta - overlap : sto + overlap])
                        == 1 + len(tmpArray) * 2
                    ):
                        offsetArray = np.append(
                            np.append(tmpArray, tmpArray[-1]), tmpArray[::-1]
                        )
                    else:
                        offsetArray = np.append(tmpArray, tmpArray[::-1])
                    dataSet[sta - overlap : sto + overlap] = (
                        dataSet[sta - overlap : sto + overlap] + offsetArray
                    )

    # UP spikes
    if "OLDnew4" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-BO")
                )
            )
        ]
    elif "five" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UB-2-UBO")
                )
            )
        ]
    elif "tir4" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UB-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UB-2-UBO")
                )
            )
        ]
    elif "new5" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "UBO-2-UBD")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-BO")
                )
            )
        ]
    elif "sixstate" in rc.args.hmm:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "SUB-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UB-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "SBO-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-BO")
                )
            )
        ]
    else:
        shortEvents = hmmEventDF[
            (
                (hmmEventDF["STATEDWELLTIME"] <= shortEventsThreshold)
                & (hmmEventDF["DELTALEVEL"] != 0)
                & (
                    (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UB-2-BO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "UB-2-UBO")
                    | (hmmEventDF["TRANSITIONLABEL"] == "BNO-2-UBO")
                )
            )
        ]
    staIdx = shortEvents["STATESTARTIDX"].values
    stoIdx = shortEvents["STATESTOPIDX"].values

    for sta, sto in zip(staIdx, stoIdx):
        if np.mean(dataSet[sta:sto]) < muBO + eventDepthFraction * np.abs(muUB - muBO):
            overlap = np.int((sto - sta) * 2)
            if (sta > 2 * overlap) & (sto < len(dataSet) - 2 * overlap):
                if np.mean(dataSet[sta - 2 * overlap : sto + 2 * overlap]) < muUB - (
                    1 - baseLevelFraction
                ) * np.abs(muUB - muBO):
                    offset = offsetP * (
                        np.mean(dataSet[sta - overlap : sto + overlap]) - muBO
                    )
                    tmpArray = (
                        np.arange(overlap + (sto - sta) // 2)
                        * offset
                        / (overlap + (sto - sta) // 2)
                    )
                    if (
                        len(dataSet[sta - overlap : sto + overlap])
                        == 1 + len(tmpArray) * 2
                    ):
                        offsetArray = np.append(
                            np.append(tmpArray, tmpArray[-1]), tmpArray[::-1]
                        )
                    else:
                        offsetArray = np.append(tmpArray, tmpArray[::-1])
                    dataSet[sta - overlap : sto + overlap] = (
                        dataSet[sta - overlap : sto + overlap] - offsetArray
                    )

    return dataSet


def slow_hmm_model(seg, rc, hmmModel):
    """
    this function reduces the transition rates on all edges except the one pointing to the same state. 
    This is done to make sure not too fast rates are picked up from spikes in data.

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmModel : pomegranate model

    Returns
    -------
    hmmModelFrozen : pomegranate model
    
    """

    modelJson = hmmModel.to_json()
    tmpDict = json.loads(modelJson)

    edgeArray = np.array(tmpDict["edges"])
    for i in range(0, len(edgeArray)):
        if (edgeArray[i, 0] != edgeArray[i, 1]) & (edgeArray[i, 2] > 0.1):
            edgeArray[i, 2] /= 5
    tmpDict["edges"] = edgeArray.tolist()

    # modelFrozen = True
    hmmModelFrozen = pgm.HiddenMarkovModel.from_json(
        json.dumps(tmpDict, separators=(",", " : "), indent=4)
    )

    logging.info("slowing transitions of HMM: {s1}".format(s1=seg.dfIndex()))

    return hmmModelFrozen


def set_hmm_model(
    seg,
    rc,
    hmmModel,
    baselineStateStd,
    eventStateStd,
    baselineStateMean,
    eventStateMean,
):
    """
    this function freezes the hmm model distributions to baseline mean and event mean.
    Only thing that can be update is the transition probabilities

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmModel : pomegranate model
    baselineStateStd : np.float
    eventStateStd : np.float
    baselineStateMean : np.float
    eventStateMean : np.float

    Returns
    -------
    hmmModelFrozen : pomegranate model
    
    """

    modelJson = hmmModel.to_json()
    tmpDict = json.loads(modelJson)

    for i in range(0, len(tmpDict["states"])):
        if tmpDict["states"][i]["name"] == "BO":
            muBO = tmpDict["states"][i]["distribution"]["parameters"][0]
            BOstate = i
        elif tmpDict["states"][i]["name"] == "UB":
            muUB = tmpDict["states"][i]["distribution"]["parameters"][0]
            UBstate = i
        elif tmpDict["states"][i]["name"] == "BNO":
            muBNO = tmpDict["states"][i]["distribution"]["parameters"][0]
            BNOstate = i
        elif tmpDict["states"][i]["name"] == "UBO":
            muUBO = tmpDict["states"][i]["distribution"]["parameters"][0]
            UBOstate = i
        elif tmpDict["states"][i]["name"] == "SBO":
            muSBO = tmpDict["states"][i]["distribution"]["parameters"][0]
            SBOstate = i
        elif tmpDict["states"][i]["name"] == "SUB":
            muSUB = tmpDict["states"][i]["distribution"]["parameters"][0]
            SUBstate = i
        elif tmpDict["states"][i]["name"] == "D2":
            muSUB = tmpDict["states"][i]["distribution"]["parameters"][0]
            D2state = i
        elif tmpDict["states"][i]["name"] == "UBD":
            muUBD = tmpDict["states"][i]["distribution"]["parameters"][0]
            UBDstate = i

    if isinstance(eventStateMean, np.ndarray):
        eventStateMean = eventStateMean.tolist()
    if isinstance(baselineStateMean, np.ndarray):
        baselineStateMean = baseline.tolist()

    tmpDict["states"][BOstate]["distribution"]["parameters"][0] = eventStateMean
    tmpDict["states"][BOstate]["distribution"]["parameters"][1] = eventStateStd
    tmpDict["states"][BOstate]["distribution"]["frozen"] = True

    if ("four" in rc.args.hmm) or ("longlinear" in rc.args.hmm) or ("asy4state" in rc.args.hmm) or ("tri4" in rc.args.hmm):
        tmpDict["states"][UBOstate]["distribution"]["parameters"][0] = eventStateMean
        tmpDict["states"][UBOstate]["distribution"]["parameters"][1] = eventStateStd
        tmpDict["states"][UBOstate]["distribution"]["frozen"] = True
        tmpDict["states"][BNOstate]["distribution"]["parameters"][0] = baselineStateMean
        tmpDict["states"][BNOstate]["distribution"]["parameters"][1] = baselineStateStd
        tmpDict["states"][BNOstate]["distribution"]["frozen"] = True

    if "five" in rc.args.hmm:
        tmpDict["states"][UBOstate]["distribution"]["parameters"][0] = eventStateMean
        tmpDict["states"][UBOstate]["distribution"]["parameters"][1] = eventStateStd
        tmpDict["states"][UBOstate]["distribution"]["frozen"] = True
        tmpDict["states"][D2state]["distribution"]["parameters"][0] = eventStateMean
        tmpDict["states"][D2state]["distribution"]["parameters"][1] = eventStateStd
        tmpDict["states"][D2state]["distribution"]["frozen"] = True

    if "new5" in rc.args.hmm:
        tmpDict["states"][UBOstate]["distribution"]["parameters"][0] = baselineStateMean
        tmpDict["states"][UBOstate]["distribution"]["parameters"][1] = baselineStateStd
        tmpDict["states"][UBOstate]["distribution"]["frozen"] = True
        tmpDict["states"][UBDstate]["distribution"]["parameters"][0] = eventStateMean
        tmpDict["states"][UBDstate]["distribution"]["parameters"][1] = eventStateStd
        tmpDict["states"][UBDstate]["distribution"]["frozen"] = True

    if "new4" in rc.args.hmm:
        tmpDict["states"][UBOstate]["distribution"]["parameters"][0] = eventStateMean
        tmpDict["states"][UBOstate]["distribution"]["parameters"][1] = eventStateStd
        tmpDict["states"][UBOstate]["distribution"]["frozen"] = True
        tmpDict["states"][BNOstate]["distribution"]["parameters"][0] = baselineStateMean
        tmpDict["states"][BNOstate]["distribution"]["parameters"][1] = baselineStateStd
        tmpDict["states"][BNOstate]["distribution"]["frozen"] = True

    if "sixstate" in rc.args.hmm:
        tmpDict["states"][UBOstate]["distribution"]["parameters"][0] = eventStateMean
        tmpDict["states"][UBOstate]["distribution"]["parameters"][1] = eventStateStd
        tmpDict["states"][SBOstate]["distribution"]["parameters"][0] = baselineStateMean
        tmpDict["states"][SBOstate]["distribution"]["parameters"][1] = baselineStateStd
        tmpDict["states"][SUBstate]["distribution"]["parameters"][0] = baselineStateMean
        tmpDict["states"][SUBstate]["distribution"]["parameters"][1] = baselineStateStd
        tmpDict["states"][UBOstate]["distribution"]["frozen"] = True
        tmpDict["states"][SUBstate]["distribution"]["frozen"] = True
        tmpDict["states"][SBOstate]["distribution"]["frozen"] = True

    if not ("two" in rc.args.hmm):
        tmpDict["states"][BNOstate]["distribution"]["parameters"][0] = baselineStateMean
        tmpDict["states"][BNOstate]["distribution"]["parameters"][1] = baselineStateStd
        tmpDict["states"][BNOstate]["distribution"]["frozen"] = True

    tmpDict["states"][UBstate]["distribution"]["parameters"][0] = baselineStateMean
    tmpDict["states"][UBstate]["distribution"]["parameters"][1] = baselineStateStd
    tmpDict["states"][UBstate]["distribution"]["frozen"] = True

    # modelFrozen = True
    hmmModelFrozen = pgm.HiddenMarkovModel.from_json(
        json.dumps(tmpDict, separators=(",", " : "), indent=4)
    )

    logging.info("Fixing distributions of HMM to data: {s1}".format(s1=seg.dfIndex()))

    return hmmModelFrozen


def build_and_fit_hmm_model(seg, rc):
    """
    main function that builds the hmm model, and fits it to the data, returns modeldict, model trace, transition probabilities and cleaned dataset
    Options:
    * rc.args.twostephmm : fit hmm model in 2 steps by first subsampled data then full data as specified
    * rc.args.fixedDistributions: freeze distribution of states either to Gaussion Mixture fit or base on lower passed filtered events found in initializaton of filter

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data

    Returns
    -------
    modelTrace : np.array
        idealized trace of hmm
    state : np.array
        state at each trace point
    hmmLogTransProb : np.array (size state x state)
        log transition probability from state i to j hmmLogTransProb[i,j]
    dataSet : np.array
        current data correct by baseline correction

    """
    hmmModel, modelDict = build_hmm_model(seg, rc)

    logging.info("Fitting HMM to {s1} ---- INITIALIZE".format(s1=seg.dfIndex()))

    dataSet = seg.Ia

    if rc.args.twostephmm:
        downsampledDataSet = lowpass_data(seg, rc, dataSet)
        hmmModel = initialize_hmm_model(seg, rc, downsampledDataSet, hmmModel)

    if rc.args.fixedDistributions:
        if rc.args.gmmdistributionfit:
            try:
                (
                    baselineStateStd,
                    eventStateStd,
                    baselineStateMean,
                    eventStateMean,
                ) = hmm_gaussian_classifier(
                    dataSet, seg, rc, "down"
                )
            except:
                logging.info("Fitting Gaussian mixture to initilize HMM failed {s1}".format(s1=seg.dfIndex()))
                (
                    baselineStateStd,
                    eventStateStd,
                    baselineStateMean,
                    eventStateMean,
                ) = calculate_baseline_std(
                    seg, rc, downsampledDataSet, dataSet, hmmModel, modelDict
                )
        else:
            (
                baselineStateStd,
                eventStateStd,
                baselineStateMean,
                eventStateMean,
            ) = calculate_baseline_std(
                seg, rc, downsampledDataSet, dataSet, hmmModel, modelDict
            )

        hmmModel = set_hmm_model(
            seg,
            rc,
            hmmModel,
            baselineStateStd,
            eventStateStd,
            baselineStateMean,
            eventStateMean,
        )

    logging.info("Fitting HMM to {s1} -------- START".format(s1=seg.dfIndex()))
    hmmModel.fit(
        [dataSet[0 : evaluate_hmm_index(dataSet, rc)]],
        algorithm="baum-welch",
        min_iterations=20,
        max_iterations=400,
        lr_decay=0.5,
        distribution_inertia=0.5,
        use_pseudocount=True,
        emission_pseudocount=1,
        n_jobs=2,
    )
    
    if not rc.args.noHmmSpikeFilter:
        dataSet = filter_spikes_ramp(seg, rc, dataSet, hmmModel, modelDict)
        hmmModel = slow_hmm_model(seg, rc, hmmModel)
        hmmModel.fit(
            [dataSet[0 : evaluate_hmm_index(dataSet, rc)]],
            algorithm="baum-welch",
            min_iterations=200,
            max_iterations=500,
            lr_decay=0.5,
            distribution_inertia=0.5,
            use_pseudocount=True,
            emission_pseudocount=1,
            n_jobs=5,
        )
        dataSet = filter_spikes_avg(seg, rc, dataSet, hmmModel, modelDict)
        hmmModel.fit(
            [dataSet[0 : evaluate_hmm_index(dataSet, rc)]],
            algorithm="baum-welch",
            min_iterations=200,
            max_iterations=500,
            lr_decay=0.5,
            distribution_inertia=0.5,
            use_pseudocount=True,
            emission_pseudocount=1,
            n_jobs=5,
        )

    ## final trace
    modelTrace, state = get_model_trace(dataSet, hmmModel)

    ## check trace there, if not run again with thawed distribution
    if np.isnan(state).any():
        hmmModel.thaw()
        hmmModel.fit(
            [dataSet[0 : evaluate_hmm_index(dataSet, rc)]],
            algorithm="baum-welch",
            min_iterations=200,
            max_iterations=500,
            lr_decay=0.5,
            distribution_inertia=0.8,
            use_pseudocount=True,
            emission_pseudocount=1,
            n_jobs=5,
        )
        logging.info(
            "Fitting hidden Markov model to {s1} ------- Thaw Again".format(
                s1=seg.dfIndex()
            )
        )
        modelTrace, state = get_model_trace(dataSet, hmmModel)

    hmmLogTransProb = hmmModel.dense_transition_matrix()

    logging.info(
        "Fitting hidden Markov model to {s1} ------------- END".format(s1=seg.dfIndex())
    )

    evaluate_hmm_model(seg, dataSet, hmmModel, modelDict)

    return modelTrace, state, hmmLogTransProb, modelDict, dataSet, hmmModel


def evaluate_hmm_model(seg, dataSet, hmmModel, hmmModelDict):
    """

    calculates BIC for the model based on rc.args.fitfraction argument (using evaluate_hmm_index())

    outputs BIC to statsDF
    
    Parameters
    ----------
    dataSet : np.array of current trace
    hmmModel : fitted hmm model instance (pomegranate)

    Returns
    -------
    none

    """

    numberOfStates = (
        len(hmmModelDict["states"].keys()) - 2
    )  # remove start and end state
    numberOfTransitions = (
        len(hmmModelDict["transitions"].keys()) - 2
    )  # remove transitions from start state
    numberOfDOFperState = 2  # for normal distribution
    logLikelyHood = hmmModel.log_probability(
        dataSet[evaluate_hmm_index(dataSet, seg.rc) :]
    )
    ML, vPath = hmmModel.viterbi(dataSet[evaluate_hmm_index(dataSet, seg.rc) :])
    DOF = numberOfTransitions + numberOfStates * numberOfDOFperState
    BIC = -2 * logLikelyHood + np.log(len(dataSet / (seg.samplerate / seg.rc.args.filter))) * DOF
    seg.add_to_statsDF("BIC", BIC)
    seg.add_to_statsDF("DOF", DOF)
    seg.add_to_statsDF("MLV", ML)
    seg.add_to_statsDF("logLikelyHood", logLikelyHood)
    return


def get_model_trace(data, model):
    """
    gets the state assignement of the hmm model for each point of the trace, returns np.array modelTrace  - the new trace (average of state distribution) and state np.array - the state numberfor each point on the trace
    Parameters
    ----------
    data : np.array
        of current data to classify
    model : pomegranate hmm model
        model for classification

    Returns
    -------
    modelTrace : np.array
        of idealized current trace
    state : np.array
        classified state for each point of data

    """
    # stateProbability = model.predict_proba(data)
    # state = np.array(np.argmax(stateProbability,1), dtype='int')
    # del stateProbability
    # modelTrace = np.array([model.states[s].distribution.parameters[0] for s in state])

    state = np.array(model.predict(data, algorithm="map"), dtype="int")
    modelTrace = np.empty_like(state, dtype=np.float)
    mean = np.mean(data)
    i = 0
    for s in state:
        try:
            modelTrace[i] = model.states[s].distribution.parameters[0]
        except:
            modelTrace[i] = mean
        i = i + 1
    if len(modelTrace) > len(data):
        modelTrace = modelTrace[1:]

    return modelTrace, state


def get_dwell_times(seg, rc):
    """

    calculates dwell times of each state and returns a huge dataframe containing all the states and the corresponding transitions

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data

    Returns
    -------
    hmmEventDF : pandas DataFrame
        'STATELABEL': current state number
        'STATESTARTIDX': start index of state
        'STATESTOPIDX': stop index of state
        'STATEDWELLTIME': duration of state in [us]
        'TRANSITIONLABEL': transition string e.g. -1 for state change from state 1 to 0
        'STATELEVEL': distribution mean value of state

    """

    logging.info(
        "Calculating state dwell times of HMM of {s1}".format(s1=seg.dfIndex())
    )

    # squatre the state to get unique steps between them
    statesSq = np.power(seg.hmmState, 2, dtype=int)

    # this vctor is 0 when maintaining state and changes to a state transition specific value when state is changed
    changeState = -statesSq[0:-1] + statesSq[1:]

    # this gets all indeices of transitions occuring
    stateTransitions = np.nonzero(changeState)[0]

    # this calculates the duration of each state (index spaceing)
    stateLength = (
        (
            np.concatenate([stateTransitions, [len(seg.hmmState)]])
            - np.concatenate([[0], stateTransitions])
        )
        / seg.samplerate
        * 1e6
    )

    # this gets the label of each transition
    transitionLabel = changeState[stateTransitions]
    transitionLabel = [
        seg.modelDict["transitions"].get("{s:d}".format(s=i), "NOT-ALLOWED")
        for i in transitionLabel
    ]

    # this gets the label of each state
    stateLabel = np.concatenate([seg.hmmState[stateTransitions], [seg.hmmState[-1]]])

    # to get the state level
    stateLevel = np.concatenate([seg.hmmTrace[stateTransitions], [seg.hmmTrace[-1]]])

    # saves duration of each state and the next transition
    hmmEventDF = pd.DataFrame(
        data=list(
            zip(
                stateLabel,
                np.concatenate([[0], stateTransitions]),
                np.concatenate([stateTransitions, [len(seg.hmmState)]]),
                stateLength,
                np.concatenate([transitionLabel, [-100]]),
                stateLevel,
                (
                    np.concatenate([stateLevel, [stateLevel[-1]]])
                    - np.concatenate([[stateLevel[0]], stateLevel])
                )[1:],
            )
        ),
        columns=[
            "STATELABEL",
            "STATESTARTIDX",
            "STATESTOPIDX",
            "STATEDWELLTIME",
            "TRANSITIONLABEL",
            "STATELEVEL",
            "DELTALEVEL",
        ],
    )
    hmmEventDF.reset_index()

    return hmmEventDF


def calculate_rates(seg):
    P = seg.hmmLogTransProb
    R = P - np.diag(np.diag(P))
    I = np.diag(np.log(np.diag(P)) * seg.samplerate)
    scale = np.divide(1, np.sum(R, axis=1)) * np.diag(I)
    rateM = I - np.multiply(R, scale[:, np.newaxis])
    return rateM


def get_hmm_transitions(seg, rc, hmmEventsDF):
    """
    converts transition probability matrix seg.hmmLogTransProb and adds them to hmmEvenvts dataframe columns

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmEventsDF : pandas dataframe
        containing dwell time of each state

    Returns
    -------
    hmmEventsDF : pandas DataFrame
        adds columns: HMM_TR_MODEL_' + transitionLabel
    """

    transitionKKeys = []
    transitionKValues = []
    for i in seg.modelDict["transitionsMidx"].keys():
        rateM = calculate_rates(seg)
        transitionKValues.append(rateM[json.loads(i)[0], json.loads(i)[1]])
        transitionKKeys.append("HMM_TR_K_" + seg.modelDict["transitionsMidx"][i])

    transitionCounts = (
        hmmEventsDF.groupby("TRANSITIONLABEL")["STATESTARTIDX"].count().values
    )
    transitionCountKeys = (
        hmmEventsDF.groupby("TRANSITIONLABEL")["STATESTARTIDX"].count().keys().values
    )
    transitionCountKeys = ["HMM_TR_COUNTS_" + str(i) for i in transitionCountKeys]

    transitionKKeys.extend(transitionCountKeys)
    values = np.append(transitionKValues, transitionCounts)

    DF = pd.DataFrame(data=list([values]), columns=transitionKKeys)
    hmmEventsDF = hmmEventsDF.join(DF.reset_index())

    return hmmEventsDF


def get_hmm_path(seg, rc, hmmEventsDF):
    """
    gets path through hmm model to BO state (two previous states)
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmEventsDF : pandas dataframe
        containing dwell time of each state

    Returns
    -------
    hmmEventsDF : pandas DataFrame
        adds column with 'PATHCOUNTS' which contains the current and the previous 2 states
    """
    modelDict = seg.modelDict
    for i in range(0, len(modelDict["states"])):
        if modelDict["states"][np.str(i)] == "BO":
            BOstate = i

    if hmmEventsDF["STATELABEL"].size > 4:
        tmp = [
            modelDict["states"][np.str(hmmEventsDF.loc[i - 2, "STATELABEL"])]
            + "-"
            + modelDict["states"][np.str(hmmEventsDF.loc[i - 1, "STATELABEL"])]
            + "-"
            + modelDict["states"][np.str(hmmEventsDF.loc[i, "STATELABEL"])]
            for i in np.arange(2, hmmEventsDF["STATELABEL"].size)
        ]
        tmp.insert(0, "0")
        tmp.insert(0, "0")
        hmmEventsDF["PATH"] = tmp
        df = hmmEventsDF.groupby(["PATH"]).size().reset_index(name="PATHCOUNTS")
        df = df[df["PATH"].str.endswith("BO")].set_index("PATH")

        hmmEventsDF = hmmEventsDF.join(df.T.reset_index(drop=True))

    return hmmEventsDF


def get_hmm_type(seg, rc, hmmEventsDF):
    """
    converts transition probability matrix seg.hmmLogTransProb and adds them to hmmEvenvts dataframe columns
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmEventsDF : pandas dataframe
        containing dwell time of each state

    Returns
    -------
    hmmEventsDF : pandas DataFrame
        adds column of HMMTPYE 

    """
    DF = pd.DataFrame(data=list([rc.args.hmm]), columns=["HMMTYPE"])
    hmmEventsDF = hmmEventsDF.join(DF)

    return hmmEventsDF


def get_state_probability(seg, rc, hmmEventDF):
    """
    calculates state probability based on sum of dwell times for each state

    Output:
    populates hmmEventsDF

    Parameters
    ----------
    seg : slice data class object
    rc : runconstant data class object
    hmmEventDF : pandas dataframe
        containing dwell time of each state

    Returns
    -------
    hmmEventDF : pandas DataFrame
        update pandas dataframe with new columns for the state probability
    """

    hmmStatesGroupedDF = (
        hmmEventDF.groupby(["STATELABEL"]).agg({"STATEDWELLTIME": "sum"})
        / hmmEventDF["STATEDWELLTIME"].sum()
    )
    tmpL = []
    print(hmmStatesGroupedDF)
    for state in hmmStatesGroupedDF.index:
        print(state)
        try:
            tmpState = seg.modelDict["states"][np.str(state)]
        except:
            tmpState = 'NOTALLOWED'
        tmpL.append("P_{}".format(tmpState))
            
    print(tmpL)
    state_label = list(tmpL)
    tmpDF = pd.DataFrame(
        data=[hmmStatesGroupedDF["STATEDWELLTIME"].values], columns=state_label
    )
    hmmEventDF = hmmEventDF.join(tmpDF, how="outer")
    return hmmEventDF


def state_dwell_time_fit_wrapper(seg, rc, hmmEventDF):
    """
    Wrapper function for state_dwell_time_fit  does the event dwell time survival fit to different for the different states

    At the moment a single exponential is fitted:
    * single exponential decay: A exp(-k t)

    Output:
    plot with survival histogram of parameter with fitted curve and fitting valu[e[[s

    Parameters
    ----------
    seg : slice data class object
    rc : runconstant data class object
    hmmEventDF : pandas dataframe
        containing dwell time of each state

    Returns
    -------
    hmmEventDF : pandas DataFrame
        update pandas dataframe with new columns retrieved in state_dwell_time_fit
    """

    newPath = rc.resultsPath / "hmmkvalues"
    newPath.mkdir(parents=True, exist_ok=True)
    hmmStatesGroupedDF = hmmEventDF.groupby(["TRANSITIONLABEL"])
    tmpDF = pd.DataFrame()
    for name, group in hmmStatesGroupedDF:
        eventFileName = seg.dfIndex() + "_TR_" + str(name)
        fittedKDF = state_dwell_time_fit(group, seg, rc, newPath / eventFileName)
        tmpDF = tmpDF.join(fittedKDF, how="outer")

    hmmEventDF = hmmEventDF.join(tmpDF.reset_index())
    return hmmEventDF


def state_dwell_time_fit(allEventLog, seg, rc, writeFile):
    """
    This function does the event dwell time survival fit for state

    At the moment a single exponential is fitted:
    * single exponential decay: A exp(-k t)

    Output:
    plot with survival histogram of parameter with fitted curve and fitting values
    populates seg.statsDF to add fitting parameter, as well as reduced chi, error of fitting parameter (1std)

    Parameters
    ----------
    allEventsLog : pd.dataframe
        all the single event parameter. Can be read from *_events.csv
    seg : slice data class object
    rc : runconstant data class object
    writeFile : Pathlib Path

    Returns
    -------
    DF : pandas dataframe
        with columns 'HMM_TR_FIT_'+transitionLabel + 'exp1_decay' and 'HMM_TR_FIT_'+transitionLabel + 'exp1_amplitude' from single exponential fit of dwell time histogram
    """

    # calculated cumulative event count for each case:
    logging.info(
        "Dwell time analysis and survival time fit for HMM {s0} transition {s1}".format(
            s0=writeFile.stem, s1=allEventLog["TRANSITIONLABEL"].unique()[0]
        )
    )

    dwellTime = "STATEDWELLTIME"
    dwellTimeRank = allEventLog[dwellTime].value_counts().sort_index(ascending=False)
    dwellTimeRank = dwellTimeRank.to_frame().reset_index()
    dwellTimeRank.rename(columns={dwellTime: "COUNT"}, inplace=True)
    dwellTimeRank.rename(columns={"index": dwellTime}, inplace=True)
    dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"] = dwellTimeRank.COUNT.cumsum()
    totalCount = dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"].max()

    # rejection of very long dwell time events for fitting
    # if not np.isnan(totalCount):
    #     thresholdDwellTime = 4.0 / rc.args.filter
    #     if dwellTimeRank[dwellTime].values[0] > thresholdDwellTime:
    #         logging.info('{s0}: DROPPING short states EVENTS because shorter than {s1:.2e} '.format(s0=writeFile.stem, s1=thresholdDwellTime))
    #         dwellTimeRank = dwellTimeRank[dwellTimeRank[dwellTime] >= thresholdDwellTime]
    # q_hi = dwellTimeRank[dwellTime].quantile(0.99)
    # if (dwellTimeRank[dwellTime] < q_hi).any():
    #     print('{s0}: DROPPING LONG EVENTS because of Quantile'.format(s0=writeFile.stem))
    # dwellTimeRank = dwellTimeRank[dwellTimeRank[dwellTime] < q_hi]
    # no fitting if too little events counted
    if len(dwellTimeRank.index) <= 3:
        logging.info("{s0}: No fit too few events".format(s0=writeFile.stem))

    # DF = pd.DataFrame(data = list(values), columns = keys)
    if not np.isnan(
        totalCount
    ):  # if there's something to fit, fit it below and then plot the fits.
        if (dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"].values[-1] > 3) and (
            dwellTimeRank[dwellTime].nunique() > 3
        ):
            # single exponential fit
            out = event_utils.lmFIT_single(
                dwellTimeRank[dwellTime].values * 1e-6,
                dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
                rc.args.filter * 2.0,
            )
            keys, values = lmfit_to_param(
                out, allEventLog["TRANSITIONLABEL"].unique()[0]
            )
            values = [values]
            DF = pd.DataFrame(data=values, columns=keys, index=[[1]])
            # plot of surival fit
            plot_hmm_state_dwellTimeDist(
                dwellTimeRank, writeFile, DF, allEventLog["TRANSITIONLABEL"].unique()[0]
            )

            return DF
    return pd.DataFrame(index=[[1]])


def lmfit_to_param(out, transitionLabel):
    """
    parses fit parameter to key value pair to write to dataframe statsDF
    Parameters
    ----------
    out : lmfit Output
        of single exponetial fitting
    transitionLabel : string
        state transition identifier

    Returns
    -------
    tmpKeys : list of srings
        sort 'HMM_TR_FIT_'+transitionLabel + 'exp1_decay'
    values : list of values
        containg fit paramter exp1_decay and exp1_amplitdue


    """
    keys = out.best_values.keys()
    values = []
    tmpKeys = []
    for key in keys:
        if "decay" in key:
            values.append(1.0 / out.best_values[key])
        else:
            values.append(out.best_values[key])

        keyString = "HMM_TR_FIT_" + str(transitionLabel) + "_"
        tmpKeys.append(keyString + key)

    return tmpKeys, values


def save_hmm_df(seg, rc, hmmEventsDF):
    """
    saves hmmEventsDF to disk

    Output:
        saves hmmEventsDF to disk

    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmEventsDF : pandas dataframe
        containing dwell time of each state and transition rate constants between states

    Returns
    -------
    none

    """

    plotFileName = seg.dfIndex()
    writeFile = rc.resultsPath / plotFileName
    hmmEventsDF.to_csv(
        str(writeFile) + "_hmm_events.csv", sep=",", encoding="utf-8", index=True
    )
    return


def plot_hmm_state_dwellTimeDist(dwellTimeRank, writeFile, DF, transitionLabel):

    """
    plots cumulative dwell time of state and single exponential fit of data


    Output:
        saves survivel plots to disk


    Parameters
    ----------
    dwellTimeRank : pandas Dataframe
        containing cumulative dwell time count of statetransition to plot
    writeFile : posix path
        path and filename to write plot to
    DF : pandas dataframe
        containg fit results
    transitionLabel : string
        transition label string

    Returns
    -------
    none

    """
    dwellTime = "STATEDWELLTIME"

    sns.set()
    sns.set_style("whitegrid")
    figsns, axsns = plt.subplots(figsize=(8, 6))
    seaborn_plot = sns.scatterplot(
        x=dwellTime,
        y="CUMULATIVEDWELLTIMECOUNT",
        data=dwellTimeRank,
        ax=axsns,
        marker="4",
        label="data",
    )

    label = "HMM_TR_FIT_" + str(transitionLabel) + "_"

    sns.lineplot(
        x=dwellTime,
        y=event_utils.single_Exp(
            [
                DF[label + "exp1_amplitude"].values[0],
                DF[label + "exp1_decay"].values[0],
            ],
            dwellTimeRank[dwellTime],
        ),
        data=dwellTimeRank,
        color="red",
        label="single",
    )

    axsns.annotate(
        "single Exp:\n   k = {s0:.2e}".format(s0=DF[label + "exp1_decay"].values[0]),
        xy=(0.15, 0.6),
        fontsize="small",
        xycoords="axes fraction",
    )

    seaborn_plot.set_xlim(auto=True)
    seaborn_plot.set_ylim(auto=True)

    axsns.set(
        xlabel="state duration [$\mu sec$]",
        ylabel="cumulative dwell time count for state transition "
        + str(transitionLabel),
    )

    axsns.set_title("\n".join(wrap(str(writeFile.name))))
    figsns.savefig(str(writeFile) + ".png", dpi=100)
    figsns.clear()
    plt.close(figsns)

    return


def plot_hmm_state_dwellHistogram(seg, rc, hmmEventsDF):
    """
    plots histogram of all state dwell times
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmEventsDF : pd.dataframe 
        contains the state dwell times

    Returns
    -------
    none

    """
    sns.set()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    hmmEventsDF["DT"] = hmmEventsDF["STATEDWELLTIME"] / 1e6
    sns.histplot(
        data=hmmEventsDF,
        x="DT",
        hue="TRANSITIONLABEL",
        binrange=[4.0 / rc.args.filter, 2],
    )

    name = "{s1}_stateDwellHist".format(s1=seg.dfIndex())
    ax.set_title(str(name))
    fig.legend()
    fig.savefig(
        rc.resultsPath / (name + ".png"), dpi=70, format="png", bbox_inches="tight"
    )
    ax.clear()
    fig.clear()
    plt.close(fig)

    return


def plot_hmm_trace(seg, rc):
    """
    plots trace of hmm and data in segments of rc.args.plotidealevents and saves it to disk


    Output:
        saves trace plots with HMM events to disk


    Parameters
    ----------
    seg : segment class data
    rc : run constant class data

    Returns
    -------
    none

    """

    dataSet = seg.baselineCorrectedData
    time = np.arange(0, np.size(dataSet), 1) / seg.samplerate

    # if not math.isnan(rc.args.detrend):
    #    data = data - filter_utils.detrend_data(seg.Ia, int(np.floor(seg.samplerate * rc.args.detrend))) + seg.cur_mean

    plotLengthSec = rc.args.plotidealevents
    plotLength = int(np.floor(seg.samplerate * plotLengthSec))
    dataLength = len(seg.Ia)
    numOfPlots = np.int(np.ceil(dataLength / plotLength))
    if len(seg.hmmTrace) != 0 and not np.isnan(seg.hmmTrace).any():
        yLimits = [
            np.nanmin(seg.hmmTrace) - seg.cur_std * 3,
            np.nanmax(seg.hmmTrace) + seg.cur_std * 3,
        ]
    else:
        yLimits = [np.nanmin(dataSet), np.nanmax(dataSet)]

    logging.info("Generating hmm event plots of " + seg.name)
    for i in range(0, numOfPlots):
        fig, ax = plt.subplots(figsize=(30, 6))
        startIdx = plotLength * i
        endIdx = plotLength * (i + 1)
        if endIdx >= dataLength:
            endIdx = dataLength - 1

        # plot data
        ax.plot(
            time[startIdx:endIdx],
            dataSet[startIdx:endIdx],
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

        if len(seg.hmmTrace) != 0:
            ax.scatter(
                time[startIdx:endIdx],
                seg.hmmTrace[startIdx:endIdx],
                label="idealized event data",
                s=2,
                c=seg.hmmState[startIdx:endIdx],
                cmap=plt.get_cmap("Set1"),
            )
            ax.plot(
                time[startIdx:endIdx],
                seg.hmmTrace[startIdx:endIdx],
                linestyle="-",
                linewidth=0.1,
                color="r",
            )
        name = "{s1}_idealEvent_{s5:0{width}}".format(
            s1=seg.dfIndex(), s5=i, width=int(math.ceil(math.log(numOfPlots + 1, 10)))
        )
        ax.set_title(str(name))
        fig.legend()
        fig.savefig(
            rc.resultsPath / (name + ".png"), dpi=70, format="png", bbox_inches="tight"
        )
        ax.clear()
        fig.clear()
        plt.close(fig)

    return


def save_hmm_model(seg, rc, hmmModel):
    """
    saves json of pomegranate model to disk

        
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmModel : pomegranate model

    Returns
    -------
    none

    """

    name = "{s1}_hmmModel.json".format(s1=seg.dfIndex())
    writeFile = rc.resultsPath / "simulatedHmmData" / name
    writeFile.parent.mkdir(parents=True, exist_ok=True)

    modelJson = hmmModel.to_json()

    with open(writeFile, "w") as f:
        f.write(modelJson)

    return


def simulate_hmm_model(seg, rc, hmmModel):
    """
    simulates one sample trace of hmmModel
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    hmmModel : pomegranate model

    Returns
    -------
    simulatedTrace : np.array

    """

    simulatedTrace = hmmModel.sample(
        n=1, length=len(seg.Ia), path=False, random_state=431
    )

    return simulatedTrace[0]


def save_simulated_hmm_trace(seg, rc, simulatedTrace):
    """
    saves simulated hmm trace as seg.Ia to pickle object using export_utils.save_segment(seg, rc)
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    simulatedTrace : np.array

    Returns
    -------
    none

    """

    iaStore = np.array(seg.Ia, copy=True)
    seg.Ia = simulatedTrace
    export_utils.save_segment(seg, rc.resultsPath / "simulatedHmmData")

    del seg.Ia
    seg.Ia = iaStore

    return


def save_normalized_trace(seg, rc):
    """
    saves baseline corrected and np.mean(Baseline) = 0 version of seg.Ia to pickle object using export_utils.save_segment(seg, rc)
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data

    Returns
    -------
    none

    """

    logging.info("saving normalized dataset {s1}".format(s1=seg.dfIndex()))

    # find state assignments for high and low level
    BNOstate = -100
    UBOstate = -100
    SBOstate = -100
    SUBstate = -100
    D2state = -100
    UBDstate = -100
    for i in range(0, len(seg.modelDict["states"])):
        if seg.modelDict["states"][np.str(i)] == "UB":
            UBstate = i
        if seg.modelDict["states"][np.str(i)] == "BO":
            BOstate = i
        if seg.modelDict["states"][np.str(i)] == "BNO":
            BNOstate = i
        if seg.modelDict["states"][np.str(i)] == "UBO":
            UBOstate = i
        if seg.modelDict["states"][np.str(i)] == "SBO":
            SBOstate = i
        if seg.modelDict["states"][np.str(i)] == "SUB":
            SUBstate = i
        if seg.modelDict["states"][np.str(i)] == "D2":
            D2state = i
        if seg.modelDict["states"][np.str(i)] == "UBD":
            UBDstate = i

    if "sixstate" in rc.args.hmm:
        if BNOstate != -100:
            seg.hmmState[seg.hmmState == BNOstate] = UBstate
        if UBOstate != -100:
            seg.hmmState[seg.hmmState == UBOstate] = BOstate
        if SBOstate != -100:
            seg.hmmState[seg.hmmState == SBOstate] = UBstate
        if SUBstate != -100:
            seg.hmmState[seg.hmmState == SUBstate] = UBstate
    # elif "new4" in rc.args.hmm:
    #     if BNOstate != -100:
    #         seg.hmmState[seg.hmmState == BNOstate] = UBstate
    #     if UBOstate != -100:
    #         seg.hmmState[seg.hmmState == UBOstate] = BOstate
    elif "five" in rc.args.hmm:
        if BNOstate != -100:
            seg.hmmState[seg.hmmState == BNOstate] = UBstate
        if UBOstate != -100:
            seg.hmmState[seg.hmmState == UBOstate] = BOstate
        if D2state != -100:
            seg.hmmState[seg.hmmState == D2state] = BOstate
    elif "new5" in rc.args.hmm:
        if BNOstate != -100:
            seg.hmmState[seg.hmmState == BNOstate] = UBstate
        if UBOstate != -100:
            seg.hmmState[seg.hmmState == UBOstate] = UBstate
        if UBDstate != -100:
            seg.hmmState[seg.hmmState == UBDstate] = BOstate
    else:
        if BNOstate != -100:
            seg.hmmState[seg.hmmState == BNOstate] = UBstate
        if UBOstate != -100:
            seg.hmmState[seg.hmmState == UBOstate] = BOstate

    # savetrue data before overwirting it
    iaStore = np.array(seg.Ia, copy=True)

    # normalize and scale data
    seg.Ia = seg.baselineCorrectedData - np.nanmean(
        seg.baselineCorrectedData[(seg.hmmState[:] == UBstate)]
    )
    seg.hmmTrace = seg.hmmTrace - np.nanmean(
        seg.baselineCorrectedData[(seg.hmmState[:] == UBstate)]
    )

    ## different scaling versions - disabled because of randomes introduced by that.
    # seg.Ia = - seg.Ia / np.nanmean(seg.Ia[(seg.hmmState[1:] != UBstate)])
    # seg.Ia = -seg.Ia / np.nanstd(seg.Ia[(seg.hmmState[:] != UBstate)]) * 0.3 ## changed 20210804
    # seg.Ia = seg.Ia / np.nanstd(seg.baselineCorrectedData[(seg.hmmState[:] == UBstate)]) * 0.3
    # write segment
    export_utils.save_segment(seg, rc.resultsPath / "scaledData" / str(seg.i))

    del seg.Ia
    seg.Ia = iaStore

    return


def read_hmm_model(seg, rc):
    """
    reads .json
    
    Parameters
    ----------
    seg : segment class data
    rc : run constant class data
    simulatedTrace : np.array

    Returns
    -------
    hmmModelFrozen : pomegranate HMM

    """

    name = "{s1}_hmmModel.json".format(s1=seg.dfIndex())
    writeFile = rc.resultsPath.parent / "HmmInput" / name
    # writeFile.parent.mkdir(parents=True, exist_ok=True)

    with open(writeFile, "r") as f:
        hmmModelFrozen = pgm.HiddenMarkovModel.from_json(f.read())

    return hmmModelFrozen
