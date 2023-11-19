import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
from scipy import stats
from scipy import special
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
import argparse

import sys
import distutils.dir_util
import shutil
from pathlib import Path
import os
from datetime import datetime
from datetime import timedelta
import time
from textwrap import wrap
import scipy.optimize
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import multiprocessing
import multiprocessing.pool
from lmfit.models import ExponentialModel
from lmfit.models import PowerLawModel
from lmfit import Model

import scipy.optimize

import argparse

parser = argparse

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def calculate_inter_event_time(allEventLog):
    ## this function calculates dwell times for k_on value fitting from event index
    eventStartIdx = allEventLog.STATESTARTIDX.values
    eventStopIdx = allEventLog.STATESTOPIDX.values
    samplerate = np.mean((eventStopIdx - eventStartIdx) / allEventLog.STATEDWELLTIME)
    # do interevent duration analysis
    noEventDwellTime = (eventStartIdx[1:] - eventStopIdx[:-1]) / samplerate
    allEventLog["INTEREVENTDWELLTIME"] = np.append(noEventDwellTime, np.nan)
    return allEventLog


def calculate_rising_edge_time(allEventLog):
    ## this function calculates times from one to the next rising / falling edge
    eventStartIdx = allEventLog.STATESTARTIDX.values
    eventStopIdx = allEventLog.STATESTOPIDX.values
    samplerate = np.mean((eventStopIdx - eventStartIdx) / allEventLog.STATEDWELLTIME)
    # do interevent duration analysis
    noEventDwellTime = (eventStartIdx[1:] - eventStartIdx[:-1]) / samplerate
    allEventLog["RISINGEDGEDWELLTIME"] = np.append(noEventDwellTime, np.nan)
    return allEventLog


def cut_off_events(dwellTimeRank, device):
    ## removes all events which are slower than the first p_off (hardcoded per device)

    numberOfEvents = dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"].max()
    boundFraction = get_boundfraction(device)
    return dwellTimeRank[
        dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"]
        <= (1 - boundFraction) * numberOfEvents
    ]


def get_device(allEventLog):
    sampleName = "_".join(
        allEventLog["index"].iloc[0].split("Chip_")[1].split("_")[0:2]
    )
    if sampleName == "AV_R4C1":
        device = "Device_1"
    elif sampleName == "AV_R4C4":
        device = "Device_2"
    elif sampleName == "old3_tl":
        device = "Device_3"
    elif sampleName == "old3_bl":
        device = "Device_4"
    elif sampleName == "n2_tr":
        device = "Device_5"
    else:
        device = "unknown"

    return device


def get_boundfraction(device):
    if device == "Device_1":
        boundFraction = 0.14
    elif device == "Device_3":
        boundFraction = 0.3
    else:
        boundFraction = 1.0
    return boundFraction


def event_dwell_time_analysis(allEventLog, writeFile, event="EVENT"):
    ## do event survival time (dwell time) analysis and exponentional fit to get rate constants.

    # calculated cumulative event count for each case:
    # index = allEventLog.loc[0,'index']
    index = str(writeFile.stem)
    DF = pd.DataFrame(index=[index], columns=get_keys())
    if event == "NOEVENT":
        dwellTime = "STATEDWELLTIME"
        allEventLog = allEventLog[
            allEventLog["STATELABEL"] == 1
        ]  # calculate_inter_event_time(allEventLog)
    elif event == "RISINGEDGE":
        dwellTime = "RISINGEDGEDWELLTIME"
        allEventLog = calculate_rising_edge_time(allEventLog)
    elif event == "EVENT":
        allEventLog = allEventLog[allEventLog["STATELABEL"] == 0]  #
        dwellTime = "STATEDWELLTIME"

    logging.info(
        "{s1} dwell time analysis and survival time fit for {s0}".format(
            s0=writeFile.stem, s1=event
        )
    )
    dwellTimeRank = allEventLog[dwellTime].value_counts().sort_index(ascending=False)
    dwellTimeRank = dwellTimeRank.to_frame().reset_index()
    dwellTimeRank.rename(columns={dwellTime: "COUNT"}, inplace=True)
    dwellTimeRank.rename(columns={"index": dwellTime}, inplace=True)
    dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"] = dwellTimeRank.COUNT.cumsum()

    # cuf off slow events according to bound fraction to extract k_fast more robust
    # dwellTimeRank = cut_off_events(dwellTimeRank, get_device(allEventLog))

    # rejection of very long dwell time events for fitting
    totalCount = dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"].max()
    # if not np.isnan(totalCount):
    #     thresholdDwellTime = dwellTimeRank.loc[dwellTimeRank['CUMULATIVEDWELLTIMECOUNT'] > int(totalCount / 2)][dwellTime].values[0] * 20
    # #print(dwellTimeRank.index)
    #     if dwellTimeRank[dwellTime].values[0] > thresholdDwellTime:
    #         print('{s0}: DROPPING LONG EVENTS because longer than {s1:.2e} '.format(s0=writeFile.stem, s1=thresholdDwellTime))
    #         dwellTimeRank = dwellTimeRank[dwellTimeRank[dwellTime] <= thresholdDwellTime]
    #     # q_hi = dwellTimeRank[dwellTime].quantile(0.99)
    #     # if (dwellTimeRank[dwellTime] < q_hi).any():
    #     #     print('{s0}: DROPPING LONG EVENTS because of Quantile'.format(s0=writeFile.stem))
    #     # dwellTimeRank = dwellTimeRank[dwellTimeRank[dwellTime] < q_hi]
    # # no fitting if to little events counted
    if len(dwellTimeRank.index) <= 3:
        print("{s0}: No fit too few events".format(s0=writeFile.stem))
        return DF

    k_guess = 10
    if dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[-1] > 5:
        # single exponential fit
        out = lmFIT_single(
            dwellTimeRank[dwellTime].values * 1e-6,
            dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
            writeFile,
        )
        keys, values = lmfit_to_param(out, event=event)
        DF = add_to_DF(DF, index, keys, values)
        k_guess = 1 / out.best_values["exp1_decay"]

    if dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[-1] > 8:
        # stretchted exponential fit
        out = lmFIT_strtchtd_exponential(
            dwellTimeRank[dwellTime].values,
            dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
            k_guess,
        )
        keys, values = lmfit_strtchtd_to_param(out, event=event)
        DF = add_to_DF(DF, index, keys, values)

        # double exponential fit
        out = lmFIT_double(
            dwellTimeRank[dwellTime].values * 1e-6,
            dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
            k_guess,
        )
        keys, values = lmfit_to_param(out, event=event)
        DF = add_to_DF(DF, index, keys, values)
        k_guess = [
            1.0 / out.best_values["exp1_decay"],
            1.0 / out.best_values["exp2_decay"],
        ]
        A_guess = [out.best_values["exp1_amplitude"], out.best_values["exp2_amplitude"]]
        if k_guess[1] > k_guess[0]:
            k_guess = k_guess[::-1]
            A_guess = A_guess[::-1]

    if dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[-1] > 14:
        out = lmFIT_quadru(
            dwellTimeRank[dwellTime].values * 1e-6,
            dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
            k_guess,
            A_guess,
        )
        keys, values = lmfit_quad_to_param(out, event=event)
        DF = add_to_DF(DF, index, keys, values)

    # plot of surival fit
    if dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[-1] > 5:
        plot_events_dwellTimeDist(
            allEventLog, dwellTimeRank, writeFile, DF, event=event
        )
        plot_quad(allEventLog, dwellTimeRank, writeFile, DF, event=event)

    # DF = add_to_DF(DF, index, 'NUMBEROFEVENTS', dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[0])
    return DF


def quad_Exp(x, dwellTime):
    # quad exponential function with x[0]*exp[-dwell * x[1]] + x[2]*exp[-dwell * x[3]]
    return (
        x[0] * np.exp(-dwellTime * 1e-6 * x[1])
        + x[2] * np.exp(-dwellTime * 1e-6 * x[3])
        + x[4] * np.exp(-dwellTime * 1e-6 * x[5])
        + x[6] * np.exp(-dwellTime * 1e-6 * x[7])
    )


def double_Exp(x, dwellTime):
    # double exponential function with x[0]*exp[-dwell * x[1]] + x[2]*exp[-dwell * x[3]]
    return x[0] * np.exp(-dwellTime * 1e-6 * x[1]) + x[2] * np.exp(
        -dwellTime * 1e-6 * x[3]
    )


def single_Exp(x, dwellTime):
    # single exponential function with x[0]*exp[-dwell * x[1]]
    return x[0] * np.exp(-dwellTime * 1e-6 * x[1])


def plot_quad(allEventLog, dwellTimeRank, writeFile, DF, event="EVENT"):
    # plot event dwell time cumulative event count with events dwell time >0 x

    if event == "NOEVENT":
        dwellTime = "STATEDWELLTIME"
        kType = "ON"

    elif event == "RISINGEDGE":
        dwellTime = "RISINGEDGEDWELLTIME"
        kType = "RISINGEDGE"

    elif event == "EVENT":
        dwellTime = "STATEDWELLTIME"
        kType = "OFF"

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

    sns.lineplot(
        x=dwellTime,
        y=single_Exp(
            [DF["A_" + kType].values[0], DF["K_" + kType + "_single"].values[0]],
            dwellTimeRank[dwellTime],
        ),
        data=dwellTimeRank,
        color="red",
        label="single",
    )
    sns.lineplot(
        x=dwellTime,
        y=double_Exp(
            [
                DF["AdS_" + kType].values[0],
                DF["K_" + kType + "_double_slow"].values[0],
                DF["AdF_" + kType].values[0],
                DF["K_" + kType + "_double_fast"].values[0],
            ],
            dwellTimeRank[dwellTime],
        ),
        data=dwellTimeRank,
        color="green",
        label="double",
    )
    # sns.lineplot(x=dwellTime, y = strtchtd_exponential(dwellTimeRank[dwellTime], DF['A_' + kType + '_stretchted'].values[0],DF['K_' + kType + '_stretchted'].values[0],DF['ALPHA_' + kType + '_stretchted'].values[0]), data = dwellTimeRank, color = 'yellow', label = 'stretchted')

    sns.lineplot(
        x=dwellTime,
        y=quad_Exp(
            [
                DF["K_" + kType + "_exp1_amplitude"].values[0],
                DF["K_" + kType + "_exp1_decay"].values[0],
                DF["K_" + kType + "_exp2_amplitude"].values[0],
                DF["K_" + kType + "_exp2_decay"].values[0],
                DF["K_" + kType + "_exp3_amplitude"].values[0],
                DF["K_" + kType + "_exp3_decay"].values[0],
                DF["K_" + kType + "_exp4_amplitude"].values[0],
                DF["K_" + kType + "_exp4_decay"].values[0],
            ],
            dwellTimeRank[dwellTime],
        ),
        data=dwellTimeRank,
        color="darkblue",
        label="quadrupol",
    )

    axsns.annotate(
        "single Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e} \ndouble Exp: \n   k_slow = {s2:.2e} +- {serrds:.2e}, \n   k_fast = {s3:.2e} +-{serrdf:.2e} with reduced Chi = {s4:.2e}".format(
            s0=DF["K_" + kType + "_single"].values[0],
            serrs=DF["K_" + kType + "_single_1simga"].values[0],
            s1=DF["K_" + kType + "_rchi_single"].values[0],
            s2=DF["K_" + kType + "_double_slow"].values[0],
            serrds=DF["K_" + kType + "_double_slow_1simga"].values[0],
            s3=DF["K_" + kType + "_double_fast"].values[0],
            serrdf=DF["K_" + kType + "_double_fast_1simga"].values[0],
            s4=DF["K_OFF_rchi_double"].values[0],
        ),
        xy=(0.15, 0.6),
        fontsize="small",
        xycoords="axes fraction",
    )

    axsns.annotate(
        "quad Exp:\n   k1 = {s0:.2e}, k2 = {s1:.2e} \n k3 = {s2:.2e}, k4 = {s3:.2e} \nreduced Chi  = {s5:.2e}".format(
            s0=DF["K_" + kType + "_exp1_decay"].values[0],
            s1=DF["K_" + kType + "_exp2_decay"].values[0],
            s2=DF["K_" + kType + "_exp3_decay"].values[0],
            s3=DF["K_" + kType + "_exp4_decay"].values[0],
            s5=DF["K_RISINGEDGE_stretchted_rchi"].values[0],
        ),
        xy=(0.15, 0.8),
        fontsize="small",
        xycoords="axes fraction",
    )

    seaborn_plot.set_xlim(auto=True)
    seaborn_plot.set_ylim(auto=True)

    if event == "EVENT":
        axsns.set(
            xlabel="Event duration [$\mu sec$]", ylabel="k_off cumulative event count"
        )
        tmp = str(writeFile.stem) + "_K_OFF"
        writeFile = writeFile.parent / tmp
    elif event == "NOEVENT":
        axsns.set(
            xlabel="interevent duration [$\mu sec$]",
            ylabel="k_on cumulative interevent count",
        )
        tmp = str(writeFile.stem) + "_K_ON"
        writeFile = writeFile.parent / tmp
    elif event == "RISINGEDGE":
        axsns.set(
            xlabel="RISING EDGE duration [$\mu sec$]",
            ylabel="k_rising cumulative count",
        )
        tmp = str(writeFile.stem) + "_K_RISINGEDGE"
        writeFile = writeFile.parent / tmp

    axsns.set_title("\n".join(wrap(str(writeFile.stem))))
    figsns.savefig(str(writeFile) + "quadEXP.png", dpi=100)
    figsns.clear()
    plt.close(figsns)

    return


def plot_events_dwellTimeDist(allEventLog, dwellTimeRank, writeFile, DF, event="EVENT"):
    # plot event dwell time cumulative event count with events dwell time >0 x

    if event == "NOEVENT":
        dwellTime = "STATEDWELLTIME"
    elif event == "RISINGEDGE":
        dwellTime = "RISINGEDGEDWELLTIME"
    elif event == "EVENT":
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

    if event == "EVENT":
        sns.lineplot(
            x=dwellTime,
            y=single_Exp(
                [DF["A_OFF"].values[0], DF["K_OFF_single"].values[0]],
                dwellTimeRank[dwellTime],
            ),
            data=dwellTimeRank,
            color="red",
            label="single",
        )
        sns.lineplot(
            x=dwellTime,
            y=double_Exp(
                [
                    DF["AdS_OFF"].values[0],
                    DF["K_OFF_double_slow"].values[0],
                    DF["AdF_OFF"].values[0],
                    DF["K_OFF_double_fast"].values[0],
                ],
                dwellTimeRank[dwellTime],
            ),
            data=dwellTimeRank,
            color="green",
            label="double",
        )
        axsns.annotate(
            "single Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e} \ndouble Exp: \n   k_slow = {s2:.2e} +- {serrds:.2e}, \n   k_fast = {s3:.2e} +-{serrdf:.2e} with reduced Chi = {s4:.2e}".format(
                s0=DF["K_OFF_single"].values[0],
                serrs=DF["K_OFF_single_1simga"].values[0],
                s1=DF["K_OFF_rchi_single"].values[0],
                s2=DF["K_OFF_double_slow"].values[0],
                serrds=DF["K_OFF_double_slow_1simga"].values[0],
                s3=DF["K_OFF_double_fast"].values[0],
                serrdf=DF["K_OFF_double_fast_1simga"].values[0],
                s4=DF["K_OFF_rchi_double"].values[0],
            ),
            xy=(0.15, 0.6),
            fontsize="small",
            xycoords="axes fraction",
        )

        sns.lineplot(
            x=dwellTime,
            y=strtchtd_exponential(
                dwellTimeRank[dwellTime],
                DF["A_OFF_stretchted"].values[0],
                DF["K_OFF_stretchted"].values[0],
                DF["ALPHA_OFF_stretchted"].values[0],
            ),
            data=dwellTimeRank,
            color="yellow",
            label="stretchted",
        )
        axsns.annotate(
            "stretchted Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e}, A = {s2:.2e}, alpha = {s3:.2f}".format(
                s0=DF["K_OFF_stretchted"].values[0],
                serrs=DF["K_OFF_stretchted_1sigma"].values[0],
                s1=DF["K_OFF_stretchted_rchi"].values[0],
                s2=DF["A_OFF_stretchted"].values[0],
                s3=DF["ALPHA_OFF_stretchted"].values[0],
            ),
            xy=(0.15, 0.8),
            fontsize="small",
            xycoords="axes fraction",
        )
    elif event == "NOEVENT":
        sns.lineplot(
            x=dwellTime,
            y=single_Exp(
                [DF["A_ON"].values[0], DF["K_ON_single"].values[0]],
                dwellTimeRank[dwellTime],
            ),
            data=dwellTimeRank,
            color="red",
            label="single",
        )
        sns.lineplot(
            x=dwellTime,
            y=double_Exp(
                [
                    DF["AdS_ON"].values[0],
                    DF["K_ON_double_slow"].values[0],
                    DF["AdF_ON"].values[0],
                    DF["K_ON_double_fast"].values[0],
                ],
                dwellTimeRank[dwellTime],
            ),
            data=dwellTimeRank,
            color="green",
            label="double",
        )
        axsns.annotate(
            "single Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e} \ndouble Exp: \n   k_slow = {s2:.2e} +- {serrds:.2e}, \n   k_fast = {s3:.2e} +-{serrdf:.2e} with reduced Chi = {s4:.2e}".format(
                s0=DF["K_ON_single"].values[0],
                serrs=DF["K_ON_single_1simga"].values[0],
                s1=DF["K_ON_rchi_single"].values[0],
                s2=DF["K_ON_double_slow"].values[0],
                serrds=DF["K_ON_double_slow_1simga"].values[0],
                s3=DF["K_ON_double_fast"].values[0],
                serrdf=DF["K_ON_double_fast_1simga"].values[0],
                s4=DF["K_ON_rchi_double"].values[0],
            ),
            xy=(0.15, 0.6),
            fontsize="small",
            xycoords="axes fraction",
        )
        sns.lineplot(
            x=dwellTime,
            y=strtchtd_exponential(
                dwellTimeRank[dwellTime],
                DF["A_ON_stretchted"].values[0],
                DF["K_ON_stretchted"].values[0],
                DF["ALPHA_ON_stretchted"].values[0],
            ),
            data=dwellTimeRank,
            color="yellow",
            label="stretchted",
        )
        axsns.annotate(
            "stretchted Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e}, A = {s2:.2e}, alpha = {s3:.2f}".format(
                s0=DF["K_ON_stretchted"].values[0],
                serrs=DF["K_ON_stretchted_1sigma"].values[0],
                s1=DF["K_ON_stretchted_rchi"].values[0],
                s2=DF["A_ON_stretchted"].values[0],
                s3=DF["ALPHA_ON_stretchted"].values[0],
            ),
            xy=(0.15, 0.8),
            fontsize="small",
            xycoords="axes fraction",
        )
    elif event == "RISINGEDGE":
        sns.lineplot(
            x=dwellTime,
            y=single_Exp(
                [DF["A_RISINGEDGE"].values[0], DF["K_RISINGEDGE_single"].values[0]],
                dwellTimeRank[dwellTime],
            ),
            data=dwellTimeRank,
            color="red",
            label="single",
        )
        sns.lineplot(
            x=dwellTime,
            y=double_Exp(
                [
                    DF["AdS_RISINGEDGE"].values[0],
                    DF["K_RISINGEDGE_double_slow"].values[0],
                    DF["AdF_RISINGEDGE"].values[0],
                    DF["K_RISINGEDGE_double_fast"].values[0],
                ],
                dwellTimeRank[dwellTime],
            ),
            data=dwellTimeRank,
            color="green",
            label="double",
        )
        axsns.annotate(
            "single Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e} \ndouble Exp: \n   k_slow = {s2:.2e} +- {serrds:.2e}, \n   k_fast = {s3:.2e} +-{serrdf:.2e} with reduced Chi = {s4:.2e}".format(
                s0=DF["K_RISINGEDGE_single"].values[0],
                serrs=DF["K_RISINGEDGE_single_1simga"].values[0],
                s1=DF["K_RISINGEDGE_rchi_single"].values[0],
                s2=DF["K_RISINGEDGE_double_slow"].values[0],
                serrds=DF["K_RISINGEDGE_double_slow_1simga"].values[0],
                s3=DF["K_RISINGEDGE_double_fast"].values[0],
                serrdf=DF["K_RISINGEDGE_double_fast_1simga"].values[0],
                s4=DF["K_RISINGEDGE_rchi_double"].values[0],
            ),
            xy=(0.15, 0.6),
            fontsize="small",
            xycoords="axes fraction",
        )
        sns.lineplot(
            x=dwellTime,
            y=strtchtd_exponential(
                dwellTimeRank[dwellTime],
                DF["A_RISINGEDGE_stretchted"].values[0],
                DF["K_RISINGEDGE_stretchted"].values[0],
                DF["ALPHA_RISINGEDGE_stretchted"].values[0],
            ),
            data=dwellTimeRank,
            color="yellow",
            label="stretchted",
        )
        axsns.annotate(
            "stretchted Exp:\n   k = {s0:.2e} +- {serrs:.2e} reduced Chi  = {s1:.2e}, A = {s2:.2e}, alpha = {s3:.2f}".format(
                s0=DF["K_RISINGEDGE_stretchted"].values[0],
                serrs=DF["K_RISINGEDGE_stretchted_1sigma"].values[0],
                s1=DF["K_RISINGEDGE_stretchted_rchi"].values[0],
                s2=DF["A_RISINGEDGE_stretchted"].values[0],
                s3=DF["ALPHA_RISINGEDGE_stretchted"].values[0],
            ),
            xy=(0.15, 0.8),
            fontsize="small",
            xycoords="axes fraction",
        )

    seaborn_plot.set_xlim(auto=True)
    seaborn_plot.set_ylim(auto=True)

    if event == "EVENT":
        axsns.set(
            xlabel="Event duration [$\mu sec$]", ylabel="k_off cumulative event count"
        )
        tmp = str(writeFile.stem) + "_K_OFF"
        writeFile = writeFile.parent / tmp
    elif event == "NOEVENT":
        axsns.set(
            xlabel="interevent duration [$\mu sec$]",
            ylabel="k_on cumulative interevent count",
        )
        tmp = str(writeFile.stem) + "_K_ON"
        writeFile = writeFile.parent / tmp
    elif event == "RISINGEDGE":
        axsns.set(
            xlabel="RISING EDGE duration [$\mu sec$]",
            ylabel="k_rising cumulative count",
        )
        tmp = str(writeFile.stem) + "_K_RISINGEDGE"
        writeFile = writeFile.parent / tmp

    axsns.set_title("\n".join(wrap(str(writeFile.stem))))
    figsns.savefig(str(writeFile) + ".png", dpi=100)
    figsns.clear()
    plt.close(figsns)
    return


def strtchtd_exponential(x, A, k, alpha):
    return A * np.exp(-((k * x * 1e-6) ** alpha))


def lmFIT_strtchtd_exponential(x, y, k_guess):
    mod = Model(strtchtd_exponential, nan_policy="omit")
    pars = mod.make_params(A=y[-1], k=k_guess, alpha=0.9)
    pars = mod.make_params(k=k_guess, alpha=0.9)
    pars["A"].set(value=y[-1], min=y[-1] * 0.5, max=y[-1] * 10.0)
    pars["k"].set(value=k_guess, min=1e-6, max=1000)
    pars["alpha"].set(value=0.9, min=0.1, max=1)
    out = mod.fit(y, pars, x=x)
    return out


def lmFIT_single(x, y, writeFile):
    exp_mod1 = ExponentialModel(prefix="exp1_", nan_policy="omit")
    pars = exp_mod1.guess(y, x=x)
    pars["exp1_amplitude"].set(value=y[-1], min=0, max=y[-1] * 10.0)
    pars["exp1_decay"].set(value=1 / 20, min=1 / 250.0, max=1000)
    mod = exp_mod1
    out = mod.fit(y, pars, x=x)
    return out


def lmFIT_double(x, y, k_guess):
    exp_mod1 = ExponentialModel(prefix="exp1_", nan_policy="omit")
    pars = exp_mod1.guess(y, x=x)
    exp_mod2 = ExponentialModel(prefix="exp2_", nan_policy="omit")
    pars.update(exp_mod2.make_params())
    pars["exp1_amplitude"].set(value=y[-1] / 2.0, min=0, max=y[-1] * 10.0)
    pars["exp1_decay"].set(value=1 / k_guess / 10.0, min=1 / 250.0, max=40)
    pars["exp2_amplitude"].set(value=y[-1] / 2.0, min=0, max=y[-1] * 10.0)
    pars["exp2_decay"].set(value=1 / k_guess, min=1 / 250.0, max=40)

    mod = exp_mod1 + exp_mod2
    out = mod.fit(y, pars, x=x)
    return out


def lmFIT_quadru(x, y, k_guess, A_guess):
    exp_mod1 = ExponentialModel(prefix="exp1_", nan_policy="omit")
    pars = exp_mod1.guess(y, x=x)
    exp_mod2 = ExponentialModel(prefix="exp2_", nan_policy="omit")
    pars.update(exp_mod2.make_params())
    exp_mod3 = ExponentialModel(prefix="exp3_", nan_policy="omit")
    pars.update(exp_mod3.make_params())
    exp_mod4 = ExponentialModel(prefix="exp4_", nan_policy="omit")
    pars.update(exp_mod4.make_params())
    pars["exp1_amplitude"].set(value=A_guess[0] / 2.0, min=0, max=y[-1] * 10.0)
    pars["exp1_decay"].set(value=1.0 / k_guess[0], min=1 / 250.0, max=2 / k_guess[0])
    pars["exp2_amplitude"].set(value=A_guess[0] / 5.0, min=0, max=y[-1] * 10.0)
    pars["exp2_decay"].set(
        value=1.0 / k_guess[0] / 5.0, min=1 / k_guess[0] / 2, max=3 / k_guess[0]
    )
    pars["exp3_amplitude"].set(value=A_guess[1] / 2.0, min=0, max=y[-1] * 10.0)
    pars["exp3_decay"].set(
        value=1.0 / k_guess[1], min=1 / k_guess[0] / 5, max=2 / k_guess[1]
    )
    pars["exp4_amplitude"].set(value=A_guess[1] / 5.0, min=0, max=y[-1] * 10.0)
    pars["exp4_decay"].set(
        value=2.0 / k_guess[1], min=1 / k_guess[0] / 2, max=5 / k_guess[1]
    )

    mod = exp_mod1 + exp_mod2 + exp_mod3 + exp_mod4
    out = mod.fit(
        y, pars, x=x, method="least_squares", fit_kws={"ftol": 1e-11, "xtol": 1e-11}
    )
    print(out.fit_report())
    # if out.params.stderr in local():
    #     print(out.params.stderr)
    return out


def get_keys():

    kTypes = ["ON", "OFF", "RISINGEDGE"]
    keys = []
    for kType in kTypes:
        keys.extend(
            [
                "K_" + kType + "_double_slow",
                "K_" + kType + "_double_slow_1simga",
                "K_" + kType + "_double_fast",
                "K_" + kType + "_double_fast_1simga",
                "K_" + kType + "_rchi_double",
                "AdS_" + kType,
                "AdF_" + kType,
            ]
        )
        keys.extend(
            [
                "K_" + kType + "_single",
                "K_" + kType + "_single_1simga",
                "K_" + kType + "_rchi_single",
                "A_" + kType,
            ]
        )
        keys.extend(
            [
                "K_" + kType + "_exp1_decay",
                "K_" + kType + "_exp2_decay",
                "K_" + kType + "_exp3_decay",
                "K_" + kType + "_exp4_decay",
            ]
        )
        keys.extend(
            [
                "K_" + kType + "_exp1_amplitude",
                "K_" + kType + "_exp2_amplitude",
                "K_" + kType + "_exp3_amplitude",
                "K_" + kType + "_exp4_amplitude",
            ]
        )
        keys.extend(
            [
                "K_" + kType + "_stretchted",
                "K_" + kType + "_stretchted_1sigma",
                "K_" + kType + "_stretchted_rchi",
                "A_" + kType + "_stretchted",
                "ALPHA_" + kType + "_stretchted",
            ]
        )
    # keys = ['K_OFF_double_slow', 'K_OFF_double_slow_1simga', 'K_OFF_double_fast', 'K_OFF_double_fast_1simga', 'K_OFF_rchi_double', 'AdS_OFF', 'AdF_OFF']

    # keys.extend(['K_OFF_single', 'K_OFF_single_1simga','K_OFF_rchi_single','A_OFF'])
    # keys.extend(['K_ON_double_slow', 'K_ON_double_slow_1simga', 'K_ON_double_fast', 'K_ON_double_fast_1simga', 'K_ON_rchi_double', 'AdS_ON', 'AdF_ON'])
    # keys.extend(['K_ON_single', 'K_ON_single_1simga','K_ON_rchi_single','A_ON'])
    # keys.extend(['K_RISINGEDGE_double_slow', 'K_RISINGEDGE_double_slow_1simga', 'K_RISINGEDGE_double_fast', 'K_RISINGEDGE_double_fast_1simga', 'K_RISINGEDGE_rchi_double', 'AdS_RISINGEDGE', 'AdF_RISINGEDGE'])
    # keys.extend(['K_RISINGEDGE_single', 'K_RISINGEDGE_single_1simga','K_RISINGEDGE_rchi_single','A_RISINGEDGE'])
    # keys.extend(['K_OFF_stretchted','K_OFF_stretchted_1sigma','K_OFF_stretchted_rchi','A_OFF_stretchted','ALPHA_OFF_stretchted'])
    # keys.extend(['K_ON_stretchted','K_ON_stretchted_1sigma','K_ON_stretchted_rchi','A_ON_stretchted','ALPHA_ON_stretchted'])
    # keys.extend(['K_RISINGEDGE_stretchted','K_RISINGEDGE_stretchted_1sigma','K_RISINGEDGE_stretchted_rchi','A_RISINGEDGE_stretchted','ALPHA_RISINGEDGE_stretchted'])
    return keys


def lmfit_strtchtd_to_param(out, event):

    if event == "EVENT":
        if len(out.best_values.keys()) == 3:
            keys = [
                "K_OFF_stretchted",
                "A_OFF_stretchted",
                "ALPHA_OFF_stretchted",
                "K_OFF_stretchted_rchi",
                "K_OFF_stretchted_1sigma",
            ]
            values = [
                out.best_values["k"],
                out.best_values["A"],
                out.best_values["alpha"],
                out.redchi,
            ]
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()["k"][2][1]
                                    - 1 / out.conf_interval()["k"][3][1],
                                    1 / out.conf_interval()["k"][3][1]
                                    - 1 / out.conf_interval()["k"][3][1],
                                ]
                            )
                        )
                    ]
                )
            else:
                values.extend([np.nan])
    elif event == "NOEVENT":
        if len(out.best_values.keys()) == 3:
            keys = [
                "K_ON_stretchted",
                "A_ON_stretchted",
                "ALPHA_ON_stretchted",
                "K_ON_stretchted_rchi",
                "K_ON_stretchted_1sigma",
            ]
            values = [
                out.best_values["k"],
                out.best_values["A"],
                out.best_values["alpha"],
                out.redchi,
            ]
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()["k"][2][1]
                                    - 1 / out.conf_interval()["k"][3][1],
                                    1 / out.conf_interval()["k"][3][1]
                                    - 1 / out.conf_interval()["k"][3][1],
                                ]
                            )
                        )
                    ]
                )
            else:
                values.extend([np.nan])
    elif event == "RISINGEDGE":
        if len(out.best_values.keys()) == 3:
            keys = [
                "K_RISINGEDGE_stretchted",
                "A_RISINGEDGE_stretchted",
                "ALPHA_RISINGEDGE_stretchted",
                "K_RISINGEDGE_stretchted_rchi",
                "K_RISINGEDGE_stretchted_1sigma",
            ]
            values = [
                out.best_values["k"],
                out.best_values["A"],
                out.best_values["alpha"],
                out.redchi,
            ]
            if out.errorbars and False:
                values.extend(
                    [np.nan]
                )  # values.extend([np.abs(np.mean([1/out.conf_interval()['k'][2][1]-1/out.conf_interval()['k'][3][1],1/out.conf_interval()['k'][3][1]-1/out.conf_interval()['k'][3][1]]))])
            else:
                values.extend([np.nan])
    return keys, values


def lmfit_quad_to_param(out, event):

    keys = out.best_values.keys()
    values = []
    tmpKeys = []
    for key in keys:
        if "decay" in key:
            values.append(1.0 / out.best_values[key])
        else:
            values.append(out.best_values[key])

        if event == "EVENT":
            keyString = "K_OFF_"
        elif event == "NOEVENT":
            keyString = "K_ON_"
        elif event == "RISINGEDGE":
            keyString = "K_RISINGEDGE_"

        tmpKeys.append(keyString + key)

    return tmpKeys, values


def lmfit_to_param(out, event):

    if event == "EVENT":
        if len(out.best_values.keys()) > 2:
            exp2 = "exp2"
            exp1 = "exp1"
            if not (1 / out.best_values[exp2 + "_decay"]) < (
                1 / out.best_values[exp1 + "_decay"]
            ):
                exp2 = "exp1"
                exp1 = "exp2"
            keys = [
                "K_OFF_double_slow",
                "K_OFF_double_fast",
                "K_OFF_rchi_double",
                "AdS_OFF",
                "AdF_OFF",
                "K_OFF_double_slow_1simga",
                "K_OFF_double_fast_1simga",
            ]
            values = [
                1 / out.best_values[exp2 + "_decay"],
                1 / out.best_values[exp1 + "_decay"],
                out.redchi,
            ]
            values.extend(
                [
                    out.best_values[exp2 + "_amplitude"],
                    out.best_values[exp1 + "_amplitude"],
                ]
            )
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()[exp2 + "_decay"][2][1]
                                    - 1 / out.conf_interval()[exp2 + "_decay"][3][1],
                                    1 / out.conf_interval()[exp2 + "_decay"][3][1]
                                    - 1 / out.conf_interval()[exp2 + "_decay"][3][1],
                                ]
                            )
                        ),
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()[exp1 + "_decay"][2][1]
                                    - 1 / out.conf_interval()[exp1 + "_decay"][3][1],
                                    1 / out.conf_interval()[exp1 + "_decay"][3][1]
                                    - 1 / out.conf_interval()[exp1 + "_decay"][3][1],
                                ]
                            )
                        ),
                    ]
                )
            else:
                values.extend([np.nan])
        else:
            keys = ["K_OFF_single", "K_OFF_rchi_single", "A_OFF", "K_OFF_single_1simga"]
            values = [
                1 / out.best_values["exp1_decay"],
                out.redchi,
                out.best_values["exp1_amplitude"],
            ]
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()["exp1_decay"][2][1]
                                    - 1 / out.conf_interval()["exp1_decay"][3][1],
                                    1 / out.conf_interval()["exp1_decay"][3][1]
                                    - 1 / out.conf_interval()["exp1_decay"][3][1],
                                ]
                            )
                        )
                    ]
                )
            else:
                values.extend([np.nan])
    elif event == "NOEVENT":
        if len(out.best_values.keys()) > 2:
            exp2 = "exp2"
            exp1 = "exp1"
            if not (1 / out.best_values[exp2 + "_decay"]) < (
                1 / out.best_values[exp1 + "_decay"]
            ):
                exp2 = "exp1"
                exp1 = "exp2"
            keys = [
                "K_ON_double_slow",
                "K_ON_double_fast",
                "K_ON_rchi_double",
                "AdS_ON",
                "AdF_ON",
                "K_ON_double_slow_1simga",
                "K_ON_double_fast_1simga",
            ]
            values = [
                1 / out.best_values[exp2 + "_decay"],
                1 / out.best_values[exp1 + "_decay"],
                out.redchi,
                out.best_values[exp2 + "_amplitude"],
                out.best_values[exp1 + "_amplitude"],
            ]
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()[exp2 + "_decay"][2][1]
                                    - 1 / out.conf_interval()[exp2 + "_decay"][3][1],
                                    1 / out.conf_interval()[exp2 + "_decay"][3][1]
                                    - 1 / out.conf_interval()[exp2 + "_decay"][3][1],
                                ]
                            )
                        ),
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()[exp1 + "_decay"][2][1]
                                    - 1 / out.conf_interval()[exp1 + "_decay"][3][1],
                                    1 / out.conf_interval()[exp1 + "_decay"][3][1]
                                    - 1 / out.conf_interval()[exp1 + "_decay"][3][1],
                                ]
                            )
                        ),
                    ]
                )
            else:
                values.extend([np.nan])
        else:
            keys = ["K_ON_single", "K_ON_rchi_single", "A_ON", "K_ON_single_1simga"]
            values = [
                1 / out.best_values["exp1_decay"],
                out.redchi,
                out.best_values["exp1_amplitude"],
            ]
            if out.errorbars:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()["exp1_decay"][2][1]
                                    - 1 / out.conf_interval()["exp1_decay"][3][1],
                                    1 / out.conf_interval()["exp1_decay"][3][1]
                                    - 1 / out.conf_interval()["exp1_decay"][3][1],
                                ]
                            )
                        )
                    ]
                )
            else:
                values.extend([np.nan])
    elif event == "RISINGEDGE":
        if len(out.best_values.keys()) > 2:
            exp2 = "exp2"
            exp1 = "exp1"
            if not (1 / out.best_values[exp2 + "_decay"]) < (
                1 / out.best_values[exp1 + "_decay"]
            ):
                exp2 = "exp1"
                exp1 = "exp2"
            keys = [
                "K_RISINGEDGE_double_slow",
                "K_RISINGEDGE_double_fast",
                "K_RISINGEDGE_rchi_double",
                "AdS_RISINGEDGE",
                "AdF_RISINGEDGE",
                "K_RISINGEDGE_double_slow_1simga",
                "K_RISINGEDGE_double_fast_1simga",
            ]
            values = [
                1 / out.best_values[exp2 + "_decay"],
                1 / out.best_values[exp1 + "_decay"],
                out.redchi,
                out.best_values[exp2 + "_amplitude"],
                out.best_values[exp1 + "_amplitude"],
            ]
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()[exp2 + "_decay"][2][1]
                                    - 1 / out.conf_interval()[exp2 + "_decay"][3][1],
                                    1 / out.conf_interval()[exp2 + "_decay"][3][1]
                                    - 1 / out.conf_interval()[exp2 + "_decay"][3][1],
                                ]
                            )
                        ),
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()[exp1 + "_decay"][2][1]
                                    - 1 / out.conf_interval()[exp1 + "_decay"][3][1],
                                    1 / out.conf_interval()[exp1 + "_decay"][3][1]
                                    - 1 / out.conf_interval()[exp1 + "_decay"][3][1],
                                ]
                            )
                        ),
                    ]
                )
            else:
                values.extend([np.nan])
        else:
            keys = [
                "K_RISINGEDGE_single",
                "K_RISINGEDGE_rchi_single",
                "A_RISINGEDGE",
                "K_RISINGEDGE_single_1simga",
            ]
            values = [
                1 / out.best_values["exp1_decay"],
                out.redchi,
                out.best_values["exp1_amplitude"],
            ]
            if out.errorbars and False:
                values.extend(
                    [
                        np.abs(
                            np.mean(
                                [
                                    1 / out.conf_interval()["exp1_decay"][2][1]
                                    - 1 / out.conf_interval()["exp1_decay"][3][1],
                                    1 / out.conf_interval()["exp1_decay"][3][1]
                                    - 1 / out.conf_interval()["exp1_decay"][3][1],
                                ]
                            )
                        )
                    ]
                )
            else:
                values.extend([np.nan])

    return keys, values


def add_to_DF(DF, index, key, value):
    # tmpDF = self.statsDF
    if isinstance(key, list):
        tmp = zip(key, value)
        for i, v in enumerate(tmp):
            DF.at[index, v[0]] = v[1]
    else:
        DF.at[index, key] = value
    return DF


def langmuir(x, Qstat, K, n):
    return Qstat * K * x ** n / (1 + K * x ** n)


def langmuirFit(x, y, writeFile):
    mod = Model(langmuir)
    pars = mod.make_params(Qstat=0.3, K=0.08, n=1.5)
    out = mod.fit(y, pars, x=x)

    fig, axes = plt.subplots(figsize=(8, 6))
    axes.plot(x, y, "xb")
    c = np.logspace(
        np.log10(np.min(x)), np.log10(np.max(x)), num=100, base=10, endpoint=True
    )
    axes.plot(c, out.eval(x=c), "r-", label="best fit")
    dely = out.eval_uncertainty(x=c, sigma=1.0)
    axes.fill_between(c, out.eval(x=c) - dely, out.eval(x=c) + dely, color="#888888")
    axes.legend(loc="best")
    axes.set_xscale("log")
    axes.annotate(
        "Langmuir FIT: qstat = {s2:.2e}, n = {s3:.2e} with K = {s4:.2e}".format(
            s2=out.best_values["Qstat"],
            s3=out.best_values["n"],
            s4=out.best_values["K"],
        ),
        xy=(0.15, 0.6),
        fontsize="small",
        xycoords="axes fraction",
    )
    axes.set(xlabel="srtn concentration [nM]", ylabel="P_DOWN")
    fig.tight_layout()
    tmp = str(writeFile.stem) + "_LangmuirFit"
    writeFile = writeFile.parent / tmp
    fig.savefig(str(writeFile) + ".png", dpi=100)
    fig.clear()
    plt.close(fig)

    return out


def gamma_of_inverse(value):
    return scipy.special.gamma(1.0 / value)


def gamma_of_1plusinverse(value):
    return scipy.special.gamma(1 + 1.0 / value)


def weighted_std(values, weights):
    averageWeighted = np.average(values, weights=weights)
    return np.sqrt(
        np.inner(np.power(values - averageWeighted, 2), weights) / np.sum(weights)
    )


def parallelFunction(fileN):
    print(fileN)
    eventsDF = pd.read_csv(fileN)
    newPath = fileN.parent / "KvalueAnalysis" / fileN.stem
    newPath.parent.mkdir(parents=True, exist_ok=True)
    newDF = event_dwell_time_analysis(eventsDF, newPath, event="EVENT")
    newDF = newDF.reset_index().set_index("index")
    tmpDF = event_dwell_time_analysis(eventsDF, newPath, event="NOEVENT")
    tmpDF = tmpDF.reset_index().set_index("index")
    newDF.update(tmpDF, overwrite=False)
    tmpDF = event_dwell_time_analysis(eventsDF, newPath, event="RISINGEDGE")
    tmpDF = tmpDF.reset_index().set_index("index")
    newDF.update(tmpDF, overwrite=False)
    newDF = (
        newDF.reset_index()
        .set_index("index")
        .join(eventsDF.loc[[0]].set_index("index"), rsuffix="_OLD")
    )
    newDF["TOTALTRACETIME"] = eventsDF["STATEDWELLTIME"].sum()
    newDF["NUMBEROFEVENTS"] = len(eventsDF["STATEDWELLTIME"]) // 2
    newDF["P_DOWN"] = (
        eventsDF[eventsDF["STATELABEL"] == 0]["STATEDWELLTIME"].sum()
        / newDF["TOTALTRACETIME"]
    )
    return newDF


#### Main loop
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Define number of THREADS to run the process in parallel. Has to match available resources on slurm or local machine.",
    )
    args = parser.parse_args()

    # set up run constants
    t = args.threads

    resultsPath = Path("/proj/jakobuchheim/share/newChunk")
    savePath = resultsPath
    files_for_frame = sorted(resultsPath.rglob("[!._]*events.csv"))

    if t > 1:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=t, maxtasksperchild=10)
        kvalueDF = pool.imap_unordered(parallelFunction, files_for_frame, chunksize=3)
        pool.close()
        pool.join()
        del pool
        print("pool done")
    else:
        ## import all data frames in folder
        kvalueDF = []
        for fileN in files_for_frame:
            kvalueDF.append(parallelFunction(fileN))

    statsDF = pd.concat(kvalueDF, ignore_index=False).reset_index()
    print(statsDF)

    ## -------------------------------------------
    # SAVE FINAL DATA FRAME
    dataframe_name = (
        "newEventsFit_dataframe_atruntime_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "UNMODIFIED.csv"
    )
    df_name = resultsPath / dataframe_name
    statsDF.to_csv(df_name, encoding="utf-8")

    ## -----------------------------------------
    # parse srtn concentration from comment
    statsDF.loc[statsDF["index"].str.contains("ser", na=False), "type"] = "srtn"
    statsDF.loc[statsDF["index"].str.contains("probe", na=False), "type"] = "apt"
    statsDF.loc[statsDF["index"].str.contains("dop", na=False), "type"] = "dop"
    statsDF.loc[statsDF["index"].str.contains("aptonly_dop", na=False), "type"] = "apt"
    statsDF.loc[statsDF["index"].str.contains("srtn", na=False), "type"] = "srtn"
    statsDF.loc[
        statsDF["index"].str.contains("aptonlyrtp2V_1", na=False), "type"
    ] = "apt"
    statsDF.loc[
        statsDF["index"].str.contains("aptonly_rtp2V", na=False), "type"
    ] = "apt"
    statsDF.loc[statsDF["index"].str.contains("blank", na=False), "type"] = "apt"
    statsDF.loc[statsDF["index"].str.contains("HIAA", na=False), "type"] = "hiaa"
    statsDF.loc[statsDF["index"].str.contains("50n", na=False), "conc"] = 50.0
    statsDF.loc[statsDF["index"].str.contains("500p", na=False), "conc"] = 0.5
    statsDF.loc[statsDF["index"].str.contains("5n", na=False), "conc"] = 5.0
    statsDF.loc[statsDF["index"].str.contains("1n", na=False), "conc"] = 1.0
    statsDF.loc[statsDF["index"].str.contains("100n", na=False), "conc"] = 100.0
    statsDF.loc[statsDF["index"].str.contains("100p", na=False), "conc"] = 0.1
    statsDF.loc[statsDF["index"].str.contains("aptonlyrtp2V_1", na=False), "conc"] = 0.0
    statsDF.loc[statsDF["index"].str.contains("25n", na=False), "conc"] = 25.0
    statsDF.loc[statsDF["index"].str.contains("10n", na=False), "conc"] = 10.0
    statsDF.loc[statsDF["index"].str.contains("5u", na=False), "conc"] = 500.0
    # statsDF.loc[statsDF["COMMENTS"].str.contains("dop", na=False),'conc'] = 0.0
    statsDF.loc[statsDF["index"].str.contains("blank", na=False), "conc"] = 0.0

    ## -----------------------------------------
    # parse in name to device number
    statsDF["SAMPLE"] = (
        statsDF["index"]
        .astype("str")
        .str.split("Chip_")
        .str.get(1)
        .str.split("_")
        .str[0:2]
        .str.join("_")
    )
    statsDF.loc[statsDF["SAMPLE"] == "AV_R4C1", "SAMPLE"] = "Device_1"
    statsDF.loc[statsDF["SAMPLE"] == "AV_R4C4", "SAMPLE"] = "Device_2"
    statsDF.loc[statsDF["SAMPLE"] == "old3_tl", "SAMPLE"] = "Device_3"
    statsDF.loc[statsDF["SAMPLE"] == "old3_bl", "SAMPLE"] = "Device_5"
    statsDF.loc[statsDF["SAMPLE"] == "n2_tr", "SAMPLE"] = "Device_4"

    ## -----------------------------------------
    # add some columns for better identification
    statsDF["sliceTime"] = (statsDF["TOTALTRACETIME"] * 1e-6).round(-1)
    statsDF["sliceTime"] = statsDF["sliceTime"].astype(int)
    statsDF["ident"] = (
        statsDF["type"]
        + "-"
        + statsDF["conc"].astype("str")
        + "nM_"
        + statsDF["SAMPLE"].astype("str")
        + "_"
        + statsDF["sliceTime"].astype("str")
    )

    ## -----------------------------------------
    # make copy of dataframe before changing stuff
    statsDF_copy = statsDF.copy(deep=True)
    # statsDF = statsDF_copy.copy(deep = True)

    # ## -----------------------------------------
    # # set Kfast to k single if fitting not good. (too low or too high Afast)
    # amplitudeThreshold = 0.05  # was 0.05
    # statsDF['kmodified'] = False
    # # K_ON
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_ON_double_fast']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_ON_single']
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_ON_double_slow']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_ON_single']
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['AdS_ON']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['A_ON'] / 2.0
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_ON_rchi_double']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_ON_rchi_single']
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_ON_double_fast_1simga']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_ON_single_1simga']
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_ON_double_slow_1simga']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_ON_single_1simga']
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['kmodified']] = True
    # statsDF.loc[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['AdF_ON']] = statsDF[statsDF['AdF_ON']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['A_ON'] / 2.0

    # # K_OFF
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_OFF_double_fast']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_OFF_single']
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_OFF_double_slow']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_OFF_single']
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['AdS_OFF']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['A_OFF'] / 2.0
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_OFF_rchi_double']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_OFF_rchi_single']
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_OFF_double_fast_1simga']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_OFF_single_1simga']
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['K_OFF_double_slow_1simga']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['K_OFF_single_1simga']
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['kmodified']] = True
    # statsDF.loc[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS'],['AdF_OFF']] = statsDF[statsDF['AdF_OFF']<amplitudeThreshold*statsDF['NUMBEROFEVENTS']]['A_OFF'] / 2.0

    ## -------------------------------------------
    # remove unreliabe K single
    statsDF["kmodified"] = False
    minAmplitudeThreshold = 0.01
    maxAmplitudeThreshold = 4

    # A_ON too high
    statsDF.loc[
        statsDF["A_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["K_ON_single", "K_ON_single_1simga", "K_ON_rchi_single"],
    ] = np.nan
    statsDF.loc[
        statsDF["A_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["A_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"], ["A_ON"]
    ] = np.nan

    # A_RISINGEDGE too high
    statsDF.loc[
        statsDF["A_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        [
            "K_RISINGEDGE_single",
            "K_RISINGEDGE_single_1simga",
            "K_RISINGEDGE_rchi_single",
        ],
    ] = np.nan
    statsDF.loc[
        statsDF["A_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["A_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["A_RISINGEDGE"],
    ] = np.nan

    # A_OFF too high
    statsDF.loc[
        statsDF["A_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["K_OFF_single", "K_OFF_single_1simga", "K_OFF_rchi_single"],
    ] = np.nan
    statsDF.loc[
        statsDF["A_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["A_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"], ["A_OFF"]
    ] = np.nan

    ## -------------------------------------------
    # check for A fast being to large
    # statsDF[statsDF['AdF_ON']>1.5*statsDF['NUMBEROFEVENTS']]['K_ON_double_fast','AdF_ON','NUMBEROFEVENTS']
    # statsDF[statsDF['A_ON']>5*statsDF['NUMBEROFEVENTS']][['K_ON_double_fast','AdF_ON','AdS_ON','NUMBEROFEVENTS','ident','A_ON','K_ON_single']]
    # statsDF[statsDF['K_ON_rchi_double']>statsDF['K_ON_rchi_single']][['K_ON_double_fast','AdF_ON','AdS_ON','NUMBEROFEVENTS','ident','A_ON','K_ON_single']]

    # K_ON
    overwriteON = [
        "K_ON_double_fast",
        "K_ON_double_slow",
        "K_ON_rchi_double",
        "K_ON_double_fast_1simga",
        "K_ON_double_slow_1simga",
    ]
    useON = [
        "K_ON_single",
        "K_ON_single",
        "K_ON_rchi_single",
        "K_ON_single_1simga",
        "K_ON_single_1simga",
    ]

    # AdF too low
    statsDF.loc[
        statsDF["AdF_ON"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        overwriteON,
    ] = statsDF[statsDF["AdF_ON"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
        useON
    ]
    statsDF.loc[
        statsDF["AdF_ON"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["AdF_ON"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["AdS_ON", "AdF_ON"],
    ] = (
        statsDF[statsDF["AdF_ON"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
            ["A_ON", "A_ON"]
        ]
        / 2.0
    )

    # AdF too high
    statsDF.loc[
        statsDF["AdF_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        overwriteON,
    ] = statsDF[statsDF["AdF_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
        useON
    ]
    statsDF.loc[
        statsDF["AdF_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["AdF_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["AdS_ON", "AdF_ON"],
    ] = (
        statsDF[statsDF["AdF_ON"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
            ["A_ON", "A_ON"]
        ]
        / 2.0
    )

    # reducedChiSqure better for single data fit
    statsDF.loc[
        statsDF["K_ON_rchi_single"] < statsDF["K_ON_rchi_double"], overwriteON
    ] = statsDF[statsDF["K_ON_rchi_single"] < statsDF["K_ON_rchi_double"]][useON]
    statsDF.loc[
        statsDF["K_ON_rchi_single"] < statsDF["K_ON_rchi_double"], ["kmodified"]
    ] = True
    statsDF.loc[
        statsDF["K_ON_rchi_single"] < statsDF["K_ON_rchi_double"], ["AdS_ON", "AdF_ON"]
    ] = (
        statsDF[statsDF["K_ON_rchi_single"] < statsDF["K_ON_rchi_double"]][
            ["A_ON", "A_ON"]
        ]
        / 2.0
    )

    # K_RISINGEDGE
    overwriteRISINGEDGE = [
        "K_RISINGEDGE_double_fast",
        "K_RISINGEDGE_double_slow",
        "K_RISINGEDGE_rchi_double",
        "K_RISINGEDGE_double_fast_1simga",
        "K_RISINGEDGE_double_slow_1simga",
    ]
    useRISINGEDGE = [
        "K_RISINGEDGE_single",
        "K_RISINGEDGE_single",
        "K_RISINGEDGE_rchi_single",
        "K_RISINGEDGE_single_1simga",
        "K_RISINGEDGE_single_1simga",
    ]

    # AdF too low
    statsDF.loc[
        statsDF["AdF_RISINGEDGE"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        overwriteRISINGEDGE,
    ] = statsDF[
        statsDF["AdF_RISINGEDGE"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]
    ][
        useRISINGEDGE
    ]
    statsDF.loc[
        statsDF["AdF_RISINGEDGE"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["AdF_RISINGEDGE"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["AdS_RISINGEDGE", "AdF_RISINGEDGE"],
    ] = (
        statsDF[
            statsDF["AdF_RISINGEDGE"]
            < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]
        ][["A_RISINGEDGE", "A_RISINGEDGE"]]
        / 2.0
    )

    # AdF too high
    statsDF.loc[
        statsDF["AdF_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        overwriteRISINGEDGE,
    ] = statsDF[
        statsDF["AdF_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]
    ][
        useRISINGEDGE
    ]
    statsDF.loc[
        statsDF["AdF_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["AdF_RISINGEDGE"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["AdS_RISINGEDGE", "AdF_RISINGEDGE"],
    ] = (
        statsDF[
            statsDF["AdF_RISINGEDGE"]
            > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]
        ][["A_RISINGEDGE", "A_RISINGEDGE"]]
        / 2.0
    )

    # reducedChiSqure better for single data fit
    statsDF.loc[
        statsDF["K_RISINGEDGE_rchi_single"] < statsDF["K_RISINGEDGE_rchi_double"],
        overwriteRISINGEDGE,
    ] = statsDF[
        statsDF["K_RISINGEDGE_rchi_single"] < statsDF["K_RISINGEDGE_rchi_double"]
    ][
        useRISINGEDGE
    ]
    statsDF.loc[
        statsDF["K_RISINGEDGE_rchi_single"] < statsDF["K_RISINGEDGE_rchi_double"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["K_RISINGEDGE_rchi_single"] < statsDF["K_RISINGEDGE_rchi_double"],
        ["AdS_RISINGEDGE", "AdF_RISINGEDGE"],
    ] = (
        statsDF[
            statsDF["K_RISINGEDGE_rchi_single"] < statsDF["K_RISINGEDGE_rchi_double"]
        ][["A_RISINGEDGE", "A_RISINGEDGE"]]
        / 2.0
    )

    # K_OFF
    overwriteOFF = [
        "K_OFF_double_fast",
        "K_OFF_double_slow",
        "K_OFF_rchi_double",
        "K_OFF_double_fast_1simga",
        "K_OFF_double_slow_1simga",
    ]
    useOFF = [
        "K_OFF_single",
        "K_OFF_single",
        "K_OFF_rchi_single",
        "K_OFF_single_1simga",
        "K_OFF_single_1simga",
    ]

    # AdF too low
    statsDF.loc[
        statsDF["AdF_OFF"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        overwriteOFF,
    ] = statsDF[statsDF["AdF_OFF"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
        useOFF
    ]
    statsDF.loc[
        statsDF["AdF_OFF"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["AdF_OFF"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["AdS_OFF", "AdF_OFF"],
    ] = (
        statsDF[statsDF["AdF_OFF"] < minAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
            ["A_OFF", "A_OFF"]
        ]
        / 2.0
    )

    # AdF too high
    statsDF.loc[
        statsDF["AdF_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        overwriteOFF,
    ] = statsDF[statsDF["AdF_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
        useOFF
    ]
    statsDF.loc[
        statsDF["AdF_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["kmodified"],
    ] = True
    statsDF.loc[
        statsDF["AdF_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"],
        ["AdS_OFF", "AdF_OFF"],
    ] = (
        statsDF[statsDF["AdF_OFF"] > maxAmplitudeThreshold * statsDF["NUMBEROFEVENTS"]][
            ["A_OFF", "A_OFF"]
        ]
        / 2.0
    )

    # reducedChiSqure better for single data fit
    statsDF.loc[
        statsDF["K_OFF_rchi_single"] < statsDF["K_OFF_rchi_double"], overwriteOFF
    ] = statsDF[statsDF["K_OFF_rchi_single"] < statsDF["K_OFF_rchi_double"]][useOFF]
    statsDF.loc[
        statsDF["K_OFF_rchi_single"] < statsDF["K_OFF_rchi_double"], ["kmodified"]
    ] = True
    statsDF.loc[
        statsDF["K_OFF_rchi_single"] < statsDF["K_OFF_rchi_double"],
        ["AdS_OFF", "AdF_OFF"],
    ] = (
        statsDF[statsDF["K_OFF_rchi_single"] < statsDF["K_OFF_rchi_double"]][
            ["A_OFF", "A_OFF"]
        ]
        / 2.0
    )

    ## -----------------------------------------
    # normalize event count for comparison
    statsDF["EVENTSPERSEC"] = statsDF["NUMBEROFEVENTS"] / statsDF["sliceTime"]

    ## -----------------------------------------
    # calculate some stats
    numberOfEvents = statsDF["NUMBEROFEVENTS"].groupby(statsDF["ident"]).sum()
    print(numberOfEvents)

    ## -----------------------------------------
    # calculate k_average for stretchted exponential
    statsDF["K_ON_stretchted_MEAN"] = statsDF["K_ON_stretchted"] / statsDF[
        "ALPHA_ON_stretchted"
    ].apply(gamma_of_1plusinverse)
    statsDF["K_OFF_stretchted_MEAN"] = statsDF["K_OFF_stretchted"] / statsDF[
        "ALPHA_OFF_stretchted"
    ].apply(gamma_of_1plusinverse)
    statsDF["K_RISINGEDGE_stretchted_MEAN"] = statsDF[
        "K_RISINGEDGE_stretchted"
    ] / statsDF["ALPHA_RISINGEDGE_stretchted"].apply(gamma_of_1plusinverse)
    statsDF["K_RATIO"] = statsDF["K_ON_single"] / statsDF["K_OFF_single"]

    # sort DF for better plotting:
    statsDF.sort_values(by=["type", "conc", "SAMPLE", "sliceTime"], inplace=True)

    ## -----------------------------------------
    # calculate effective K considering amplitude:
    statsDF["EFFECTIVE_K_ON"] = (
        statsDF["K_ON_double_fast"] * statsDF["AdF_ON"]
        + statsDF["K_ON_double_slow"] * statsDF["AdS_ON"]
    ) / (statsDF["AdS_ON"] + statsDF["AdF_ON"])
    statsDF["EFFECTIVE_K_ON_ERR"] = np.nan
    statsDF["EFFECTIVE_K_RISINGEDGE"] = (
        statsDF["K_RISINGEDGE_double_fast"] * statsDF["AdF_RISINGEDGE"]
        + statsDF["K_RISINGEDGE_double_slow"] * statsDF["AdS_RISINGEDGE"]
    ) / (statsDF["AdS_RISINGEDGE"] + statsDF["AdF_RISINGEDGE"])
    statsDF["EFFECTIVE_K_RISINGEDGE_ERR"] = np.nan
    statsDF["EFFECTIVE_K_OFF"] = (
        statsDF["K_OFF_double_fast"] * statsDF["AdF_OFF"]
        + statsDF["K_OFF_double_slow"] * statsDF["AdS_OFF"]
    ) / (statsDF["AdS_OFF"] + statsDF["AdF_OFF"])
    statsDF["EFFECTIVE_K_OFF_ERR"] = np.nan

    dataframe_name = (
        "newEventsFit_dataframe_atruntime_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "cleanup.csv"
    )
    df_name = resultsPath / dataframe_name
    statsDF.to_csv(df_name, encoding="utf-8")
    ## ----------------------------------
    # boxplots for all devices to get overview
    # boxplotlist = ['K_ON_double_fast','K_ON_double_slow','K_ON_single','K_RISINGEDGE_double_fast','K_RISINGEDGE_double_slow','K_RISINGEDGE_single','K_OFF_double_fast','K_OFF_double_slow','K_OFF_single','EVENTSPERSEC','P_DOWN', 'EFFECTIVE_K_ON', 'EFFECTIVE_K_OFF', 'EFFECTIVE_K_RISINGEDGE','K_OFF_stretchted','K_ON_stretchted','ALPHA_OFF_stretchted','ALPHA_ON_stretchted','K_ON_stretchted_MEAN','K_OFF_stretchted_MEAN','K_RATIO','K_RISINGEDGE_stretchted_MEAN']
    boxplotlist = [
        "K_RATIO",
        "K_ON_double_fast",
        "K_ON_double_slow",
        "K_ON_single",
        "K_OFF_double_fast",
        "K_OFF_double_slow",
        "K_OFF_single",
        "EVENTSPERSEC",
        "P_DOWN",
        "K_OFF_stretchted",
        "K_ON_stretchted",
        "ALPHA_OFF_stretchted",
        "ALPHA_ON_stretchted",
        "K_ON_stretchted_MEAN",
        "K_OFF_stretchted_MEAN",
        "K_RISINGEDGE_stretchted_MEAN",
    ]
    axlimitlist = [
        3,
        250,
        50,
        150,
        250,
        50,
        150,
        5,
        1,
        100,
        100,
        1,
        1,
        100,
        100,
        5,
        10,
        1,
        1,
        40,
        80,
    ]
    ylimits = dict(zip(boxplotlist, axlimitlist))
    for i in boxplotlist:

        for group in statsDF.groupby(["SAMPLE"]):
            tmpDF = group[1]
            sns.set()
            sns.set_style("whitegrid")
            figsns, axsns = plt.subplots(figsize=(12, 8))

            ax = sns.boxplot(
                x="ident", y=i, data=tmpDF, hue="conc", ax=axsns, dodge=False
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

            ax.set_ylim([0, ylimits[i]])

            figsns.tight_layout()
            fileName = tmpDF["SAMPLE"].values[0] + "_" + i + "_boxplot.png"
            figsns.savefig(savePath / fileName, dpi=200)
            figsns.clear()
            plt.close(figsns)

    boxplotlist.extend(
        [
            "BASELINESTD",
            "conc",
        ]
    )  # ,'type'])

    ## ----------------------------------
    # calculate max likely k values
    stdlist = [
        "K_ON_double_fast",
        "K_ON_double_slow",
        "K_ON_single",
        "K_RISINGEDGE_double_fast",
        "K_RISINGEDGE_double_slow",
        "K_RISINGEDGE_single",
        "K_OFF_double_fast",
        "K_OFF_double_slow",
        "K_OFF_single",
        "EVENTSPERSEC",
        "P_DOWN",
        "K_ON_double_fast_1simga",
        "K_RISINGEDGE_double_fast_1simga",
        "K_OFF_double_fast_1simga",
    ]
    stdlist.extend(
        [
            "NUMBEROFEVENTS",
            "SAMPLE",
            "type",
            "sliceTime",
            "index",
            "conc",
            "ident",
            "AdF_ON",
            "AdF_RISINGEDGE",
            "AdF_OFF",
        ]
    )
    stdlist.extend(
        [
            "K_OFF_stretchted_1sigma",
            "K_ON_stretchted_1sigma",
            "K_OFF_stretchted",
            "K_ON_stretchted",
            "K_OFF_stretchted_MEAN",
            "K_ON_stretchted_MEAN",
        ]
    )
    subDF = statsDF[stdlist]
    # subDF = subDF.dropna()
    subDF["K_ON_double_fast_1simga"].fillna(60, inplace=True)
    subDF["K_RISINGEDGE_double_fast_1simga"].fillna(60, inplace=True)
    subDF["K_OFF_double_fast_1simga"].fillna(60, inplace=True)
    subDF["K_ON_double_fast"].fillna(0.00001, inplace=True)
    subDF["K_RISINGEDGE_double_fast"].fillna(0.00001, inplace=True)
    subDF["K_OFF_double_fast"].fillna(0.00001, inplace=True)
    subDF["AdF_ON"].fillna(1e-30, inplace=True)
    subDF["AdF_RISINGEDGE"].fillna(1e-30, inplace=True)
    subDF["AdF_OFF"].fillna(1e-30, inplace=True)
    subDF["K_OFF_stretchted"].fillna(0.00001, inplace=True)
    subDF["K_ON_stretchted"].fillna(0.00001, inplace=True)
    subDF["K_OFF_stretchted_MEAN"].fillna(0.00001, inplace=True)
    subDF["K_ON_stretchted_MEAN"].fillna(0.00001, inplace=True)
    subDF["K_OFF_stretchted_1sigma"].fillna(60, inplace=True)
    subDF["K_ON_stretchted_1sigma"].fillna(60, inplace=True)
    statsDF["MLH_K_ON_FAST"] = np.nan
    statsDF["MLH_K_ON_FAST_ERR"] = np.nan
    statsDF["MLH_K_RISINGEDGE_FAST"] = np.nan
    statsDF["MLH_K_RISINGEDGE_FAST_ERR"] = np.nan
    statsDF["MLH_K_OFF_FAST"] = np.nan
    statsDF["MLH_K_OFF_FAST_ERR"] = np.nan
    statsDF["MLH_K_OFF_stretchted"] = np.nan
    statsDF["MLH_K_OFF_stretchted_ERR"] = np.nan
    statsDF["MLH_K_ON_stretchted"] = np.nan
    statsDF["MLH_K_ON_stretchted_ERR"] = np.nan
    statsDF["MLH_K_OFF_stretchted_MEAN"] = np.nan
    statsDF["MLH_K_OFF_stretchted_MEAN_ERR"] = np.nan
    statsDF["MLH_K_ON_stretchted_MEAN"] = np.nan
    statsDF["MLH_K_ON_stretchted_MEAN_ERR"] = np.nan

    # for group in subDF.groupby(['SAMPLE','type','conc','sliceTime']):
    for group in subDF.groupby(["ident"]):

        weights = 1 / (
            group[1]["K_ON_double_fast_1simga"] / group[1]["K_ON_double_fast"]
        )
        averageWeighted = np.average(group[1]["K_ON_double_fast"], weights=weights)
        averageError = weighted_std(group[1]["K_ON_double_fast"], weights=weights)
        statsDF.loc[statsDF["ident"] == group[0], ["MLH_K_ON_FAST"]] = averageWeighted
        statsDF.loc[statsDF["ident"] == group[0], ["MLH_K_ON_FAST_ERR"]] = averageError
        statsDF.loc[
            statsDF["K_ON_double_fast_1simga"] > statsDF["MLH_K_ON_FAST_ERR"],
            ["MLH_K_ON_FAST_ERR"],
        ] = statsDF[statsDF["K_ON_double_fast_1simga"] > statsDF["MLH_K_ON_FAST_ERR"]][
            "K_ON_double_fast_1simga"
        ]

        weights = 1 / (
            group[1]["K_RISINGEDGE_double_fast_1simga"]
            / group[1]["K_RISINGEDGE_double_fast"]
        )
        averageWeighted = np.average(
            group[1]["K_RISINGEDGE_double_fast"], weights=weights
        )
        averageError = weighted_std(
            group[1]["K_RISINGEDGE_double_fast"], weights=weights
        )
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_RISINGEDGE_FAST"]
        ] = averageWeighted
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_RISINGEDGE_FAST_ERR"]
        ] = averageError
        statsDF.loc[
            statsDF["K_RISINGEDGE_double_fast_1simga"]
            > statsDF["MLH_K_RISINGEDGE_FAST_ERR"],
            ["MLH_K_RISINGEDGE_FAST_ERR"],
        ] = statsDF[
            statsDF["K_RISINGEDGE_double_fast_1simga"]
            > statsDF["MLH_K_RISINGEDGE_FAST_ERR"]
        ][
            "K_RISINGEDGE_double_fast_1simga"
        ]

        weights = group[1]["AdF_OFF"].apply(np.sqrt) / (
            group[1]["K_OFF_double_fast_1simga"]
            * group[1]["K_OFF_double_fast_1simga"]
            * group[1]["K_OFF_double_fast_1simga"]
            / group[1]["K_OFF_double_fast"]
            / group[1]["K_OFF_double_fast"]
            / group[1]["K_OFF_double_fast"]
        )
        averageWeighted = np.average(group[1]["K_OFF_double_fast"], weights=weights)
        averageError = weighted_std(group[1]["K_OFF_double_fast"], weights=weights)
        statsDF.loc[statsDF["ident"] == group[0], ["MLH_K_OFF_FAST"]] = averageWeighted
        statsDF.loc[statsDF["ident"] == group[0], ["MLH_K_OFF_FAST_ERR"]] = averageError
        statsDF.loc[
            statsDF["K_OFF_double_fast_1simga"] > statsDF["MLH_K_OFF_FAST_ERR"],
            ["MLH_K_OFF_FAST_ERR"],
        ] = statsDF[
            statsDF["K_OFF_double_fast_1simga"] > statsDF["MLH_K_OFF_FAST_ERR"]
        ][
            "K_OFF_double_fast_1simga"
        ]

        weights = 1 / (
            group[1]["K_ON_stretchted_1sigma"]
            * group[1]["K_ON_stretchted_1sigma"]
            / group[1]["K_ON_stretchted"]
            / group[1]["K_ON_stretchted"]
        )
        averageWeighted = np.average(group[1]["K_ON_stretchted"], weights=weights)
        averageError = weighted_std(group[1]["K_ON_stretchted"], weights=weights)
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_ON_stretchted"]
        ] = averageWeighted
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_ON_stretchted_ERR"]
        ] = averageError
        statsDF.loc[
            statsDF["K_ON_stretchted_1sigma"] > statsDF["MLH_K_ON_stretchted_ERR"],
            ["MLH_K_ON_stretchted_ERR"],
        ] = statsDF[
            statsDF["K_ON_stretchted_1sigma"] > statsDF["MLH_K_ON_stretchted_ERR"]
        ][
            "K_ON_stretchted_1sigma"
        ]

        weights = 1 / (
            group[1]["K_OFF_stretchted_1sigma"]
            * group[1]["K_OFF_stretchted_1sigma"]
            / group[1]["K_OFF_stretchted"]
            / group[1]["K_OFF_stretchted"]
        )
        averageWeighted = np.average(group[1]["K_OFF_stretchted"], weights=weights)
        averageError = weighted_std(group[1]["K_OFF_stretchted"], weights=weights)
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_OFF_stretchted"]
        ] = averageWeighted
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_OFF_stretchted_ERR"]
        ] = averageError
        statsDF.loc[
            statsDF["K_OFF_stretchted_1sigma"] > statsDF["MLH_K_OFF_stretchted_ERR"],
            ["MLH_K_OFF_stretchted_ERR"],
        ] = statsDF[
            statsDF["K_OFF_stretchted_1sigma"] > statsDF["MLH_K_OFF_stretchted_ERR"]
        ][
            "K_OFF_stretchted_1sigma"
        ]

        weights = 1 / (
            group[1]["K_ON_stretchted_1sigma"]
            * group[1]["K_ON_stretchted_1sigma"]
            / group[1]["K_ON_stretchted"]
            / group[1]["K_ON_stretchted"]
        )
        averageWeighted = np.average(group[1]["K_ON_stretchted_MEAN"], weights=weights)
        averageError = weighted_std(group[1]["K_ON_stretchted_MEAN"], weights=weights)
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_ON_stretchted_MEAN"]
        ] = averageWeighted
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_ON_stretchted_MEAN_ERR"]
        ] = averageError
        statsDF.loc[
            statsDF["K_ON_stretchted_1sigma"] > statsDF["MLH_K_ON_stretchted_MEAN_ERR"],
            ["MLH_K_ON_stretchted_MEAN_ERR"],
        ] = statsDF[
            statsDF["K_ON_stretchted_1sigma"] > statsDF["MLH_K_ON_stretchted_MEAN_ERR"]
        ][
            "K_ON_stretchted_1sigma"
        ]

        weights = 1 / (
            group[1]["K_OFF_stretchted_1sigma"]
            * group[1]["K_OFF_stretchted_1sigma"]
            / group[1]["K_OFF_stretchted"]
            / group[1]["K_OFF_stretchted"]
        )
        averageWeighted = np.average(group[1]["K_OFF_stretchted_MEAN"], weights=weights)
        averageError = weighted_std(group[1]["K_OFF_stretchted_MEAN"], weights=weights)
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_OFF_stretchted_MEAN"]
        ] = averageWeighted
        statsDF.loc[
            statsDF["ident"] == group[0], ["MLH_K_OFF_stretchted_MEAN_ERR"]
        ] = averageError
        statsDF.loc[
            statsDF["K_OFF_stretchted_1sigma"]
            > statsDF["MLH_K_OFF_stretchted_MEAN_ERR"],
            ["MLH_K_OFF_stretchted_MEAN_ERR"],
        ] = statsDF[
            statsDF["K_OFF_stretchted_1sigma"]
            > statsDF["MLH_K_OFF_stretchted_MEAN_ERR"]
        ][
            "K_OFF_stretchted_1sigma"
        ]

    KONList = ["MLH_K_ON_FAST", "MLH_K_ON_stretchted", "MLH_K_ON_stretchted_MEAN"]
    KOFFList = ["MLH_K_OFF_FAST", "MLH_K_OFF_stretchted", "MLH_K_OFF_stretchted_MEAN"]

    ## ----------------------------------
    # Plot MLH_K values vs concentration
    for group in statsDF[statsDF["type"].isin(["srtn", "apt"])].groupby(["SAMPLE"]):

        for i in range(0, len(KONList)):
            sns.set()
            sns.set_style("whitegrid")
            figsns, axsns = plt.subplots(figsize=(8, 8))
            print(group[1].columns)
            seab = sns.lineplot(
                x="conc",
                y=KONList[i],
                data=group[1],
                hue="sliceTime",
                estimator=None,
                ci=None,
                palette=sns.light_palette(
                    "green", n_colors=len(group[1]["sliceTime"].unique())
                ),
            )
            axsns.errorbar(
                group[1]["conc"],
                group[1][KONList[i]],
                yerr=group[1][KONList[i] + "_ERR"],
                elinewidth=1,  # width of error bar line
                ecolor="k",  # color of error bar
                capsize=10,  # cap length for error bar
                capthick=1,  # cap thickness for error bar
                fmt="none",
                zorder=50,
                alpha=0.25,
            )

            sns.lineplot(
                x="conc",
                y=KOFFList[i],
                data=group[1],
                hue="sliceTime",
                estimator=None,
                ci=None,
                palette=sns.light_palette(
                    "navy", n_colors=len(group[1]["sliceTime"].unique())
                ),
            )
            axsns.errorbar(
                group[1]["conc"],
                group[1][KOFFList[i]],
                yerr=group[1][KOFFList[i] + "_ERR"],
                elinewidth=1,  # width of error bar line
                ecolor="k",  # color of error bar
                capsize=10,  # cap length for error bar
                capthick=1,  # cap thickness for error bar
                fmt="none",
                zorder=50,
                alpha=0.25,
            )

            custom_lines = [
                Line2D([0], [0], color="green", lw=2),
                Line2D([0], [0], color="navy", lw=2),
            ]
            axsns.legend(custom_lines, ["K_ON", "K_OFF"])
            axsns.set_ylim([0, 250])
            seab.set_xscale("log")
            axsns.set(xlabel="srtn concentration [nM]", ylabel="K value [1/s]")
            figsns.tight_layout()
            fileName = (
                str(group[0]) + "_" + KONList[i] + "_and_K_OFF_vs_concentration.png"
            )
            figsns.savefig(savePath / fileName, dpi=200)
            figsns.clear()
            plt.close(figsns)

        ## ----------------------------------
        # pairplot
        sns.set()
        sns.set_style("whitegrid")
        seab = sns.pairplot(
            group[1],
            hue="conc",
            markers="+",
            diag_kind="hist",
            vars=[
                "MLH_K_OFF_FAST",
                "K_OFF_double_fast",
                "MLH_K_ON_FAST",
                "K_ON_double_fast",
                "MLH_K_RISINGEDGE_FAST",
                "K_RISINGEDGE_double_fast",
                "P_DOWN",
            ],
            height=3,
        )
        fileName = str(group[0]) + "_pairplot.png"
        seab.savefig(savePath / fileName, dpi=200)
        plt.close()

    ## -------------------------------------------
    # SAVE FINAL DATA FRAME
    dataframe_name = (
        "newEventsFit_dataframe_atruntime_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".csv"
    )
    df_name = resultsPath / dataframe_name
    statsDF.to_csv(df_name, encoding="utf-8")

    ## -------------------------------------------
    # READ FINAL DATA FRAME _ UNCOMMENT
    # statsDF = pd.read_csv(resultsPath / 'newEventsFit_dataframe_atruntime_20200521-145209.csv') CURRENT REFERENCE 20200522
    # statsDF = pd.read_csv(resultsPath / 'newEventsFit_dataframe_atruntime_20200612-222203.csv')
    # statsDF = pd.read_csv(resultsPath / 'newEventsFit_dataframe_atruntime_20200901-111221UNMODIFIED.csv')

    ## -------------------------------------------
    # Langmuir fitting

    for group in subDF.groupby(["SAMPLE", "type", "sliceTime"]):
        fileName = str(group[0][0]) + "_sliceTime_" + str(group[0][2]) + "s__P_DOWN.png"
        tmpDF = group[1].dropna(subset=["P_DOWN", "conc"])
        if len(tmpDF["conc"].unique()) > 2:
            out = langmuirFit(tmpDF["conc"], tmpDF["P_DOWN"], savePath / fileName)

    ## -------------------------------------------
    # Device 2 - selectivity
    boxplotlist = [
        "K_ON_single",
        "K_OFF_single",
        "EVENTSPERSEC",
        "P_DOWN",
        "K_ON_stretchted",
        "K_ON_stretchted_MEAN",
        "K_OFF_stretchted",
        "K_OFF_stretchted_MEAN",
    ]
    axlimitlist = [100, 15, 15, 150, 40, 50, 2, 0.3, 20, 20, 20, 20]
    ylimits = dict(zip(boxplotlist, axlimitlist))
    for i in boxplotlist:
        tmpDF = statsDF[statsDF["SAMPLE"] == "Device_2"]
        sns.set()
        sns.set_style("whitegrid")
        figsns, axsns = plt.subplots(figsize=(12, 8))

        ax = sns.boxplot(
            x="ident", y=i, data=tmpDF, hue="sliceTime", ax=axsns, dodge=False
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_ylim([0, ylimits[i]])

        figsns.tight_layout()
        fileName = tmpDF["SAMPLE"].values[0] + "_" + i + "_boxplot.png"
        figsns.savefig(savePath / fileName, dpi=200)
        figsns.clear()
        plt.close(figsns)

    ## -------------------------------------------
    # Device 4 - wash cycle
    statsDF.loc[statsDF["index"].str.contains("wash1", na=False), "washcycle"] = "wash1"
    statsDF.loc[statsDF["index"].str.contains("wash2", na=False), "washcycle"] = "wash2"
    statsDF.loc[statsDF["index"].str.contains("wash3", na=False), "washcycle"] = "wash3"
    statsDF.loc[
        statsDF["index"].str.contains("initial", na=False), "washcycle"
    ] = "incubation"
    boxplotlist = [
        "K_ON_single",
        "K_RISINGEDGE_single",
        "K_OFF_single",
        "EVENTSPERSEC",
        "P_DOWN",
        "MLH_K_ON_stretchted",
        "MLH_K_ON_stretchted_MEAN",
        "MLH_K_OFF_stretchted",
        "MLH_K_OFF_stretchted_MEAN",
        "K_ON_stretchted",
        "K_ON_stretchted_MEAN",
        "K_OFF_stretchted",
        "K_OFF_stretchted_MEAN",
    ]
    axlimitlist = [
        15,
        15,
        50,
        10,
        1.0,
        20,
        20,
        100,
        100,
        20,
        20,
        100,
        100,
        20,
        20,
        20,
        20,
        20,
    ]
    ylimits = dict(zip(boxplotlist, axlimitlist))
    statsDF.sort_values(by=["washcycle"], inplace=True)
    for i in boxplotlist:
        tmpDF = statsDF[statsDF["SAMPLE"] == "Device_4"]
        sns.set()
        sns.set_style("whitegrid")
        figsns, axsns = plt.subplots(figsize=(12, 8))

        ax = sns.boxplot(
            x="washcycle", y=i, data=statsDF, hue="sliceTime", ax=axsns, dodge=True
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_ylim([0, ylimits[i]])

        figsns.tight_layout()
        fileName = tmpDF["SAMPLE"].values[0] + "_" + i + "_washcycle_boxplot.png"
        figsns.savefig(savePath / fileName, dpi=200)
        figsns.clear()
        plt.close(figsns)

    # --------------------------------------------------
    ## Dirtribution of events per slice
    for group in statsDF.groupby(["ident"]):

        tmpDF = group[1]
        sns.set()
        sns.set_style("whitegrid")
        figsns, axsns = plt.subplots(figsize=(12, 8))

        ax = sns.distplot(tmpDF["NUMBEROFEVENTS"], norm_hist=True)

        figsns.tight_layout()
        fileName = tmpDF["ident"].values[0] + "_" + "_distplot.png"
        figsns.savefig(savePath / fileName, dpi=200)
        figsns.clear()
        plt.close(figsns)
