import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
import distutils.dir_util
import shutil
from pathlib import Path
import os
from datetime import datetime
from datetime import timedelta
import time

import scipy.optimize

import argparse

parser = argparse

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
matplotlib.use("Agg")

#### Main loop
if __name__ == "__main__":

    resultsPath = Path.cwd().parent / "DataAndResults/pandas_Device_4"
    savePath = resultsPath
    files_for_frame = sorted(resultsPath.rglob("dataframe_atruntime_*.csv"))

    # import all data frames in folder
    DFlist = []
    for fileN in files_for_frame:
        DFlist.append(pd.read_csv(fileN))
    statsDF = pd.concat(DFlist, ignore_index=True).reset_index()

    # parse srtn concentration from comment

    statsDF.loc[statsDF["COMMENTS"].str.contains("ser", na=False), "type"] = "srtn"
    statsDF.loc[statsDF["COMMENTS"].str.contains("probe", na=False), "type"] = "apt"
    statsDF.loc[statsDF["COMMENTS"].str.contains("dop", na=False), "type"] = "dop"
    statsDF.loc[
        statsDF["COMMENTS"].str.contains("aptonly_dop", na=False), "type"
    ] = "apt"
    statsDF.loc[statsDF["COMMENTS"].str.contains("srtn", na=False), "type"] = "srtn"
    statsDF.loc[
        statsDF["COMMENTS"].str.contains("aptonlyrtp2V_1", na=False), "type"
    ] = "apt"
    statsDF.loc[statsDF["COMMENTS"].str.contains("HIAA", na=False), "type"] = "hiaa"
    statsDF.loc[statsDF["COMMENTS"].str.contains("50n", na=False), "conc"] = 50.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("5n", na=False), "conc"] = 5.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("1n", na=False), "conc"] = 1.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("100n", na=False), "conc"] = 100.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("100p", na=False), "conc"] = 0.1
    statsDF.loc[
        statsDF["COMMENTS"].str.contains("aptonlyrtp2V_1", na=False), "conc"
    ] = 0.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("25n", na=False), "conc"] = 25.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("10n", na=False), "conc"] = 10.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("5u", na=False), "conc"] = 500.0
    # statsDF.loc[statsDF["COMMENTS"].str.contains("dop", na=False),'conc'] = 0.0

    statsDF.loc[
        statsDF["exp_name"].str.contains("wash1", na=False), "washcycle"
    ] = "wash1"
    statsDF.loc[
        statsDF["exp_name"].str.contains("wash2", na=False), "washcycle"
    ] = "wash2"
    statsDF.loc[
        statsDF["exp_name"].str.contains("wash3", na=False), "washcycle"
    ] = "wash3"
    statsDF.loc[
        statsDF["exp_name"].str.contains("initial", na=False), "washcycle"
    ] = "incubation"

    # fill nan for no slice
    statsDF["sliceTime"].fillna(statsDF["TOTALTIME"], inplace=True)

    # add some columns for better identification

    statsDF["ident"] = (
        statsDF["type"]
        + "-"
        + statsDF["conc"].astype("str")
        + "nM_"
        + statsDF["SAMPLE"].astype("str")
        + "_"
        + statsDF["sliceTime"].astype("str")
    )
    # normalize event count for comparison
    statsDF["EVENTSPERSEC"] = statsDF["NUMBEROFEVENTS"] / statsDF["sliceTime"]

    # calculate some stats
    numberOfEvents = statsDF["NUMBEROFEVENTS"].groupby(statsDF["ident"]).sum()
    print(numberOfEvents)

    # sort DF for better plotting:
    statsDF.sort_values(by=["washcycle"], inplace=True)

    ## ----------------------------------
    # boxplots
    boxplotlist = [
        "K_ON_double_fast",
        "K_ON_double_slow",
        "K_ON_single",
        "K_OFF_double_fast",
        "K_OFF_double_slow",
        "K_OFF_single",
        "EVENTSPERSEC",
        "P_DOWN",
    ]
    axlimitlist = [100, 15, 15, 150, 40, 50, 10, 1.0]
    ylimits = dict(zip(boxplotlist, axlimitlist))
    for i in boxplotlist:

        sns.set()
        sns.set_style("whitegrid")
        figsns, axsns = plt.subplots(figsize=(12, 8))

        ax = sns.boxplot(
            x="washcycle", y=i, data=statsDF, hue="sliceTime", ax=axsns, dodge=True
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_ylim([0, ylimits[i]])

        figsns.tight_layout()
        fileName = i + "_boxplot.png"
        figsns.savefig(savePath / fileName, dpi=200)
        figsns.clear()
        plt.close(figsns)

    boxplotlist.extend(
        ["K_ON_ERR_double", "K_ON_ERR_single", "BASELINESTD", "conc"]
    )  # ,'type'])

    print(boxplotlist)

    subDF = statsDF[boxplotlist]
    subDF = subDF.dropna()
    print(subDF)
    print(subDF[subDF["K_OFF_double_fast"] == subDF.K_OFF_double_fast.max()])
    subDropDF = subDF[(np.abs(stats.zscore(subDF)) < 7).all(axis=1)]
    subDropDF = subDropDF[
        subDropDF["K_OFF_double_fast"] < statsDF.loc[1, "MINLENGTH"] / 4
    ]
    print(
        subDropDF[subDropDF["K_OFF_double_fast"] == subDropDF.K_OFF_double_fast.max()]
    )
    # subDropDF = subDropDF[(np.abs(stats.zscore(subDropDF)) < 4).all(axis=1)]
    print(
        subDropDF[subDropDF["K_OFF_double_fast"] == subDropDF.K_OFF_double_fast.max()]
    )
    print(subDropDF)

    sns.pairplot(
        subDropDF,
        hue="conc",
        diag_kind="kde",
        vars=[
            "K_ON_ERR_double",
            "K_ON_ERR_single",
            "BASELINESTD",
            "K_ON_double_fast",
            "K_ON_double_slow",
            "K_ON_single",
            "K_OFF_double_fast",
            "K_OFF_double_slow",
            "K_OFF_single",
            "EVENTSPERSEC",
            "P_DOWN",
        ],
    )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
    fileName = "pairplot.png"
    plt.savefig(savePath / fileName, dpi=200)

    sub2 = statsDF[
        (
            (statsDF["sliceTime"] == 60.0)
            & (statsDF["conc"] < 200.0)
            & (statsDF["type"].str.contains("srtn"))
        )
    ]
    print(sub2)
    sns.set()
    sns.set_style("whitegrid")
    figsns, axsns = plt.subplots(figsize=(8, 8))

    sns.scatterplot(x="conc", y="EVENTSPERSEC", data=sub2, hue="SAMPLE", ax=axsns)
    # sns.scatterplot(x="conc", y='EVENTSPERSEC', data=statsDF)#[(statsDF['sliceTime'] == 60 & statsDF['type'].str.contains('srtn'))], ax = axsns)

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    figsns.tight_layout()
    fileName = "EVENTSPERSEC_vs_conc.png"
    figsns.savefig(savePath / fileName, dpi=200)
    figsns.clear()
    plt.close(figsns)

    sns.set()
    sns.set_style("whitegrid")
    figsns, axsns = plt.subplots(figsize=(8, 8))

    ax = sns.scatterplot(
        x="K_OFF_double_fast", y="K_OFF_ERR_double", hue="ident", data=statsDF, ax=axsns
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    figsns.tight_layout()
    fileName = "K_OFF_ERR_double_vs_k_OFF_fast_boxplot.png"
    figsns.savefig(savePath / fileName, dpi=600)
    figsns.clear()
    plt.close(figsns)

    sns.set()
    sns.set_style("whitegrid")
    figsns, axsns = plt.subplots(figsize=(8, 8))

    ax = sns.scatterplot(
        x="K_OFF_double_fast",
        y="K_OFF_double_slow",
        hue="ident",
        data=statsDF,
        ax=axsns,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    figsns.tight_layout()
    fileName = "K_OFF_ERR_double_vs_k_OFF_fast_boxplot.png"
    figsns.savefig(savePath / fileName, dpi=600)
    figsns.clear()
    plt.close(figsns)

    # print(statsDF.groupby(['SAMPLE','type','conc','sliceTime']).agg({K_ON_double_fast : [np.mean, 'mean']}))
    # statsDF.groupby('conc')['K_ON_double_fast'].agg(np.cov, aweights=statsDF.groupby('conc')['K_ON_ERR_double'].groups)

    stdlist = boxplotlist
    stdlist.extend(["NUMBEROFEVENTS", "SAMPLE", "type", "sliceTime"])
    subDF = statsDF[stdlist]
    subDF = subDF.dropna()

    for group in subDF.groupby(["SAMPLE", "type", "conc", "sliceTime"]):

        group[1]["K_ON_ERR_double"].clip(lower=0.1, inplace=True)
        weights = group[1]["NUMBEROFEVENTS"] + 1 / group[1]["K_ON_ERR_double"]
        # print(weights)
        stdWeighted = np.sqrt(np.cov(group[1]["K_ON_double_fast"], aweights=weights))
        averageWeighted = np.average(group[1]["K_ON_double_fast"], weights=weights)
        print(stdWeighted - group[1]["K_ON_double_fast"].std())
