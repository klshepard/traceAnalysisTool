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

    resultsPath = Path("/proj/jakobuchheim/share/newChunk")
    savePath = resultsPath
    files_for_frame = sorted(
        resultsPath.rglob("Device_*[!_]/dataframe_atruntime_*.csv")
    )

    # import all data frames in folder
    DFlist = []
    for fileN in files_for_frame:
        DFlist.append(pd.read_csv(fileN))
    statsDF = pd.concat(DFlist, ignore_index=True).reset_index()

    # parse srtn concentration from comment

    statsDF.loc[statsDF["SAMPLE"] == "AV_R4C1", "SAMPLE"] = "Device_1"
    statsDF.loc[statsDF["SAMPLE"] == "AV_R4C4", "SAMPLE"] = "Device_2"
    statsDF.loc[statsDF["SAMPLE"] == "old3_tl", "SAMPLE"] = "Device_3"
    statsDF.loc[statsDF["SAMPLE"] == "old3_bl", "SAMPLE"] = "Device_5"
    statsDF.loc[statsDF["SAMPLE"] == "n2_tr", "SAMPLE"] = "Device_4"

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
    statsDF.loc[
        statsDF["COMMENTS"].str.contains("aptonly_rtp2V", na=False), "type"
    ] = "apt"
    statsDF.loc[statsDF["COMMENTS"].str.contains("blank", na=False), "type"] = "apt"
    statsDF.loc[statsDF["COMMENTS"].str.contains("HIAA", na=False), "type"] = "hiaa"
    statsDF.loc[statsDF["COMMENTS"].str.contains("50n", na=False), "conc"] = 50.0
    statsDF.loc[statsDF["COMMENTS"].str.contains("500p", na=False), "conc"] = 0.5
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
    statsDF.loc[statsDF["COMMENTS"].str.contains("blank", na=False), "conc"] = 0.0

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
    statsDF.sort_values(
        by=["SAMPLE", "sliceTime", "type", "conc", "washcycle"], inplace=True
    )

    statsDF.to_csv(
        resultsPath / "giantDFwithAllResults.csv", sep=",", encoding="utf-8", index=True
    )
