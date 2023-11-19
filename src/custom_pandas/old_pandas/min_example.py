import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
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

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
matplotlib.use("Agg")


def run_main(statsDF, scatterlist, saveName, fileName):
    """
    Runs the main function for plotting using the list, pulling from statsDF and the written dataframe at runtime,

    Parameters
    ----------
    statsDF : Pandas DF
        Date
    scatterlist : List
        The list of thing over which to scatterplot.
    saveName : PosixPath
        The path to save to.
    fileName : string
        The filename to save.

    Returns
    -------
    None
        Will write a file.

    """
    for i in scatterlist:
        sns.set()
        sns.set_style("whitegrid")
        figsns, axsns = plt.subplots(figsize=(5, 5))
        ax = sns.scatterplot(x="I_mean", y=i, data=statsDF, hue="SAMPLE", ax=axsns)
        figsns.tight_layout()
        fileName = i + "--vs--I_mean.png"
        figsns.savefig(savePath / fileName, dpi=200)
        figsns.clear()
        plt.close(figsns)

    return


if __name__ == "__main__":
    scatterlist = ["VAC"]
    savePath = Path.cwd().parent / "DataAndResults/results"
    files_for_frame = sorted(resultsPath.rglob("dataframe_atruntime_*.csv"))
    # import all data frames in folder
    DFlist = []
    for fileN in files_for_frame:
        DFlist.append(pd.read_csv(fileN))
    statsDF = pd.concat(DFlist)
    statsDF["run"] = statsDF["exp_name"].str.split("_").str[-2]
    statsDF["run"] = statsDF["run"].astype("category")
    # calculate some stats
    print(statsDF["VAG"])
    # sort DF for better plotting:
    statsDF.sort_values(by=["exp_name"], inplace=True)
    print(statsDF)
    print(statsDF.columns)
    run_main(statsDF, scatterlist, savePath, fileName)
