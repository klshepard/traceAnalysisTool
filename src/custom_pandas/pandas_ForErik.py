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

resultsPath = Path.cwd().parent / "DataAndResults/results"
savePath = resultsPath
files_for_frame = sorted(resultsPath.rglob("dataframe_atruntime_*.csv"))

# import all data frames in folder
DFlist = []
for fileN in files_for_frame:
    DFlist.append(pd.read_csv(fileN))
statsDF = pd.concat(DFlist)  # read_csv(files_for_frame[0]).reset_index()
print(statsDF)
print(statsDF.columns)
print(statsDF["exp_name"])
print(statsDF["exp_name"].str.split("_"))

statsDF["ident"] = statsDF["exp_name"].str.split("_Experiment_").str[1]
statsDF["run"] = statsDF["exp_name"].str.split("_").str[-2]
statsDF["run"] = statsDF["run"].astype("category")
print(statsDF["run"])
# normalize event count for comparison

numberOfEvents = statsDF["NUMBEROFEVENTS"].groupby(statsDF["ident"]).sum()
print(numberOfEvents)
print(statsDF["VAG"])
statsDF.sort_values(by=["exp_name", "ident"], inplace=True)

#### Main loop
if __name__ == "__main__":
    scatterlist = [
        "K_ON_double_fast",
        "K_ON_double_slow",
        "K_ON_single",
        "K_OFF_double_fast",
        "K_OFF_double_slow",
        "K_OFF_single",
    ]

    for i in scatterlist:

        sns.set()
        sns.set_style("whitegrid")
        figsns, axsns = plt.subplots(figsize=(8, 8))
        ax = sns.scatterplot(x="VAG", y=i, data=statsDF, hue="ident", ax=axsns)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
        # ax.set_ylim([0,20])
        figsns.tight_layout()
        fileName = i + "-vs-vg.png"
        figsns.savefig(savePath / fileName, dpi=600)
        figsns.clear()
        plt.close(figsns)
