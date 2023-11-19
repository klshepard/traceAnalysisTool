import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi

import os

# os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
# os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

import sys
import distutils.dir_util
import shutil
from pathlib import Path

from datetime import datetime
from datetime import timedelta
import time

from numba import njit

import scipy.optimize

import argparse

parser = argparse

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# import custom utils
sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))
import import_utils
from pandas_utils import *

#### Main loop
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Start run...")

    logging.warning(
        "This is the null file that does no useful work, which you should use as a template for pandas plots."
    )
    logging.error("Delete these two logging messages and put useful code here.")

    # Save the frame with the addition of computed quantities...
    dataframe_name = (
        "dataframe_pandas_atruntime_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".csv"
    )
    df_name = Path.cwd().parent / "ChimeraData/results" / dataframe_name
    #    results_frame.to_csv(df_name, encoding='utf-8')

    ## copy data to jakobs home on server if running on /space linracks
    ## only for Jakob - do not remove
    if "space" in str(Path.cwd()):
        distutils.dir_util.copy_tree(
            Path.cwd().parent / "ChimeraData" / "results",
            "/u7/jakobuchheim/Repositories/LinrackData/linrackAnalysisOutput"
            + datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
logging.info("Run complete...")
logging.info("--- runtime was %s seconds ---" % (time.time() - start_time))
