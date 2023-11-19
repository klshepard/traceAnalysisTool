import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
from scipy import stats
from scipy import special
import pandas as pd
from IPython.display import Image, display, HTML
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns

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
from lmfit.models import LinearModel
import scipy.optimize
import csv

pd.set_option("display.max_rows", None)

HTML(
    """<style type="text/css">
  table.dataframe td, table.dataframe th {
    max-width: none;
    white-space: normal;
    line-height: normal;
    padding: 0.3em 0.5em;
  }
</style>
"""
)


def batch_run(pythonCMD, NCPU, MEM, LR, QU, LOC="/proj/jakobuchheim/Repositories"):
    if QU == "PI":
        queue = "BATCHPI=1"
    else:
        queue = "BATCH=1"

    l = os.system(
        "cd {sx}/Schnellstapel/; make {s} ITHREADS={s0} IMEM={s1} LINRACK={s2} CCMD={s3} jupytermake".format(
            sx=LOC, s=queue, s0=NCPU, s1=MEM, s2=LR, s3=pythonCMD
        ).replace(
            "$(THREADS)", str(NCPU)
        )
    )
    return l


def checkout(branch, LOC="/proj/jakobuchheim/Repositories"):
    l = os.system(
        "cd {s0}/DataAndResults/; git fetch --all; git checkout -f {s1}".format(
            s0=LOC, s1=branch
        )
    )
    return l


def make_branch(branch, data, LOC="/proj/jakobuchheim/Repositories"):
    l = os.system(
        "cd {s0}/DataAndResults/; git fetch --all; git checkout -b {s1}".format(
            s0=LOC, s1=branch
        )
    )
    l = os.system("cd {s0}/DataAndResults/; rm data.txt".format(s0=LOC))
    with open(
        "{s0}/DataAndResults/data.txt".format(s0=LOC), "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        for line in data:
            csv_writer.writerow([line])
    l = os.system(
        'cd {s0}/DataAndResults/; git add data.txt; git commit -m "auto create branch"; git push --set-upstream origin {s1}'.format(
            s0=LOC, s1=branch
        )
    )
    return l


def make_branchwopush(branch, data, LOC="/proj/jakobuchheim/Repositories"):
    l = os.system(
        "cd {s0}/DataAndResults/; git fetch --all; git checkout -b {s1}".format(
            s0=LOC, s1=branch
        )
    )
    l = os.system("cd {s0}/DataAndResults/; rm data.txt".format(s0=LOC))
    with open(
        "{s0}/DataAndResults/data.txt".format(s0=LOC), "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        for line in data:
            csv_writer.writerow([line])
    l = os.system(
        'cd {s0}/DataAndResults/; git add data.txt; git commit -m "auto create branch"'.format(
            s0=LOC
        )
    )
    return l
