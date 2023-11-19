import math
import numpy as np
import scipy.io as sio
import scipy.signal as ssi
import pandas as pd
import sys
import distutils.dir_util
import shutil
from pathlib import Path
import os
from datetime import datetime
from datetime import timedelta
import time
import decimal
import logging

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
matplotlib.use("Agg")


def enclose_the_pandas():
    """
    Basic setup for Pandas operations.

    Architectually, relies on reading variables from this file's namespace, as opposed to explicity function params.  BUG?

    Parameters
    ----------
    None.

    Returns
    -------
    resultsPath : Path
        The path to drop in the results
    savePath : Path
        The path to drop in the results -- who architected this thing and how does it differ from the above?  BUG
    statsDF : DataFrame()
        The dataframe that was loaded from the experimental big-data run.

    """

    resultsPath = Path.cwd().parent / "DataAndResults/results"
    savePath = resultsPath
    files_for_frame = sorted(resultsPath.rglob("dataframe_atruntime_*.csv"))

    # import all data frames in folder
    DFlist = []

    for fileN in files_for_frame:
        DFlist.append(pd.read_csv(fileN))

    statsDF = pd.concat(DFlist)

    return resultsPath, savePath, statsDF


def get_events_files():
    """
    Get the events files lists.

    Architectually, relies on reading variables from this file's namespace, as opposed to explicity function params.  BUG?

    Parameters
    ----------
    None.

    Returns
    -------
    eventsfiles : List
        List of the events csvs.

    """
    resultsPath = Path.cwd().parent / "DataAndResults/results/events"
    files_for_frame = sorted(resultsPath.rglob("*_events.csv"))
    return files_for_frame


def get_cols_from_name(filename: Path) -> dict:
    """
    Given a file path, parses it for the purposes of getting a dataframe out. TODO maybe depricated, since I write this all to the event.csv now.

    TODO could take input from load_utils.parse_file_name(), but this is tight enough for Chimera data only for now.

    Architectually, relies on reading variables from this file's namespace, as opposed to explicity function params.  BUG?

    Parameters
    ----------
    filename : List of Paths
        A Path() objects that point to event CSVs.

    Returns
    -------
    out_dict : Dict()
        Dict of things I care about from this filename.
    """
    out_dict = {}
    file_name = str(filename)
    print(filename.stem.split("_"))
    out_dict["device"] = filename.stem.split("_")[1]
    # any others should come from the output of event_utils, and be written right in the CSV.
    print(out_dict)
    return out_dict


def make_events_df(events_files: list) -> pd.DataFrame:
    """
    Given an list of events csvs, returns a single massive dataframe that has all the event info.

    Architectually, relies on reading variables from this file's namespace, as opposed to explicity function params.  BUG?

    Parameters
    ----------
    events_files : List of Paths
        A List() of Path() objects that point to event CSVs.

    Returns
    -------
    eventsDF : pandas DF
        Dataframe containing events information
    """
    assert len(events_files) > 0
    the_frame = pd.read_csv(events_files[0])
    for a_file in events_files[1:]:
        frame = pd.read_csv(a_file)
        # TODO add relevant cols
        the_frame = the_frame.append(
            frame, ignore_index=True
        )  # TODO unfuck dtype warning.

    return the_frame


def calculate_conductivity(V, d, L, C, sigma_surf):
    V_units = []
    I = []
    C = decimal.Decimal(C)
    d = decimal.Decimal(d)
    L = decimal.Decimal(L)
    sigma_surf = decimal.Decimal(sigma_surf)
    for v in V:
        V_units.append(v * 1e3)
        I.append(
            decimal.Decimal(1e12) * decimal.Decimal(v) * G(C, d, L, sigma_surf)
        )  # picoamps

    return V_units, I


def G(C, d, L, sigma):
    ## takes decimals only.
    ## Do the actual math, following: https://pubs.acs.org/doi/full/10.1021/nl052107w
    pi = decimal.Decimal(math.pi)
    e = decimal.Decimal(1.6e-19)  # Coulomb
    mu_k = decimal.Decimal(7.616e-8)  # m^2 per Volt per second
    mu_cl = decimal.Decimal(7.909e-8)  # m^2 per Volt per second
    q_k = e
    q_cl = -e
    n = (
        C * decimal.Decimal(1e3) * decimal.Decimal(6.023e23)
    )  # mol/liter * 1e3 liter/m^3 * 6.023e23/mol gives "particles"/m^3
    # TODO assuming sigma is positive, screen is mu_cl, check otherwise.
    G = (
        (pi / 4)
        * (d ** 2 / L)
        * ((mu_k * q_k + mu_cl * q_cl) * n + mu_cl * 4 * sigma / d)
    )
    return G
