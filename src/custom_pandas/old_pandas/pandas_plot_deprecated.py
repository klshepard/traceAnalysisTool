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

    ## Input argument parser
    # General run definitions
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--HEKA", action="store_true", help="reads HEKA ascii data from HEKAData folder"
    )
    args = parser.parse_args()

    if args.HEKA:
        resultsPath = Path.cwd().parent / "HEKAData/results"
    else:
        resultsPath = Path.cwd().parent / "ChimeraData/results"

    # Clear previous results
    resultsArchivePath = resultsPath.parent / "resultsArchive"
    outputPlace = "AnalysisOutput_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    distutils.dir_util.copy_tree(resultsPath, str(resultsArchivePath / outputPlace))

    # set up run constants
    dataPath = resultsPath.parent / "dump"
    VA = import_utils.get_VA(dataPath)
    file_for_frame = sorted(resultsPath.rglob("dataframe_atruntime_*.csv"))[-1]
    logging.info("I am loading \n" + str(file_for_frame))
    results_frame = pd.read_csv(
        file_for_frame,
        usecols=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ],
    )  # BUG clearly should clean that up...

    ## Fix gate voltages here...
    ## VOLTFIX REQ
    for i, j in enumerate(zip(results_frame.VAC, results_frame.vg)):
        results_frame.vg[i] = import_utils.fix_voltage(VA, j[0], j[1])

    logging.info("Dataframe looks like: \n" + results_frame.to_string())

    # BUG the three pairplots how have problems after a single colum is all NaNs, becuase it's a single exponential fit.
    # df = results_frame.dropna()
    # g = sns.pairplot(df.dropna(how='all'), hue='conc')
    # g.savefig(resultsPath / 'pairplot_all.png', dpi=600)
    # plt.close()

    # g = sns.pairplot(results_frame.dropna(), hue='conc', vars=['VAC', 'vg', 'I_gate', 'I_mean', 'I_std'])
    # g.savefig(resultsPath / 'pairplot_raw.png', dpi=600)
    # plt.close()

    for ex in results_frame.groupby(["sample", "config"]):
        model_frame = pd.DataFrame(columns=["device", "config", "conc", "A", "x0", "B"])
        df = ex[1][["VAC", "I_mean", "conc"]]
        df["I_mean"] = df["I_mean"] / 1000
        df["VAC"] = df["VAC"] / 1000
        df["conc"] = df["conc"].apply(lambda x: str(x) + " M")
        ex_device = ex[0][0]
        for thing in df.groupby(["conc"]):
            ex_conc = str(thing[0])
            fit_frame = pd.DataFrame(thing[1])
            fit_frame = fit_frame.sort_values(by=["VAC"])
            if len(fit_frame) > 3:
                voltages = fit_frame["VAC"].to_numpy()
                amperages = fit_frame["I_mean"].to_numpy()
                fp = fit_single(voltages, amperages)
                assert len(fp) == 3
                config = ex[0][1]
                #                offset = my_exp(0, fp[0], fp[1], fp[2])
                model_frame = model_frame.append(
                    pd.DataFrame(
                        {
                            "device": [ex_device],
                            "config": config,
                            "conc": [ex_conc],
                            "A": [fp[0]],
                            "x0": [fp[1]],
                            "B": [fp[2]],
                        }
                    )
                )
        save_name = ex[0][0] + "_config_" + ex[0][1] + "_modelframe.csv"
        model_frame.to_csv(Path.cwd().parent / "ChimeraData/results" / save_name)

    files_for_model = sorted(resultsPath.rglob("*_modelframe.csv"))
    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "I_mean", "conc"]]
        df["I_mean"] = df["I_mean"] / 1000
        df["VAC"] = df["VAC"] / 1000
        df["I_model"] = np.nan
        # df['vg'] = df['vg']/1000
        # df['vg_halfcell'] = df['VAC']/2-df['vg'] # anode minus gate.
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        df = df.sort_values(by=["VAC"])
        ex_device = ex[0][0]
        for (
            index,
            row,
        ) in df.iterrows():  # BUG -- aweful and awefil -- pulls in model EVERY TIME...
            ex_conc = row["conc"]
            for name in files_for_model:
                if (
                    name.stem.split("_")[0] == ex_device
                    and name.stem.split("_")[2] == "C"
                ):
                    logging.info("Found a model to load.")
                    model = pd.read_csv(name, usecols=[1, 2, 3, 4, 5, 6])
                    A = model.loc[
                        (model["device"] == ex_device) & (model["conc"] == ex_conc)
                    ]["A"]
                    x0 = model.loc[
                        (model["device"] == ex_device) & (model["conc"] == ex_conc)
                    ]["x0"]
                    B = model.loc[
                        (model["device"] == ex_device) & (model["conc"] == ex_conc)
                    ]["B"]
                    df.at[index, "I_model"] = my_exp(row["VAC"], A, x0, B)[0]
        the_fig, the_ax = plt.subplots(figsize=(8, 8))
        the_ax.grid()
        the_ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        the_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
        the_ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        the_ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        the_ax.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        the_ax.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        sns.scatterplot(x="VAC", y="I_mean", data=df, hue="conc", ax=the_ax)
        sns.lineplot(x="VAC", y="I_model", data=df, hue="conc", ax=the_ax)
        plt.xlabel(r"$V_{AC}$ [V]")
        plt.ylabel(r"$I_{A}$ [nA]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_IV.png"
        the_fig.savefig(Path.cwd().parent / "ChimeraData/results" / fileName, dpi=300)
        plt.close()

    files_for_model = sorted(resultsPath.rglob("*_modelframe.csv"))
    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["sample", "conc", "COMMENTS"]
    ):
        df = ex[1][["VAC", "I_mean", "I_gate"]]
        df = df.sort_values(by=["VAC"])
        ex_conc = str(ex[0][1]) + " M"
        ex_device = ex[0][0]
        for name in files_for_model:
            mod_device = name.stem.split("_")[0]
            if mod_device == ex_device and name.stem.split("_")[2] == "C":
                logging.info("Found a model to load.")
                model = pd.read_csv(name, usecols=[1, 2, 3, 4, 5, 6])
                A = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["A"]
                x0 = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["x0"]
                B = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["B"]
        sub_cur = []
        for v in df["VAC"]:
            sub_cur.append(my_exp(v, A, x0, B)[0])
        the_fig, the_ax = plt.subplots(figsize=(8, 8))
        print(sub_cur)
        the_ax.grid()
        #        plt.xlim(-.6, 0.15)
        #        plt.ylim(-5, 2)
        the_ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        the_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        the_ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        the_ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        the_ax.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        the_ax.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        the_ax.plot(
            df["VAC"].to_numpy() / 1000,
            df["I_mean"].to_numpy() / 1000,
            ".",
            label="$I_{A}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAC"].to_numpy() / 1000,
            sub_cur,
            ".",
            label="Model $I_{inject}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAC"].to_numpy() / 1000,
            df["I_mean"].to_numpy() / 1000 - sub_cur,
            ".",
            label="$I_{pore}$ [nA]",
            markersize=10,
        )
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "_MKCl_"
            + ex[0][2]
            + "_IVAC_model.png"
        )
        plt.xlabel(r"$V_{AC}$")
        the_ax.legend()
        the_fig.savefig(Path.cwd().parent / "ChimeraData/results" / fileName, dpi=600)
        plt.close()

    del model

    files_for_model = sorted(resultsPath.rglob("*_modelframe.csv"))
    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["sample", "conc", "COMMENTS", "VAC"]
    ):
        df = ex[1][["VAC", "I_mean", "I_gate", "VAG"]]
        df["vg_halfcell"] = df["VAC"] / 2 - df["VAG"]  # anode minus gate.
        print(df["vg_halfcell"])
        df = df.sort_values(by=["VAC"])
        ex_conc = str(ex[0][1]) + " M"
        ex_device = ex[0][0]
        for name in files_for_model:
            mod_device = name.stem.split("_")[0]
            if mod_device == ex_device and name.stem.split("_")[2] == "C":
                logging.info("Found a model to load.")
                model = pd.read_csv(name, usecols=[1, 2, 3, 4, 5, 6])
                A = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["A"]
                x0 = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["x0"]
                B = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["B"]
        sub_cur = []
        for v in df["vg_halfcell"]:
            sub_cur.append(my_exp(v / 1000, A, x0, B).to_numpy()[0])
        r = df.shape[0]
        if r < 3:
            continue
        the_fig, the_ax = plt.subplots(figsize=(8, 8))
        the_ax.grid()
        plt.xlim(-0.1, 0.4)
        plt.ylim(-30, 20)
        the_ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        the_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        the_ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        the_ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        the_ax.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        the_ax.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            sub_cur,
            ".",
            label="Model $I_{inject}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            df["I_mean"].to_numpy() / 1000,
            ".",
            label="$I_{A}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            df["I_gate"].to_numpy() / 1000,
            ".",
            label="$I_{gate}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            df["I_mean"].to_numpy() / 1000 - df["I_gate"].to_numpy() / 1000,
            ".",
            label="$I_{pore}$ [nA]",
            markersize=10,
        )
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + ex[0][2]
            + "_VAC_"
            + str(ex[0][3])
            + "MKCl_IVg_subtraction.png"
        )
        plt.xlabel(r"$V_{AG}$")
        the_ax.legend()
        the_fig.savefig(Path.cwd().parent / "ChimeraData/results" / fileName, dpi=600)
        plt.close()

    logging.info("Start model gate plots.")

    files_for_model = sorted(resultsPath.rglob("*_modelframe.csv"))
    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["sample", "conc", "COMMENTS", "VAC"]
    ):
        df = ex[1][["VAC", "I_mean", "I_gate", "VAG"]]
        df = df.sort_values(by=["VAC"])
        ex_conc = str(ex[0][1]) + " M"
        ex_device = ex[0][0]
        sub_cur = []
        for name in files_for_model:
            mod_device = name.stem.split("_")[0]
            if mod_device == ex_device and name.stem.split("_")[2] == "C":
                logging.info("Found a model to load.")
                model = pd.read_csv(name, usecols=[1, 2, 3, 4, 5, 6])
                A = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["A"]
                x0 = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["x0"]
                B = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["B"]
                for v in df["VAC"]:
                    sub_cur.append(my_exp(v / 1000, A, x0, B).to_numpy()[0])
        r = df.shape[0]
        print(df.shape)
        if r < 12:
            continue
        the_fig, the_ax = plt.subplots(figsize=(8, 8))
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            df["I_mean"].to_numpy() / 1000,
            ".",
            label="$I_{A}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            sub_cur,
            ".",
            label="Model $I_{inject}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(
            df["VAG"].to_numpy() / 1000,
            df["I_mean"].to_numpy() / 1000 - sub_cur,
            ".",
            label="$I_{pore}$ [nA]",
            markersize=10,
        )
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + ex[0][2]
            + "_VAC_"
            + str(ex[0][3])
            + "MKCl_IVg_model.png"
        )
        plt.xlabel(r"$V_{AG}$")
        the_ax.legend()
        the_fig.savefig(Path.cwd().parent / "ChimeraData/results" / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        snsIVplot(ex, resultsPath)
        snsIVplotregression(ex, resultsPath)

    # Only do normalized vg plots on Series where the pin is down
    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["sample", "conc"]
    ):
        snsIVgateplot(ex, resultsPath)

    # Only do normalized vg plots on Series where the pin is down and the gate current is low.
    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["sample", "conc"]
    ):
        df = ex[1][["R", "V_g_norm", "I_mean", "I_gate", "COMMENTS"]]
        df = df[df["I_gate"] / df["I_mean"] < 0.2]
        r = df.shape[0]
        if r < 4:
            continue
        plt.figure(figsize=(8, 8))
        IVplot2 = sns.scatterplot(
            x="V_g_norm", y="R", hue="COMMENTS", data=df, zorder=2
        )

        # reposition legend
        box = IVplot2.get_position()
        IVplot2.set_position(
            [box.x0, box.y0, box.width, box.height * 0.85]
        )  # resize position
        IVplot2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2)

        plt.xlabel(r"$\frac{V_g}{V_{AC}}$")
        plt.ylabel(r"$\frac{V_{AC}}{I_A}$ [M$\Omega$]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_R-Vnorm_lowgate.png"
        IVplot2.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["sample", "conc"]
    ):
        df = ex[1][["R", "V_g_norm", "I_mean", "I_gate", "COMMENTS"]]
        r = df.shape[0]
        if r < 4:
            continue
        plt.figure(figsize=(8, 8))
        IVplot2 = sns.scatterplot(
            x="V_g_norm", y="R", hue="COMMENTS", data=df, zorder=2
        )

        # reposition legend
        box = IVplot2.get_position()
        IVplot2.set_position(
            [box.x0, box.y0, box.width, box.height * 0.85]
        )  # resize position
        IVplot2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2)

        plt.xlabel(r"$\frac{V_g}{V_{AC}}$")
        plt.ylabel(r"$\frac{V_{AC}}{I_A}$ [M$\Omega$]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_R-Vnorm.png"
        IVplot2.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    # at fixed VAC, plot runs of I_mean vs v_g
    for ex in results_frame.groupby(["sample", "conc", "VAC"]):
        df = ex[1][["VAG", "I_mean", "COMMENTS"]]
        r = df.shape[0]
        if r < 5:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAG", y="I_mean", data=df, hue="COMMENTS", zorder=2)
        plt.xlabel(r"$V_{AG}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "MKCl_VAC_"
            + str(ex[0][2])
            + "mV_Ia-Vg.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    # Plot I_mean only for "small" gate currents, with the pin down
    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["VAC", "VAG", "I_mean", "I_gate", "COMMENTS"]]
        df = df[df["COMMENTS"].str.contains("pindown")]
        df = df[df["I_gate"] / df["I_mean"] < 0.2]
        r = df.shape[0]
        if r < 4:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAC", y="I_mean", data=df, hue="COMMENTS", zorder=2)
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_Ia-VAC_lowgate.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    # at fixed VAC, plot runs of I_mean vs v_g
    for ex in results_frame.groupby(["sample", "conc", "VAC"]):
        df = ex[1][["VAG", "I_mean", "COMMENTS", "I_gate"]]
        df = df[df["COMMENTS"].str.contains("pindown")]
        df = df[df["I_gate"] / df["I_mean"] < 0.2]
        r = df.shape[0]
        if (
            r < 4
        ):  # if whatever calculation you did above got you less than four points, forget it...
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAG", y="I_mean", data=df, hue="COMMENTS", zorder=2)
        plt.xlabel(r"$V_{AG}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "MKCl_VAC_"
            + str(ex[0][2])
            + "mV_Ia-Vg_lowgate.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    # at fixed VAC, plot runs of I_mean vs v_g
    for ex in results_frame.groupby(["sample", "conc", "VAC"]):
        df = ex[1][["VAG", "I_mean", "COMMENTS", "I_gate"]]
        df = df[df["COMMENTS"].str.contains("pindown")]
        r = df.shape[0]
        if (
            r < 4
        ):  # if whatever calculation you did above got you less than four points, forget it...
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAG", y="I_mean", data=df, hue="COMMENTS", zorder=2)
        plt.xlabel(r"$V_{AG}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "MKCl_VAC_"
            + str(ex[0][2])
            + "mV_Ia-Vg.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    pattern = r"pinup|thirdpinin|percent"
    frame = results_frame[results_frame["COMMENTS"].str.contains(pattern)]
    for ex in frame.groupby(["sample"]):
        df = ex[1][["VAC", "I_mean", "conc", "COMMENTS", "I_std"]]
        df["conc"] = df["conc"].apply(lambda x: str(x) + " M")
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(
            x="VAC", y="I_mean", data=df, hue="COMMENTS", size="conc"
        )
        IVplot.errorbar(
            df.VAC,
            df.I_mean,
            yerr=df.I_std,
            elinewidth=10,  # width of error bar line
            ecolor="k",  # color of error bar
            capsize=0,  # cap length for error bar
            capthick=1,  # cap thickness for error bar
            fmt="none",
            zorder=50,
            alpha=0.25,
        )
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = ex[0] + "_Ia_VAC_all.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    pattern = r"pinup|thirdpinin"
    frame = results_frame[results_frame["COMMENTS"].str.contains(pattern)]
    for ex in frame.groupby(["sample"]):
        df = ex[1][["VAC", "I_mean", "conc", "COMMENTS", "I_std"]]
        df["conc"] = df["conc"].apply(lambda x: str(x) + " M")
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(
            x="VAC", y="I_mean", data=df, hue="COMMENTS", size="conc"
        )
        IVplot.errorbar(
            df.VAC,
            df.I_mean,
            yerr=df.I_std,
            elinewidth=10,  # width of error bar line
            ecolor="k",  # color of error bar
            capsize=0,  # cap length for error bar
            capthick=1,  # cap thickness for error bar
            fmt="none",
            zorder=50,
            alpha=0.25,
        )
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = ex[0] + "_Ia_VAC_pinup.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    pattern = r"percent"
    frame = results_frame[results_frame["COMMENTS"].str.contains(pattern)]
    for ex in frame.groupby(["sample"]):
        df = ex[1][["VAC", "I_mean", "conc", "COMMENTS", "I_std"]]
        df["conc"] = df["conc"].apply(lambda x: str(x) + " M")
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(
            x="VAC", y="I_mean", data=df, hue="COMMENTS", size="conc"
        )
        IVplot.errorbar(
            df.VAC,
            df.I_mean,
            yerr=df.I_std,
            elinewidth=10,  # width of error bar line
            ecolor="k",  # color of error bar
            capsize=0,  # cap length for error bar
            capthick=1,  # cap thickness for error bar
            fmt="none",
            zorder=50,
            alpha=0.25,
        )
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = ex[0] + "_Ia_VAC_all_gatehold.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "I_gate", "COMMENTS"]]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="I_gate", data=df, hue="COMMENTS")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"$I_{G}$ [pA]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_gateresidual.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "I_gate", "I_mean", "COMMENTS"]]
        df["I_norm"] = df["I_gate"] / df["I_mean"]
        df = df[df["V_g_norm"] <= 0.5]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="I_norm", data=df, hue="COMMENTS")
        IVplot.axhline(1, color="r")
        IVplot.axhline(-1, color="r")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"$I_{G}/I_{A}$ [1]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_normgateresidual.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    # OK, step towards fitted currents.
    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][
            ["V_g_norm", "fit_cur", "fit_time1", "fit_time2", "fit_error", "COMMENTS"]
        ]
        df = df[df["fit_time1"] < 120]
        if not df["fit_time2"].isna().all():
            df = df[df["fit_time2"] < 120]
        r = df.shape[0]
        if r < 5:
            print(r)
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="fit_cur", data=df, hue="COMMENTS")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"Fitted $I_{A}$ [pA]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_fitted.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    df = results_frame[
        ["V_g_norm", "I_gate", "fit_cur", "COMMENTS", "fit_time1"]
    ].dropna()
    plt.figure(figsize=(5, 5))
    df["fit_time_log"] = np.log10(df["fit_time1"])
    IVplot = sns.distplot(df["fit_time_log"])
    plt.xlabel(r"log10 of fit times in sec")
    fileName = "results_fit_time1_hist.png"
    IVplot.figure.savefig(resultsPath / fileName, dpi=600)
    plt.close()

    # at fixed VAC, plot runs of fitted I_mean vs v_g
    for ex in results_frame.groupby(["sample", "conc", "VAC"]):
        df = ex[1][
            ["VAG", "fit_cur", "fit_time1", "fit_time2", "fit_error", "COMMENTS"]
        ]
        df = df[df["fit_time1"] < 120]
        if not df["fit_time2"].isna().all():
            df = df[
                df["fit_time2"] < 120
            ]  # BUG god this is a mess -- has to be a better way here...
        r = df.shape[0]
        if r < 5:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(
            x="VAG", y="fit_cur", data=df, hue="COMMENTS", zorder=2
        )
        plt.xlabel(r"$V_{AG}$ [mV]")
        plt.ylabel(r"Fitted $I_{A}$ [pA]")
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "MKCl_VAC_"
            + str(ex[0][2])
            + "mV_Ia-Vg_fitted.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    # at fixed VAC, plot runs of fitted I_mean vs v_g_norm
    for ex in results_frame.groupby(["sample", "conc", "VAC"]):
        df = ex[1][["V_g_norm", "fit_cur", "fit_time1", "fit_time2", "COMMENTS"]]
        df = df[df["V_g_norm"] <= 0.5]
        df = df[df["fit_time1"] < 120]
        r = df.shape[0]
        if r < 5:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(
            x="V_g_norm", y="fit_cur", data=df, hue="COMMENTS", zorder=2
        )
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"Fitted $I_{A}$ [pA]")
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "MKCl_VAC_"
            + str(ex[0][2])
            + "mV_Ia-Vgnorm_fitted.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "I_gate", "fit_cur", "COMMENTS"]]
        df["I_norm"] = df["I_gate"] / df["fit_cur"]
        df = df[df["V_g_norm"] <= 0.5]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="I_norm", data=df, hue="COMMENTS")
        IVplot.axhline(1, color="r")
        IVplot.axhline(-1, color="r")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"$I_{G}/I_{A}$ [1]")
        fileName = (
            ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_normgateresidualfitted.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "I_gate", "fit_cur", "COMMENTS", "fit_time1"]]
        df = df[df["V_g_norm"] <= 0.5]
        df = df[df["fit_time1"] < 120]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(
            x="V_g_norm",
            y="I_gate",
            data=df,
            hue="COMMENTS",
            label="I_gate",
            marker="+",
        )
        sns.scatterplot(
            x="V_g_norm",
            y="fit_cur",
            data=df,
            hue="COMMENTS",
            ax=IVplot,
            label="fit_cur",
            marker="x",
        )
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"$I_{A} - I_{G}$ [pA]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_gateandanodefitted.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "gate_fit_cur", "fit_cur", "COMMENTS", "fit_time1"]]
        df["I_sub"] = df["fit_cur"] - df["gate_fit_cur"] / 2
        df = df[df["V_g_norm"] <= 0.5]
        df = df[df["fit_time1"] < 120]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="I_sub", data=df, hue="COMMENTS")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"$I_{A} - I_{G}$ [pA]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_gatesubfitted.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][
            ["V_g_norm", "I_gate", "fit_cur", "COMMENTS", "fit_time1", "fit_time2"]
        ]
        df = df[df["fit_time1"] < 200]
        df = df[df["fit_time2"] < 200]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(6, 6))
        IVplot = sns.scatterplot(
            x="V_g_norm", y="fit_time1", data=df, hue="COMMENTS", label="t1", marker="+"
        )
        sns.scatterplot(
            x="V_g_norm",
            y="fit_time2",
            data=df,
            hue="COMMENTS",
            ax=IVplot,
            label="t2",
            marker="x",
        )
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r"fit_time [sec]")
        fileName = ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_fit_times.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "gate_fit_cur", "fit_cur", "COMMENTS"]]
        df["I_norm"] = df["gate_fit_cur"] / df["fit_cur"]
        df = df[df["V_g_norm"] <= 0.5]
        df = df[df["I_norm"] <= 1000]
        df = df[df["I_norm"] >= -1000]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="I_norm", data=df, hue="COMMENTS")
        IVplot.axhline(1, color="r")
        IVplot.axhline(-1, color="r")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r" Fits $I_{G}/I_{A}$ [1]")
        fileName = (
            ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_normgateresidualbothfitted.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "conc"]):
        df = ex[1][["V_g_norm", "gate_fit_cur", "I_mean", "COMMENTS"]]
        df["I_norm"] = df["gate_fit_cur"] / df["I_mean"]
        df = df[df["V_g_norm"] <= 0.5]
        df = df[df["I_norm"] <= 1000]
        df = df[df["I_norm"] >= -1000]
        r = df.shape[0]
        if r < 3:
            continue
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="V_g_norm", y="I_norm", data=df, hue="COMMENTS")
        IVplot.axhline(1, color="r")
        IVplot.axhline(-1, color="r")
        plt.xlabel(r"Normalized $V_{AG}$ [1]")
        plt.ylabel(r" Fits $I_{G}/I_{A}$ [1]")
        fileName = (
            ex[0][0] + "_salt_" + str(ex[0][1]) + "MKCl_normgateresidualInotfitted.png"
        )
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "R", "conc"]]
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAC", y="R", data=df, hue="conc")
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$\frac{V_{AC}}{I_A}$ [M$\Omega$]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_R.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "I_mean", "conc"]]
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAC", y="I_mean", data=df, hue="conc")
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$I_{A}$ [pA]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_IV.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    results_frame["CCHIP"] = (
        results_frame["CCHIP"] * 1e12
    )  # effing pandas flips out with super-small values
    results_frame["RCHIP"] = (
        results_frame["RCHIP"] / 1e6
    )  # now both resistances are in megaohm

    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "CCHIP", "conc"]]
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAC", y="CCHIP", data=df, hue="conc")
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$C$ [pF]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_cap.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "RCHIP", "conc"]]
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        plt.figure(figsize=(8, 8))
        IVplot = sns.scatterplot(x="VAC", y="RCHIP", data=df, hue="conc")
        plt.xlabel(r"$V_{AC}$ [mV]")
        plt.ylabel(r"$R$ [M$\Omega$]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_res.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    del IVplot

    from lmfit import Model

    def exponential(x, A, x0, B):
        return A * np.exp(x / x0) + B

    gmodel = Model(exponential)

    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "I_mean", "conc"]]
        df["I_mean"] = df["I_mean"] / 1000
        df["VAC"] = (
            df["VAC"] / 1000
        )  # prepare for stupidity -- this overflows if it's in pA and millivolts.
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        plt.figure(figsize=(8, 8))
        for thing in df.groupby(["conc"]):
            fit_frame = pd.DataFrame(thing[1])
            fit_frame = fit_frame.sort_values(
                by=["VAC"]
            )  # dude, holy crap -- you have to sort this explicitly
            print(fit_frame)
            if len(fit_frame) > 3:
                result = gmodel.fit(
                    fit_frame["I_mean"], x=fit_frame["VAC"], A=5, x0=0.2, B=-5
                )
                print(thing[0])
                print(result.fit_report())
                print(result.best_fit)
                plt.plot(fit_frame["VAC"], result.best_fit, label=thing[0])
        IVplot = sns.scatterplot(x="VAC", y="I_mean", data=df, hue="conc")
        plt.xlabel(r"$V_{AC}$ [V]")
        plt.ylabel(r"$I_{A}$ [nA]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_IV_fitted.png"
        IVplot.figure.savefig(resultsPath / fileName, dpi=600)
        plt.close()

    for ex in results_frame.groupby(["sample", "config"]):
        df = ex[1][["VAC", "I_mean", "conc"]]
        df["I_mean"] = df["I_mean"] / 1000
        df["VAC"] = df["VAC"] / 1000
        df["conc"] = df["conc"].apply(lambda x: str(x) + " M")
        plt.figure(figsize=(8, 8))
        for thing in df.groupby(["conc"]):
            fit_frame = pd.DataFrame(thing[1])
            fit_frame = fit_frame.sort_values(by=["VAC"])
            if len(fit_frame) > 3:
                result = gmodel.fit(
                    fit_frame["I_mean"], x=fit_frame["VAC"], A=5, x0=0.2, B=-5
                )
                fit_frame["I_mod"] = result.eval(x=fit_frame["VAC"])
                result.plot_fit()
                plt.xlabel(r"$V_{AC}$ [V]")
                plt.ylabel(r"$I_{A}$ [nA]")
                fileName = (
                    ex[0][0] + "_config_" + str(ex[0][1]) + "_conc_" + thing[0] + ".png"
                )
                plt.savefig(resultsPath / fileName, dpi=600)
                plt.close()

    # Save the frame with the addition of R and V_g_norm
    dataframe_name = (
        "dataframe_pandas_atruntime_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".csv"
    )
    df_name = resultsPath / dataframe_name
    results_frame.to_csv(df_name, encoding="utf-8")

    logging.info("Run complete...")
    logging.info("--- runtime was %s seconds ---" % (time.time() - start_time))

    ## Graveyard -- the below do execute OK, but are somehwat usesless..
    # # Id(VAC) colored by salt, with fixed at equivalent vg
    # pattern = r'pinup|percent|thirdpinin'
    # frame = results_frame[results_frame['COMMENTS'].str.contains(pattern)]
    # for ex in frame.groupby(['sample', 'COMMENTS']):
    #     df = ex[1][['VAC', 'I_mean', 'conc']]
    #     df['conc'] = df['conc'].apply(lambda x: str(x)+' M')  # this is annoying -- if you hue by a purely numerical value, sns will rescale stuff.
    #     plt.figure(figsize=(8,8))
    #     IVplot=sns.scatterplot(x='VAC', y='I_mean', data=df, hue='conc')
    #     plt.xlabel(r"$V_{AC}$ [mV]")
    #     plt.ylabel(r"$I_{A}$ [pA]")
    #     fileName = ex[0][0]+'_'+ex[0][1]+'_Ia_VAC.png'
    #     IVplot.figure.savefig(resultsPath / fileName, dpi=600)
    #     plt.close()


#     files_for_model = sorted(resultsPath.rglob('*_model.csv'))

#     for ex in results_frame[results_frame['exp_name'].str.contains('pindown')].groupby(['sample','conc']):
#         df = ex[1][['VAC', 'I_mean', 'I_gate', 'COMMENTS']]
#         ex_conc = str(int(float(ex[0][1])*1000))
#         ex_device = ex[0][0]
#         for name in files_for_model:
#             print(name)
#             print(name.stem)
#             print(name.stem.split('_'))
#             mod_device = name.stem.split('_')[0]
#             mod_config = name.stem.split('_')[2]
#             mod_conc = name.stem.split('_')[4]
#             if mod_device == ex_device and mod_config == 'C' and mod_conc == ex_conc:
#                 print(name)
#                 print(ex[0])
#                 logging.info("Found a model to load.")
# #                the_model = load_model(name, funcdefs={'my_exp': my_exp})
#         r = df.shape[0]
#         if r < 4:
#             continue
#         plt.figure(figsize=(8,8))
#         IVplot2=sns.scatterplot(x='VAC', y='I_mean', hue='COMMENTS', data=df, zorder=2)

#         plt.xlabel(r"$$V_{AC}$")
#         plt.ylabel(r"model-corrected $I_A$")
#         fileName = ex[0][0]+'_salt_'+str(ex[0][1]) + 'MKCl_IV_model.png'
#         IVplot2.figure.savefig(Path.cwd().parent/'ChimeraData/results'/ fileName, dpi=600)
#         plt.close()


# from lmfit.model import Model, save_model, load_model

# gmodel = Model(my_exp)

# for ex in results_frame.groupby(['sample', 'config']):
#     df = ex[1][['VAC', 'I_mean', 'conc']]
#     df['I_mean'] = df['I_mean']/1000
#     df['VAC'] = df['VAC']/1000  #prepare for stupidity -- this overflows if it's in pA and millivolts.
#     df['conc'] = df['conc'].apply(lambda x: str(x)+' M')  #moronic automatic rescaling for hue by float
#     plt.figure(figsize=(8,8))
#     for thing in df.groupby(['conc']):
#         fit_frame = pd.DataFrame(thing[1])
#         fit_frame = fit_frame.sort_values(by=['VAC']) # dude, holy crap -- you have to sort this explicitly
#         print(fit_frame)
#         if len(fit_frame) > 3:
#             result = gmodel.fit(fit_frame['I_mean'], x=fit_frame['VAC'], A=5, x0=.2, B=-5)
#             print(thing[0])
#             print(result.fit_report())
#             print(result.best_fit)
#             plt.plot(fit_frame['VAC'], result.best_fit, label=thing[0])
#     IVplot=sns.scatterplot(x='VAC', y='I_mean', data=df, hue='conc')
#     plt.xlabel(r"$V_{AC}$ [V]")
#     plt.ylabel(r"$I_{A}$ [nA]")
#     fileName = ex[0][0]+'_config_'+str(ex[0][1]) + '_IV_fitted.png'
#     IVplot.figure.savefig(Path.cwd().parent/'ChimeraData/results' / fileName, dpi=600)
#     plt.close()

# del IVplot

# for ex in results_frame.groupby(['sample', 'config']):
#     df = ex[1][['VAC', 'I_mean', 'conc']]
#     df['I_mean'] = df['I_mean']/1000
#     df['VAC'] = df['VAC']/1000
#     df['conc'] = df['conc'].apply(lambda x: str(x)+' M')
#     plt.figure(figsize=(8,8))
#     for thing in df.groupby(['conc']):
#         fit_frame = pd.DataFrame(thing[1])
#         fit_frame = fit_frame.sort_values(by=['VAC'])
#         if len(fit_frame) > 3:
#             result = gmodel.fit(fit_frame['I_mean'], x=fit_frame['VAC'], A=5, x0=.2, B=-5)
#             c = str(int(float(thing[0].split(' ')[0])*1000))
#             save_name = ex[0][0] + '_config_' + str(ex[0][1]) + '_conc_'+c+'_model.csv'
#             save_model(result, Path.cwd().parent/'ChimeraData/results' / save_name)
#             fit_frame['I_mod'] = result.eval(x=fit_frame['VAC'])
#             result.plot_fit()
#             plt.xlabel(r"$V_{AC}$ [V]")
#             plt.ylabel(r"$I_{A}$ [nA]")
#             fileName = ex[0][0] + '_config_' + str(ex[0][1]) + '_conc_'+thing[0]+'.png'
#             plt.savefig(Path.cwd().parent/'ChimeraData/results' / fileName, dpi=600)
#             plt.close()
