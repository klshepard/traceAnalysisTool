#######################################
## author: Jakob Buchheim and Boyan Penkov
## company: Columbia University
##
## version history:
##      * created: 20191001
##      * version 1.1: 20191214 -- integrate with object-based reader
##
## description:
## * based on Matlab fitting routine
## * calculates capacitance of device / chips
## * contains hardcoded values valid for Chimera Instrument V100 in Shepard Lab at Columbia University
## * calibration values and details can be found here: https://docs.google.com/spreadsheets/d/1HOjkFIiZbyhScoSb6g8yU1bGWzJNFNVQyEymDOSfnXs/edit#gid=0
#######################################

import numpy as np
import fft_utils
import gc
import math
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))
from import_utils import get_stem

matplotlib.use("Agg")
logging.basicConfig(
    level=logging.DEBUG, format=" %(asctime)s %(levelname)s - %(message)s"
)


def calculateDeviceParameter(seg, rc, noPlot=False):
    """
    Given a segment and runconstants, this will fit to the canonical four-term polynomial model for the noise PSD.

    Will add the relevant constants in statsDF, maaybe fix this explicity TODO

    Parameters
    ----------
    seg : Segment()
        The segment to fit.
    rc : runConstants
        run constants bject
    noPlot : bool, default is false
        If true, don't produce the plot.

    Returns
    -------
    statsValues : list
        List of things that this addes to statsDF.
    """
    k = 1.38e-23
    T = 273.0 + 20.0
    q = 1.602e-19
    openheadstagenoise = 1.64435e-028 * 1e24  # pA^2 fitted value
    inputCapacitance = 2.68392015288e-11  # F fitted value
    vn2pi = 7.14624966881e-09  # fitted value
    fitCutoff = 500.0e3
    fitDecimator = 50
    maxLen = 2e7
    dataLen = len(seg.IaRAW)
    if dataLen >= maxLen:
        dataLen = np.int(maxLen)
    nperseg = int(np.size(seg.IaRAW[:dataLen]) // seg.rc.npersegPARAM)
    noverlap = nperseg // seg.rc.noverlapPARAM
    f, P = fft_utils.fft_welch(
        seg.IaRAW[:dataLen], seg.samplerateRAW, nperseg, noverlap
    )

    ## Fit polynomial to PSD
    fit_f = f[1 : int(np.floor(np.size(f) * fitCutoff / np.max(f)))]
    inputpsdtimesf = np.multiply(
        P[1 : np.size(fit_f) + 1 : fitDecimator], fit_f[::fitDecimator]
    )  # since 1/f term required multiplying with f to end up having deg3 polynomail f^0+f^1+f^2+f^3
    # fitweights = np.reciprocal(np.multiply(np.square(fit_f),inputpsdtimesf))        # fit weight to match low frequency portion better! (less fitting points)
    # fitweights = np.reciprocal(np.multiply(fit_f, inputpsdtimesf))
    fitweights = np.reciprocal(
        np.multiply(
            P[1 : np.size(fit_f) + 1 : fitDecimator],
            np.power(fit_f[::fitDecimator], 1.5),
        )
    )  # 20191003 best result at this point
    # fitweights = np.reciprocal(np.multiply(fit_f,np.multiply(fit_f,fit_f)))                # 20191007 best result at this point

    PSDNorm = np.linalg.norm(inputpsdtimesf)
    weightNorm = np.linalg.norm(fitweights)
    freqcoeffs = np.polyfit(
        fit_f[::fitDecimator], inputpsdtimesf / PSDNorm, 3, w=fitweights / weightNorm
    )
    freqcoeffs = freqcoeffs * PSDNorm

    # fLog = np.logspace(np.log10(5), np.log10(fitCutoff), num = 2.0e3, base = 10.0, endpoint = True)
    # inputpsdtimesfLog = np.interp(fLog, fit_f, inputpsdtimesf)
    # fitweightsLog = np.power(fLog,-1.5)
    # PSDNormLog = np.linalg.norm(inputpsdtimesfLog)
    # weightNormLog = np.linalg.norm(fitweightsLog)
    # freqcoeffsLog = np.polyfit(fLog, inputpsdtimesfLog / PSDNormLog, 3, w = fitweightsLog / weightNormLog)
    # freqcoeffsLog = freqcoeffsLog * PSDNormLog

    # Calculate pore / chip parameter
    freqcoeffs = np.abs(freqcoeffs)
    squareterm = freqcoeffs[0]
    linearterm = freqcoeffs[1]
    constantterm = freqcoeffs[2]
    flickerterm = freqcoeffs[3]

    # shot noise:  I^2/Hz = 2qI calculate from currentoffset
    # expectedshotnoiseleak = 2*q*abs(currentoffset);
    # expectedshotnoiseIDC = 2*q*abs(Idc);
    # fprintf('Expected shot noise from Idc: %g \n', expectedshotnoiseIDC);

    # thermal noise:    I^2/Hz = 4*k*T/R
    residualthermalnoise = np.sqrt(constantterm ** 2 - openheadstagenoise ** 2) * 1e-24
    R_noisefit = 4.0 * k * T / (residualthermalnoise)
    # fprintf('Residual thermal noise: %g \n', residualthermalnoise)

    R_opennoisefit = 4.0 * k * T / (openheadstagenoise) * 1e-24
    # flicker noise:

    # capacitive noise:  I^2/Hz = vn^2 * (2*pi*f)^2 * C^2
    # C_noisefit = np.sqrt( squareterm / (vn**2 * 4.0 * np.pi**2) )

    C_noisefit = np.sqrt(squareterm * 1e-24 / (vn2pi ** 2)) - inputCapacitance

    ## save fit values to dict
    statValues = {
        "CCHIP": C_noisefit,
        "RCHIP": R_noisefit,
        "I2NOISE-1": flickerterm,
        "I2NOISE0": constantterm,
        "I2NOISE1": linearterm,
        "I2NOISE2": squareterm,
    }

    seg.add_to_statsDF(
        ["CCHIP", "RCHIP", "I2NOISE-1", "I2NOISE0", "I2NOISE1", "I2NOISE2"],
        [C_noisefit, R_noisefit, flickerterm, constantterm, constantterm, squareterm],
    )  # TODO might want to rename these, but careful with downstream breakage.

    logging.info(
        "Slice {s1:}: R: {s2:.03e}Ohm C: {s3:.03e}F".format(
            s1=seg.name, s2=R_noisefit, s3=C_noisefit
        )
    )

    ## Plot PSD and fit function / values
    if not noPlot:
        stem = get_stem(seg)
        plotmegaFileName = stem + "_" + str(seg.VAC) + "mV_csvfit.png"
        savepathname = rc.resultsPath / plotmegaFileName
        finv = np.reciprocal(f[1::10])
        fig = plt.figure(figsize=(8, 6))
        gs = matplotlib.gridspec.GridSpec(1, 1)
        ax2 = plt.subplot(gs[0, :])
        ax2.loglog(f, P, "-", color="blue", linewidth=0.5)
        ax2.loglog(
            f[1::10],
            np.multiply(np.polyval(freqcoeffs, f[1::10]), finv),
            "--",
            color="black",
            linewidth=2.0,
        )
        ax2.plot(
            f[1::10],
            np.multiply(np.polyval([freqcoeffs[0], 0, 0, 0], f[1::10]), finv),
            "-r",
        )
        ax2.plot(
            f[1::10],
            np.multiply(np.polyval([0, freqcoeffs[1], 0, 0], f[1::10]), finv),
            "-r",
        )
        ax2.plot(
            f[1::10],
            np.multiply(np.polyval([0, 0, freqcoeffs[2], 0], f[1::10]), finv),
            "-r",
        )
        ax2.plot(
            f[1::10],
            np.multiply(np.polyval([0, 0, 0, freqcoeffs[3]], f[1::10]), finv),
            "-r",
        )
        #        ax2.set_ylim(1e-4, 1e2)
        #        ax2.set_xlim(5e0, 5e6)
        # TODO no, these only make sense for nanopore data, so let mpl figure out the bounds
        ax2.set_xlabel(r"Frequency [$Hz$]")
        ax2.set_ylabel(r"PSD [$pA^2/Hz$]")
        ax2.grid(True, which="major", ls="-", color="0.65")
        C_pF = C_noisefit * 1e12
        R_MOhm = R_noisefit * 1e-6
        ax2.annotate(
            "C: {s1:.2f} pF, R: {s2:.2f} MOhm".format(s1=C_pF, s2=R_MOhm),
            xy=(0.05, 0.05),
            xytext=(0.05, 0.05),
            textcoords="figure fraction",
            xycoords="figure fraction",
        )
        fig.savefig(savepathname, dpi=600)

        fig.clear()
        plt.close(fig)

    return statValues
