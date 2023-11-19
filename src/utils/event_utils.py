#######################################
## author: Boyan Penkov, Jakob Buchheim
## company: Columbia University
#######################################

import numpy as np
import logging
from numba import njit, float64, int64, typeof
import copy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
from sklearn.mixture import GaussianMixture
import setENV_utils
import math
from textwrap import wrap
from scipy.special import gamma
from scipy.stats import t
from scipy.ndimage import uniform_filter1d
from lmfit.models import ExponentialModel
from lmfit.models import PowerLawModel
from lmfit import Model
import pybaselines

from pathlib import Path

sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src"))

import export_utils
import filter_utils
import plot_utils
from import_utils import get_stem

# plt.rc('text', usetex=True)
matplotlib.use("Agg")
plt.rcParams["agg.path.chunksize"] = 20000
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 0.5
plt.rcParams["axes.titlesize"] = 10

logging.basicConfig(
    level=logging.DEBUG, format=" %(asctime)s %(levelname)s - %(message)s"
)


def moving_average(a, window):
    """
    Calculates backward looking moving average.

    Parameters
    ----------
    a : nparray
        The data to take the average over.
    window : int
    the window size.

    Returns
    -------
    out : np.float32()
        array of length a with moving average in backward direction out[i]=mean(a[i-window,i]).  Considers beginning of a and implements smaller averaging windows over the first WINDOW points
    """
    return uniform_filter1d(
        a, size=window, output=np.float32, mode="mirror", origin=((window - 1) // 2)
    )

def moving_average_fw(a, window):
    """
    caculates forward looking moving average

    Parameters
    ----------
    a : np.array
    data to be averaged
        window: int
    window length for monving average

    Returns
    -------
    out : nparray
        array of length a with moving average in forward direction out[i]=mean(a[i,i+window]), considers end of a and implemts smaller averaging windows
    """
    return uniform_filter1d(
        a, size=window, output=np.float32, mode="mirror", origin=-(window // 2)
    )


def min_std(a, baseline, noEventsIndex, maxLikelyStd, gmmStd, seg):
    """
    default function to calculating standard deviation for each pass. Returns minimum of baseline correct std and once!!! calculated std distribution

    Parameters
    ----------
    a : np.array of data for which to calculate the standard deviation
    baseline : moving average value of baseline
    noEventsIndex : np.int array of length(a) with 1 where there is no event recognized
    maxLikelyStd : standard deviation calculated for each segment
    gmmStd : standard deviation calculated by gaussian mixture model fit to cumulative trace amplitudes
    seg : segment data for logging output

    Returns
    -------
    Returns min(std(data_runner),GMMstd) minimum of baseline correct standard deviation and gaussian mixture model standard deviation,
    EXCEPTION: 
        min(std(data_runner),GMMstd) < 1e-1 (too small - indiation of empty slice or saturated current). 
        To prevent finding so many false postive events because std is too small std is increased to 1
    """

    # Standard deviation based on corrected baseline
    try:
        std = baselinecorrected_std(a, baseline, np.array(noEventsIndex, dtype=np.int64))
    except:
        logging.info(
                "baselinecorrected STD calculation failed of experiment {s0}_{s1:.0f}mV".format(
                    s0=seg.name, s1=seg.VAC
                )
            )
        std = seg.cur_std

    fixedStd = std
    if not math.isnan(seg.exp.rc.args.eventThreshold):
        fixedStd = seg.exp.rc.args.eventThreshold / seg.exp.rc.args.PDF

    # return lower value of both methods
    stdValues = [std, maxLikelyStd, gmmStd, fixedStd]
    stdValues = np.sort(stdValues)
    for i in stdValues:
        if i > 0.8:
            return i
        else:
            return 0.8
            logging.info(
                "Standard deviation of experiment {s0}_{s1:.0f}mV  low: std = {s2}, set to 1".format(
                    s0=seg.name, s1=seg.VAC, s2=minStd
                )
            )


@njit(cache=True)
def baselinecorrected_std(dataSet, baseline, noEventsIndex):
    """
    Standard deviation with correction for baselline drift and outside event only

    Parameters
    ----------
    dataSet : np.array of data for which to calculate the standard deviation
    baseline : moving average value of baseline
    noEventsIndex : np.int array of length(a) with 1 where there is no event recognized

    Returns
    -------
    Returns minimum of baseline correct standard deviation
    """
    return np.std(dataSet[noEventsIndex] - baseline[noEventsIndex])


# @njit(cache=True) Storing Exceptions fails in nopython mode for jit
def max_likely_std(dataSet, window):
    """
    Standard deviation based on segments of data with histogram max bins

    Parameters
    ----------
    dataSet : np.array of data for which to calculate the standard deviation
    window: int of window length for monving average


    Returns
    -------
    most often occuring standard deviation for dataSet sliced up into windows
    """
    try:
        seg_deviations = segments_std(dataSet, window)
        hist, bins = np.histogram(seg_deviations, bins=40)
        maxIdx = np.argmax(hist)
        return (bins[maxIdx] + bins[maxIdx + 1]) * 0.5
    except Exception as e:
        return 1.0


def gaussian_classifier(dataSet, seg, rc, allowedEventType):

    """
    calculates baseline value and standard deviation of data based on gaussian mixture model fit, fits 3 gaussian peaks - usually works nicely
    Problem when data in saturation, no Histogram fit possible if single value.

    Assumes minimum bin size of 5.0

    Output: plot of histogram with peak fitting

    Parameters
    ----------
    dataSet : np.array of data for which to calculate the standard deviation
    seg : segment class data
    rc : run constant class data
    allowedEventType : str
        passes allowed event type to be able to return the correct baseline value

    Returns
    -------
    baseline_avg : baseline average value: for cases looking for up and down events: highest histogram peak; for cases looking for up events: lowest current peak; for cases looking for down events: highest current peak;
    baseline_std : standard deviation of baseline gaussian peak
    fittedHist : fitted gaussian mixture object
    """

    name = seg.dfIndex() + "_Histogram.png"
    writeFile = rc.resultsPath / "events" / name

    # make histogram over full trace
    try:
        numberOfBins = int(np.abs(np.max(dataSet) - np.min(dataSet)) / 5.0)
    except ValueError:  # some crap if it's a straight line here.
        numberOfBins = 40
    if numberOfBins < 3:
        return np.max(dataSet), 1000, np.nan
    elif numberOfBins <= 40:
        numberOfBins = 40
        # logging.info("issues with bin size - too small set to 20")

    rdmSize = np.int(5e5)
    if rdmSize > len(dataSet):
        rdmSize = np.int(len(dataSet))

    smallDataSet = np.random.choice(dataSet, rdmSize, replace=False)

    hist, bins = np.histogram(smallDataSet, bins=numberOfBins, density=True)

    # initialize gaussian mixture model, use to many to find all peaks
    numberOfGaussians = 2  # for PCP data with high concentration 5 is correct
    if allowedEventType == "both":
        numberOfGaussians = 3

    fittedHist = GaussianMixture(
        numberOfGaussians, covariance_type="spherical", tol=1e-6, max_iter=500
    )

    try:
        # fit the data to the model
        fittedHist.fit(X=smallDataSet.reshape(-1, 1))

        # create dataframe from fitting parameter to better handle the fitted stuff
        statsData = pd.DataFrame(
            data=list(
                zip(
                    np.array(fittedHist.means_).flatten(),
                    np.sqrt(fittedHist.covariances_.flatten()),
                    np.exp(fittedHist.score_samples(fittedHist.means_)),
                    fittedHist.predict(fittedHist.means_),
                )
            ),
            columns=["MEAN", "STANDARDDEVIATION", "P", "LABEL"],
        )
        # statsData.sort_values(by = 'P', ascending = False, inplace = True)
        logging.info(
            "Fitted Gaussian Mixture Model for current trace of experiment {s0}_{s1:.0f}mV \t Results: \n {s2}".format(
                s0=seg.name, s1=seg.VAC, s2=statsData
            )
        )

        # reference plot for comparison
        fig2, ax = plt.subplots()
        ax.grid(True, which="both", ls="-", color="0.65")
        ax.set_xlabel(r"$I_{A}$ [$pA$]")
        ax.set_ylabel(r"$probability$ ")
        ax.set_title("\n".join(wrap(name)))

        ax.bar((bins[:-1] + bins[1:]) / 2, hist, (bins[0] - bins[1]) * 0.8)
        fittedHist_y = np.exp(fittedHist.score_samples(bins.reshape(-1, 1)))
        ax.plot(bins, fittedHist_y, "-r")

        responsibilities = fittedHist.predict_proba(bins.reshape(-1, 1))
        fittedHist_y_individual = responsibilities * fittedHist_y[:, np.newaxis]
        ax.plot(bins, fittedHist_y_individual, "--r")

        fig2.savefig(writeFile, dpi=100, format="png")
        fig2.clear()
        plt.close(fig2)

        # assembling function output (return values for highest peak)
        outputIdx = 0
        if allowedEventType == "up" and not statsData[statsData.P > 0.02].MEAN.empty:
            outputIdx = statsData[
                statsData.P > 0.02
            ].MEAN.idxmin()  # largest current (baseline) value if probability is not too low
        elif (
            allowedEventType == "down" and not statsData[statsData.P > 0.02].MEAN.empty
        ):
            outputIdx = statsData[statsData.P > 0.02].MEAN.idxmax()
        else:
            outputIdx = statsData.P.idxmax()  # highest peak value

        baseline_avg = statsData.iloc[outputIdx].MEAN
        baseline_std = statsData.iloc[outputIdx].STANDARDDEVIATION
        del hist, bins, statsData, fig2, ax, fittedHist_y

    except Exception as e:
        logging.warning(
            "No histogram gaussion mixture model fit possible for {s0},\n\t\t running without --HistBaseline on this slice".format(
                s0=seg.name
            )
        )
        baseline_avg = np.NAN
        baseline_std = (np.NAN,)
        fittedHist = None
        pass

    return baseline_avg, baseline_std, fittedHist


def get_allowed_event(seg, rc):
    """
    gives allowed event type based on cmd line input and mean current level

    Parameters
    ----------
    seg : np.array of data to to calculate the standard deviation
    rc: int of window length for monving average

    Returns
    -------
    allowedEventType : str
        'up', 'down' or 'both'

    """
    allowedEventType = rc.args.eventType
    if seg.cur_mean < -10 and allowedEventType == "blockage":
        allowedEventType = "up"
    elif seg.cur_mean < -10 and allowedEventType == "enhance":
        allowedEventType = "down"
    elif (seg.cur_mean > 10) and allowedEventType == "blockage":
        allowedEventType = "down"
    elif (seg.cur_mean > 10) and allowedEventType == "enhance":
        allowedEventType = "up"
    elif (seg.cur_mean < 10) and (seg.cur_mean > -10):
        allowedEventType = "both"
    else:
        allowedEventType = "both"

    return allowedEventType


@njit(cache=True)
def segments_std(dataSet, window):
    """
    Calculates std of data set for each window section

    Parameters
    ----------
    dataSet : np.array of data to to calculate the standard deviation
    window: int of window length for monving average

    Returns
    -------
    seg : array of length dataSet - window with containing the standard deviation of each window
    """
    seg = np.empty(len(dataSet[:: np.int(np.ceil(window))]), np.uint16)
    for i, x in np.ndenumerate(seg):
        seg[i] = np.std(
            dataSet[
                np.int(np.ceil(window) * i[0]) : np.int(
                    np.ceil(window) * (i[0] + 1)
                )
            ]
        )
    return seg


def find_events(seg, rc):
    """
    Implemenation of a multipass baseline-correcting deviation eventfinder, following Plesa and Dekker's 2015 paper.

    We select a constant WINDOW, and begin by calculating -- at each point of the array of current values -- the backwards-looking moving average over a subarray of length WINDOW (this implied we drop the first WINDOW points).

    We then begin PASSCOUNT loops over the entire current array.  An event entrance is a single deviation by more than PDF standard deviations from the baseline.  An event exit is a recursion to the mean closer than PDF_reversal standard deviations.  While in an event, the deviation is not updated.  For each loop less than the final loop, we update any candidate event points with the calculated mean of the array at the event entry.  This is used to avoid baseline drift in the event detector code; we can see the effects here by the variation in baseline we see in the outputted plot of baselines, especially for long events.

    Per point, the state machine tracks transitions from no event to up event or down event, from down event to exit and from up event to exit.  Critically, it does not track the comparatively rare case of a transition from an active up event to a down event, which is a step of about PDF+PDF_reversal sigma (which, for 6 sigma at 1 MS/sec is will occur once per 500 seconds, which is a long file but not extraordinarily long.  At 10 Mhz (CNP4?), this will happen once a minute, which will be a pain...)

    Assumes to start outside of event, but looks for event from data[1] onwards.

    We reject events of length shorter than minLengthSec which is 4x the filtercutoff frequency.

    We reject events which are too shallow mineventdepth.

    We reject events which have a alpha error larger alphaError

    We calculate event parameter (see output)

    We fit event rate constants with single and double exponential decay. SHOULD BE UPDATED to Weibull

    We do not check for events if the number of unique values in seg.Ia is smaller than 10. This prevents failures in the histogram for standard deviation, decimation errors in plotting a gmm failure. Further it saves computational time on meaningless data.

    Outputs

    PLOTS:
    baseline plot containing: raw data of found events, baselines of all iterative passes, decimated data of entire trace, shaded region of event threshold, event threshold value
    event plot: if events found: scatter plot with blockage current vs. dwell time
    undecimated raw data trace plot depending on : rc.args.plotundecimated

    DATA:

    writes event .csv file containing columns=['EVENTTHRESHOLD', 'EVENTTRIGGERVALUE', 'EVENTTYPE', 'EVENTSTARTIDX', 'EVENTSTOPIDX', 'EVENTBASELINE', 'EVENTDWELLTIME', 'EVENTDEPTH', 'EVENTSTD', 'EVENTPVALUE', 'EVENTTVALUE']) for each event and  general information about the event finding / full trace: 'STRIDE', 'MINLENGTH', 'PDF', 'PDFREVERSAL', 'FILTERCUTOFF', 'SAMPLERATE', 'BASELINESTD', 'AVEVENTSTD', 'TOTALUPEVENTTIME', 'TOTALDOWNEVENTTIME', 'TOTALTRACETIME', 'P_UP', 'P_DOWN', 'DETREND', 'STOPBAND', 'NOTCH', 'NUMBEROFEVENTS', 'gatevoltage', 'salt', 'conc', 'device'

    populates seg.statsDF with aggregated event data: 'STRIDE',  'MINLENGTH', 'PDF', 'PDFREVERSAL', 'FILTERCUTOFF', 'SAMPLERATE', 'BASELINESTD', 'AVEVENTSTD',  'TOTALUPEVENTTIME', 'TOTALDOWNEVENTTIME',  'TOTALTRACETIME', 'P_UP', 'P_DOWN', 'DETREND', 'STOPBAND', 'NOTCH', 'NUMBEROFEVENTS' and K_OFF_single, K_ON_single, K_OFF_double_slow, K_OFF_double_fast, K_OFF_ERR_single, K_OFF_ERR_double, K_ON_double_slow, K_ON_double_fast

    Options:

    eventype: Looking for 'blockage' or 'enhancement' or 'both' event types. This option works for both scenarios current being larger or smaller than zero: if baseline > 0: current enhanceing event is "UP", if baseline < 0: current enhancing event is "DOWN", there is a dead band for the baseline of 10UNIT around 0, if baseline within this band it will always look for UP and DOWN events.

    HistBaseline: initialized the baseline for the first pass of event finding with a current histogram peak value based on the event type looked for. For subsequent passes baseline and standard deivation are updated according to standard algorithm. This option should be used only in combination with detrend or for non drifting data sets.

    detrend: substracts strong lowpass filtered data from seg.data to supress baseline drift and fluctuations, lowpass timescale is set by rc.args.detrend parameter

    passes: number of passes of iterative event finding. Event finding terminates after 3 passes when no event was found. Additional runs do not yield any update in the baseline if no event was found so they are just overhead

    PDF: threshold parameter multiple of standard deviation for event detection event start level

    PDFreversal: threshold parameter multiple of standard deviation for event detection event stop level

    mineventdepth: minimum average event depth considered

    alphaError: allowable alpha error for t-test on event detection (posterior to finding the event)

    Parameters
    ----------
    seg : np.array
        data to to calculate the standard deviation
    rc: int
        window length for moving average

    Returns
    -------
    events corrected baseline (if events are found)
    """

    # terminate event finder right away if no data

    setENV_utils.setENV_threads(int(1))

    # Don't run if it's basically flat TODO
    earlyAbort = False
    if (type(seg.Ia) == np.int16) and (len(np.unique(seg.Ia)) < 10):
        earlyAbort = True
    elif (type(seg.Ia) == np.float32) and (
        len(np.unique(np.around(seg.Ia, decimals=3))) < 10
    ):
        earlyAbort = True

    if earlyAbort:
        logging.info(
            "No event finding because of railing slice of experiment {s0}_{s1:.0f}mV \t".format(
                s0=seg.name, s1=seg.VAC
            )
        )
        return seg.Ia

    # Read pass over arguments
    # Define event finder sensitivity
    PDF = rc.args.PDF  # How many sigma to enter
    PDF_reversal = rc.args.PDFreversal
    assert PDF >= PDF_reversal
    passcount = rc.args.passes
    minLengthSec = 0.75 / rc.args.filter
    if not math.isnan(rc.args.mineventlength):
        if minLengthSec < rc.args.mineventlength:
            minLengthSec = rc.args.mineventlength

    minLength = int(np.floor(seg.samplerate * minLengthSec))
    logging.info(
        "{s0}_{s1}mV min event length in points is is {s2}".format(
            s0=seg.name, s1=seg.VAC, s2=minLength
        )
    )
    stride = int(np.floor(seg.samplerate * rc.args.avwindow))
    dataSetSize = len(seg.Ia)
    allowedEventType = get_allowed_event(seg, rc)

    ## Plotting baseline - Setup
    # By how much are we decimating for this plot?
    plotFileName = seg.dfIndex()
    writeFile = rc.resultsPath / "events" / plotFileName

    decer = 50
    if dataSetSize > seg.samplerate * 60.0:
        decer = dataSetSize // (seg.samplerate * 60.0 / decer)

    # fig_longtrace = plt.figure()
    fig_longtrace, ax0_longtrace = plt.subplots(figsize=(30, 6))
    ax0_longtrace.grid(True, which="both", ls="-", color="0.65")
    ax0_longtrace.set_ylabel(r"$I_{A}$ [$pA$]")
    ax0_longtrace.set_xlabel(r"$T$ [$s$]")
    ax0_longtrace.set_title(str(writeFile.stem))

    ## event finding
    # initialize variables:
    inEvent = False  # BUG? This is an assumption -- justified, but still....
    eventType = ""
    eventCounter = 0
    omittedEventCounter = 0
    noEventsIndex = np.array([i for i in range(dataSetSize)], dtype=np.uint64)

    # variables for loop
    # TODO The clean way to do this is to have some eventfinder function that handles the low-level algo, and just wraps it around to pass to a Slice()
    dataSet = seg.Ia

    if not math.isnan(rc.args.detrend):
        tmp = filter_utils.detrend_data(
            dataSet, int(np.floor(seg.samplerate * rc.args.detrend))
        )
        dataSet = dataSet - tmp + seg.cur_mean
        del tmp
        logging.info(
            "Event finding in {s0}_{s1}mV: Detrend data with strong lowpass filter".format(
                s0=seg.name, s1=seg.VAC
            )
        )

    dataRunner = np.array(dataSet, copy=True)
    dataSetMinLengthAv = moving_average_fw(dataSet, int(minLength))
    dataSetMinLengthAvEoE = dataSetMinLengthAv
    if rc.args.endOfEventParameter > 1.0:
        del dataSetMinLengthAvEoE
        dataSetMinLengthAvEoE = moving_average_fw(dataSet, int(minLength * rc.args.endOfEventParameter))

    maxLikelyStd = max_likely_std(dataRunner, stride)
    avg = moving_average(dataRunner, stride)
    avg[0:stride] = np.mean(dataRunner)
    idealTrace = np.array(avg, copy=True)
    compressedTrace = [[0, avg[0]]]

    if rc.args.HistBaseline:
        gmmBaseline, gmmStd, gmm = gaussian_classifier(
            dataSet, seg, rc, allowedEventType
        )
        if not np.isnan(gmmBaseline):
            avg = np.ones_like(dataSet) * gmmBaseline
    else:
        gmmBaseline = np.NAN
        gmmStd = np.NAN
        gmm = None

    sigmaValue = min_std(
        dataRunner, avg, noEventsIndex, maxLikelyStd, gmmStd=gmmStd, seg=seg
    )

    ## iterative loop for event finding
    for passes in range(passcount):

        logging.info(
            "Event finding in {s0}_{s1}mV: starting pass {s2:d}".format(
                s0=seg.name, s1=seg.VAC, s2=passes + 1
            )
        )

        # Update average and standard deviation for current pass
        if passes != 0:
            # all other passes average is updated with moving average incorporating the events found
            if rc.args.HistBaseline and passes <= 2:
                avg = (avg + moving_average(dataRunner, stride)) * 0.5
            elif passes < 4:
                avg = moving_average(dataRunner, stride)
                avg[0:stride] = np.mean(dataRunner)
            else:
                avg = moving_average(dataRunner, stride)

            sigmaValue = min_std(
                dataRunner, avg, noEventsIndex, maxLikelyStd, sigmaValue, seg=seg
            )

        # if passes > 2:
        #     PDF = PDFtrue

        logging.info(
            "Event finding in {s0}_{s1}mV: DONE - updating baseline and std".format(
                s0=seg.name, s1=seg.VAC
            )
        )

        # plot stuff
        ax0_longtrace = plot_utils.plot_pass(
            ax0_longtrace,
            passes,
            avg,
            sigmaValue,
            passcount,
            seg.samplerate,
            decer,
            PDF,
            seg,
        )

        if (passes == passcount - 1) or ((passes >= 2) and (eventCounter == 0)):

            ax0_longtrace = plot_utils.plot_endpass(
                ax0_longtrace, passes, dataSet, dataRunner, seg.samplerate, decer
            )
            if not math.isnan(rc.args.detrend):
                ax0_longtrace = plot_utils.plot_detrend(
                    ax0_longtrace, seg, decer, stride, rc
                )
            # Check for early abort when baseline has not changed = no event has been found.
            if (passes >= 2) and (eventCounter == 0):
                logging.info(
                    "No event found in {s0}_{s1}mV: in pass: {s2:d} / {s3:d}; terminate finding".format(
                        s0=seg.name, s1=seg.VAC, s2=passes + 1, s3=passcount
                    )
                )
                ax0_longtrace.annotate(
                    "Pass: "
                    + r"${0:d}: No event found. Baseline not changed, no event will be found. Check baseline tracking, stride and PDF$".format(
                        passes + 1
                    ),
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                )
                idealTrace = np.array(avg, copy=True)
                break

        upEvent = (dataSet > (avg + (PDF * sigmaValue))) & (dataSetMinLengthAv > (avg + (PDF * sigmaValue)))
        downEvent = (dataSet < (avg - (PDF * sigmaValue))) & (dataSetMinLengthAv < (avg - (PDF * sigmaValue)))
        endUpEvent = (dataSet < avg + (PDF_reversal * sigmaValue)) & (dataSetMinLengthAvEoE < (avg + (PDF_reversal * sigmaValue)))
        endDownEvent = (dataSet > avg - (PDF_reversal * sigmaValue)) & (dataSetMinLengthAvEoE > (avg - (PDF_reversal * sigmaValue)))
        currEventStart = 0
        currEventStop = 0
        inUpEventFlag = False
        inDownEventFlag = False
        logEventFlag = False
        lenEvent = 0.0
        eventMean = 0.0
        aEventType = np.int8(0)
        if allowedEventType == 'down':
            aEventType = -1
        elif allowedEventType == 'up':
            aEventType = 1
        eventType = np.int8(0)
        eventCounter = np.uint32(0)
        eventLog = []
        dataRunner = np.array(dataSet, copy=True)
        noEventsIndex = np.array([i for i in range(dataSetSize)], dtype=np.uint64)
        noEvent = np.array([False for i in range(dataSetSize)])

        for i in range(dataSetSize):
            if not inUpEventFlag and not inDownEventFlag:
                if upEvent[i]:
                    if not inUpEventFlag:
                        inUpEventFlag = True
                        currEventStart = i
                        eventType = 1
                elif downEvent[i]:
                    if not inDownEventFlag:
                        inDownEventFlag = True
                        currEventStart = i
                        eventType = -1
                else:
                    pass
                    noEvent[i] = True

            elif inUpEventFlag:
                if endUpEvent[i] == True:
                    inUpEventFlag = False
                    currEventStop = i - 1
                    logEventFlag = True
                elif i >= dataSetSize - 1:
                    currEventStop = i - 1
                    logEventFlag = True

            elif inDownEventFlag:
                if endDownEvent[i] == True:
                    inDownEventFlag = False
                    currEventStop = i - 1
                    logEventFlag = True
                elif i >= dataSetSize - 1:
                    currEventStop = i - 1
                    logEventFlag = True
            
            if logEventFlag:
                lenEvent = (currEventStop - currEventStart) / seg.samplerate
                if lenEvent > minLengthSec:
                    pValue = -1.0
                    eventMean = np.mean(dataSet[currEventStart : currEventStop])
                    if rc.args.alphaError > 0.0:
                        dof = np.ceil(
                            lenEvent * rc.args.filter
                        )  # using real sample values not oversample as sqrt(n)
                        tStat = (
                            (eventMean - avg[currEventStart]) / sigmaValue * np.sqrt(dof)
                        )  # assuming equal variance in event and on baseline
                        pValue = (t.sf(abs(tStat), dof)) * 2.0
                    if (
                        ((aEventType == 0) or (eventType == aEventType))
                        and ( np.abs(eventMean - avg[currEventStart]) > rc.args.mineventdepth)
                        and (pValue < rc.args.alphaError)
                    ):
                        eventCounter += 1
                        noEvent[currEventStart : currEventStop] = False
                        if (passes == passcount - 1):
                            eventLog.append([currEventStart, currEventStop, eventType])
                            if rc.args.interpolateBaseline:
                                dataRunner[currEventStart : currEventStop] = dataSet[currEventStart : currEventStop] - eventMean + avg[currEventStart]
                            else:
                                dataRunner[currEventStart : currEventStop] = (
                                    np.random.normal(
                                        avg[currEventStart],
                                        sigmaValue, 
                                        currEventStop - currEventStart
                                    )
                                )
                        else:
                            length = currEventStop - currEventStart
                            if rc.args.interpolateBaseline & (length > 3 * stride) & (eventCounter != 1):
                                x = np.arange(0,length,1)
                                A = np.vstack([x[stride : -stride], np.ones(len(x[stride : -stride ]))]).T
                                m, c = np.linalg.lstsq(A, dataSet[currEventStart + stride : currEventStop - stride], rcond=None)[0]
                                dataRunner[currEventStart : currEventStop] = avg[currEventStart] + m * (x - length//2)
                            else:
                                dataRunner[currEventStart : currEventStop] = avg[currEventStart]

                currEventStart = i
                currEventStop = i
                logEventFlag = False
                inDownEventFlag = False
                inUpEventFlag = False

        noEventsIndex = noEventsIndex[noEvent]

    idealTrace = np.array(avg, copy = True)
    eventLog = np.array(eventLog)
    # Write event logs to dataframe
    if eventCounter > 0:
        logging.info(
            "Found and wrote {s2} events in {s0}_{s1}mV while looking for event type: {s3} (omitted events = {s4})".format(
                s0=seg.name,
                s1=seg.VAC,
                s2=eventCounter,
                s3=allowedEventType,
                s4=omittedEventCounter,
            )
        )
        eventStartIdx = eventLog[:,0]
        eventStopIdx = eventLog[:,1]
        eventTypeList = ['up' if x == 1 else 'down' for x in eventLog[:,2]]
        eventDwellTime = [np.float32(j-i) / seg.samplerate * 1e6 for i, j in eventLog[:,0:2]]
        eventTriggerValue = dataSet[eventStartIdx]
        eventThreshold = [PDF * sigmaValue] * eventCounter
        eventBaseline = avg[eventStartIdx]
        eventDepth = [np.mean(dataSet[i:j]) for i, j in eventLog[:,0:2]]
        eventDepth -= eventBaseline
        eventMaxDepth = [np.max(dataSet[i:j]) if up == 1 else np.min(dataSet[i:j]) for i, j, up in eventLog[:,0:3]]
        eventMaxDepth -= eventBaseline
        eventStd = [np.std(dataSet[i:j]) for i, j in eventLog[:,0:2]]
        eventDof = [np.ceil(dt * 1e-6 * rc.args.filter) for dt in eventDwellTime]
        eventTValue = eventDepth * np.sqrt(np.array(eventDof)) / sigmaValue
        eventPValue = [(t.sf(abs(tStat), dof)) * 2.0 for i, (tStat, dof) in enumerate(zip(eventTValue, eventDof))]
        compressedTrace = np.transpose(np.array([[i - 1 for i in eventStartIdx] ,eventBaseline, eventStartIdx, eventDepth + eventBaseline, eventStopIdx, eventDepth + eventBaseline, [i + 1 for i in eventStopIdx], avg[eventStopIdx]]))
        compressedTrace = compressedTrace.reshape(eventCounter * 4, 2)
        compressedTrace = np.insert(compressedTrace, 0, [0,avg[0]], axis = 0)
        compressedTrace = np.append(compressedTrace, [[len(avg),avg[-1]]], axis = 0)
        k=0
        for idxm, (i,j) in enumerate(zip(eventStartIdx, eventStopIdx)):
            idealTrace[i:j] = eventDepth[k] + eventBaseline[k]
            k+=1

        # save event to data frame
        allEventLog = pd.DataFrame(
            data=list(
                zip(
                    list(eventThreshold),
                    list(eventTriggerValue),
                    list(eventTypeList),
                    list(eventStartIdx),
                    list(eventStopIdx),
                    list(eventBaseline),
                    list(eventDwellTime),
                    list(eventDepth),
                    list(eventStd),
                    list(eventPValue),
                    list(eventTValue),
                    list(eventMaxDepth),
                )
            ),
            columns=[
                "EVENTTHRESHOLD",  # How many standard deviations forces a state change.
                "EVENTTRIGGERVALUE",  # The value (amplitude) that triggered the entry
                "EVENTTYPE",  # UP or DOWN
                "EVENTSTARTIDX",  # The first point of the event
                "EVENTSTOPIDX",  # The last point of the event
                "EVENTBASELINE",  # The baseline value of the event.
                "EVENTDWELLTIME",  # The duration of the event, in microseconds.
                "EVENTDEPTH",  # The depth of the event -- average minus baseline
                "EVENTSTD",  # np.std() *inside* the event
                "EVENTPVALUE",  # Student's t-test probability of error.
                "EVENTTVALUE",  # Student's t-test test statistic.
                "EVENTMAXDEPTH",  # Maximum depth of event -- peak minus baseline
            ],
        )

        # So the game to play here is that stuff that's written to *EVERY* entry in the eventlog goes to alleventlog...  this is confusing.  TODO should check if this writes different stuff to the csv and the event logs.
        allEventLog["NUMBEROFEVENTS"] = eventCounter
        allEventLog["TOTALUPEVENTTIME"] = allEventLog[
            allEventLog.EVENTTYPE == "up"
        ].EVENTDWELLTIME.sum()

        allEventLog["TOTALDOWNEVENTTIME"] = allEventLog[
            allEventLog.EVENTTYPE == "down"
        ].EVENTDWELLTIME.sum()

        allEventLog["TOTALTRACETIME"] = (
            np.size(dataSet) / seg.samplerate * 1000000
        )  # in microseconds? TODO

        allEventLog["P_UP"] = allEventLog.TOTALUPEVENTTIME / allEventLog.TOTALTRACETIME
        allEventLog["P_DOWN"] = (
            allEventLog.TOTALDOWNEVENTTIME / allEventLog.TOTALTRACETIME
        )

        allEventLog["BASELINESTD"] = sigmaValue
        allEventLog["AVEVENTSTD"] = allEventLog.EVENTSTD.mean()
        allEventLog["VAG"] = np.mean(
            seg.VAG
        )  # So this is the average gate voltage, when we have the vg as a time series.
        allEventLog["VAC"] = seg.VAC
        allEventLog["salt"] = seg.salt
        allEventLog["conc"] = seg.conc
        allEventLog["device"] = seg.sample
        allEventLog["STRIDE"] = stride // seg.samplerate
        allEventLog["MINLENGTH"] = minLengthSec * 1e6
        allEventLog["PDF"] = PDF
        allEventLog["PDFREVERSAL"] = PDF_reversal
        allEventLog["FILTERCUTOFF"] = rc.args.filter
        allEventLog["SAMPLERATE"] = seg.samplerate
        allEventLog["DETREND"] = rc.args.detrend
        allEventLog["STOPBAND"] = rc.args.stopBand
        allEventLog["NOTCH"] = rc.args.notch

        allEventLog.to_csv(
            str(writeFile) + "_events.csv", sep=",", encoding="utf-8", index=True
        )

        # write aggregated event stats to statsDF
        statsList = [
            "NUMBEROFEVENTS",
            "TOTALTRACETIME",
            "TOTALUPEVENTTIME",
            "TOTALDOWNEVENTTIME",
            "P_UP",
            "P_DOWN",
            "AVEVENTSTD",
            "BASELINESTD",
            "STRIDE",
            "MINLENGTH",
            "PDF",
            "PDFREVERSAL",
            "FILTERCUTOFF",
            "SAMPLERATE",
            "DETREND",
            "STOPBAND",
            "NOTCH",
        ]
        seg.join_statsDF(
            allEventLog.loc[0, statsList]
            .to_frame()
            .T.set_index(pd.Index([seg.dfIndex()]))
        )

        # add events to baseline plot:
        for i, j in enumerate(zip(eventStartIdx, eventStopIdx)):
            ax0_longtrace.plot(
                np.arange(j[0], j[1] - 1, 1) / seg.samplerate,
                dataSet[j[0] : j[1] - 1],
                marker="x",
                linestyle="None",
                color="blue",
                markersize=1,
                markeredgewidth=0.5,
            )
            ax0_longtrace.plot(
                j[0] / seg.samplerate,
                dataSet[j[0]],
                marker="^",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="green",
                markersize=1,
                markeredgewidth=0.5,
            )

        # Here, we plot a little scattergram of all the events, their type and some statistics -- this was originally in here, but conceptually makes more sense to break out as a function.
        plot_utils.plot_events_scatterplot(allEventLog, writeFile, seg)

    else:
        logging.info(
            "No events found, no events written in {s0}_{s1}mV".format(
                s0=seg.name, s1=seg.VAC
            )
        )

    # format longtrace figure
    fig_longtrace.legend()
    fig_longtrace.savefig(
        str(writeFile) + "_eventfinderBaseline.png", dpi=200, format="png"
    )
    fig_longtrace.clear()
    plt.close(fig_longtrace)

    plot_utils.data_plot(seg, rc, avg, sigmaValue, PDF, dataSetMinLengthAv, dataSet)

    logging.info("End event finding on {s0}_{s1}mV".format(s0=seg.name, s1=seg.VAC))
    seg.idealTrace = idealTrace
    seg.compressedTrace = np.array(compressedTrace)

    del avg, fig_longtrace, noEventsIndex

    return dataRunner


def event_dwell_time_analysis(allEventsLog, seg, rc):
    """
    This function calls the event dwell time survival fit to different models.

    The following parameters are fitted:

            event dwell time = K_OFF
            inter event dwell time = K_ON
            time from start point of one to the next event = RISINGEDGE


    Parameters
    ----------
    allEventsLog : pd.dataframe
        all the single event parameter. Can be read from events.csv
    seg : slice data class object
        rc : runconstant data class object

    Returns
    -------
    None.
        Will change segs statsDF.
    """

    eventFileName = seg.dfIndex() + "_"
    newPath = rc.resultsPath / "events" / "kvalues" / eventFileName
    newPath.parent.mkdir(parents=True, exist_ok=True)

    # create empty keys
    keys = get_keys()
    values = np.empty(len(keys))
    values[:] = np.nan
    seg.add_to_statsDF(keys, values)

    event_dwell_time_fit(allEventsLog, seg, rc, newPath, dwellTimeSelector="EVENT")
    event_dwell_time_fit(allEventsLog, seg, rc, newPath, dwellTimeSelector="NOEVENT")
    event_dwell_time_fit(allEventsLog, seg, rc, newPath, dwellTimeSelector="RISINGEDGE")

    return


def event_dwell_time_fit(allEventLog, seg, rc, writeFile, dwellTimeSelector="EVENT"):
    """
    This function does the event dwell time survival fit to different models.
    The following parameters are fitted based on dwellTimeSelector:

    'EVENT': event dwell time = K_OFF
    'NOEVENT': inter event dwell time = K_ON
    'RISINGEDGE': time from start point of one to the next event = RISINGEDGE

    At the moment 3 different functions are fitted, which is a total mess:
    * single exponential decay A exp(k t)
    * double exponential decay A_fast exp(k_fast t) + A_slow exp(k_slow t)
    * stretched exponential decay (aka Weibull distribution) A exp((k t) to the power alpha)
    * this converges to single exponential if alpha = 1
    * in this case the apparent K = K / Gamma(1 / alpha)

    Output:
    3 plots with survival histogram of parameter with fitted curve and fitting values
    populates seg.statsDF to add fitting parameter, as well as reduced chi, error of fitting parameter (1std)

    Parameters
    ----------
    allEventsLog : pd.dataframe
        all the single event parameter. Can be read from events.csv
    seg : slice data class object
    rc : runconstant data class object
    writeFile : Pathlib Path
    dwellTimeSelector : str
        options: 'EVENT', 'NOEVENT', 'RISINGEDGE" to select which dwell time will be fitted

    Returns
    -------
    none
    """

    # calculated cumulative event count for each case:
    if dwellTimeSelector == "NOEVENT":
        dwellTime = "INTEREVENTDWELLTIME"
        allEventLog = calculate_inter_event_time(allEventLog)
    elif dwellTimeSelector == "RISINGEDGE":
        dwellTime = "RISINGEDGEDWELLTIME"
        allEventLog = calculate_rising_edge_time(allEventLog)
    elif dwellTimeSelector == "EVENT":
        dwellTime = "EVENTDWELLTIME"

    logging.info(
        "{s1} dwell time analysis and survival time fit for {s0}".format(
            s0=writeFile.stem, s1=dwellTimeSelector
        )
    )
    dwellTimeRank = (
        allEventLog[dwellTime].round(-2).value_counts().sort_index(ascending=False)
    )
    dwellTimeRank = dwellTimeRank.to_frame().reset_index()
    dwellTimeRank.rename(columns={dwellTime: "COUNT"}, inplace=True)
    dwellTimeRank.rename(columns={"index": dwellTime}, inplace=True)
    dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"] = dwellTimeRank.COUNT.cumsum()

    # rejection of very long dwell time events for fitting
    totalCount = dwellTimeRank["CUMULATIVEDWELLTIMECOUNT"].max()
    # if not np.isnan(totalCount):
    #     thresholdDwellTime = dwellTimeRank.loc[dwellTimeRank['CUMULATIVEDWELLTIMECOUNT'] > int(totalCount / 2)][dwellTime].values[0] * 20
    # #print(dwellTimeRank.index)
    #     if dwellTimeRank[dwellTime].values[0] > thresholdDwellTime:
    #         logging.info('{s0}: DROPPING LONG EVENTS because longer than {s1:.2e} '.format(s0=writeFile.stem, s1=thresholdDwellTime))
    #         dwellTimeRank = dwellTimeRank[dwellTimeRank[dwellTime] <= thresholdDwellTime]
    #     # q_hi = dwellTimeRank[dwellTime].quantile(0.99)
    #     # if (dwellTimeRank[dwellTime] < q_hi).any():
    #     #     print('{s0}: DROPPING LONG EVENTS because of Quantile'.format(s0=writeFile.stem))
    #     # dwellTimeRank = dwellTimeRank[dwellTimeRank[dwellTime] < q_hi]
    # no fitting if too little events counted
    if len(dwellTimeRank.index) <= 3:
        logging.info("{s0}: No fit too few events".format(s0=writeFile.stem))

    k_guess = 10
    if not np.isnan(
        totalCount
    ):  # if there's something to fit, fit it below and then plot the fits.
        if dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[-1] > 3:
            # single exponential fit
            out = lmFIT_single(
                dwellTimeRank[dwellTime].values * 1e-6,
                dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
                rc.args.filter * 2.0,
            )
            keys, values = lmfit_to_param(out, dwellTimeSelector)
            seg.add_to_statsDF(keys, values)
            k_guess = 1 / out.best_values["exp1_decay"]

        if dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values[-1] > 5:
            # double exponential fit
            out = lmFIT_double(
                dwellTimeRank[dwellTime].values * 1e-6,
                dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
                k_guess,
                rc.args.filter * 2.0,
            )
            keys, values = lmfit_to_param(out, dwellTimeSelector)
            seg.add_to_statsDF(keys, values)

            # stretchted exponential fit
            out = lmFIT_strtchtd_exponential(
                dwellTimeRank[dwellTime].values,
                dwellTimeRank.CUMULATIVEDWELLTIMECOUNT.values,
                k_guess,
                rc.args.filter * 2.0,
            )
            keys, values = lmfit_strtchtd_to_param(out, dwellTimeSelector)
            seg.add_to_statsDF(keys, values)

        # plot of surival fit
        plot_events_dwellTimeDist(
            dwellTimeRank, writeFile, seg.statsDF, dwellTimeSelector
        )

    return


def get_keys():
    """
    This function creates all possible keys for the k-value fitting.
    prevents errors when accessing not fitted k-values down the line

    Parameters
    ------------
    None.

    Returns
    ---------
    keys : list of strings
        with all keys
    """

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
                "K_" + kType + "_stretchted",
                "K_" + kType + "_stretchted_1sigma",
                "K_" + kType + "_stretchted_rchi",
                "A_" + kType + "_stretchted",
                "ALPHA_" + kType + "_stretchted",
            ]
        )
    return keys


def calculate_inter_event_time(allEventLog):
    """
    This function calculates dwell times for k_on value fitting from event index

    Parameters
    ----------
    allEventsLog : pd.dataframe
        all the single event parameter. Can be read from the events.csv

    Returns
    -------
    allEventsLog : pd.DataFrame
    """
    eventStartIdx = allEventLog.EVENTSTARTIDX.values
    eventStopIdx = allEventLog.EVENTSTOPIDX.values
    samplerate = np.mean((eventStopIdx - eventStartIdx) / allEventLog.EVENTDWELLTIME)
    # do interevent duration analysis
    noEventDwellTime = (eventStartIdx[1:] - eventStopIdx[:-1]) / samplerate
    allEventLog["INTEREVENTDWELLTIME"] = np.append(noEventDwellTime, np.nan)
    return allEventLog


def calculate_rising_edge_time(allEventLog):
    """
    This function calculates times from one to the next rising / falling edge

        Parameters
        ----------
        allEventsLog : pd.dataframe
            all the single event parameter. Can be read from events.csv

        Returns
        -------
        allEventsLog : pd.DataFrame
    """
    eventStartIdx = allEventLog.EVENTSTARTIDX.values
    eventStopIdx = allEventLog.EVENTSTOPIDX.values
    samplerate = np.mean((eventStopIdx - eventStartIdx) / allEventLog.EVENTDWELLTIME)
    # do interevent duration analysis
    noEventDwellTime = (eventStartIdx[1:] - eventStartIdx[:-1]) / samplerate
    allEventLog["RISINGEDGEDWELLTIME"] = np.append(noEventDwellTime, np.nan)
    return allEventLog


def double_Exp(x, dwellTime):
    """
    for plotting: double exponential function with x[0]*exp[-dwell * x[1]] + x[2]*exp[-dwell * x[3]]
    converts from us input

    Parameters
    ----------
    x : np.float array, length 4
        contains fitting parameter of function

    dwellTime: np.float array
        in us, time in 1e-6s

    Returns
    -------
    function value
    """
    return x[0] * np.exp(-dwellTime * 1e-6 * x[1]) + x[2] * np.exp(
        -dwellTime * 1e-6 * x[3]
    )


def single_Exp(x, dwellTime):
    """
    for plotting: double exponential function with single exponential function with x[0]*exp[-dwell * x[1]]
    converts from us input

    Parameters
    ----------
    x : np.float array, length 2
        contains fitting parameter of function

    dwellTime: np.float array
        in us, time in 1e-6s

    Returns
    -------
    function value
    """
    return x[0] * np.exp(-dwellTime * 1e-6 * x[1])


def strtchtd_exponential(x, A, k, alpha):
    """
    for plotting: double exponential function with single exponential function with A*exp[-(k * x)**alpha]
    converts from us input

    Parameters
    ----------
    x : np.float array,
        time in 1e-6s
    A : np.float
        parameter of function
    k : np.float
        paramter of funciton
    alpha : np.float
        parameter of function


    Returns
    -------
    function value
    """
    return A * np.exp(-((k * x * 1e-6) ** alpha))


def plot_events_dwellTimeDist(dwellTimeRank, writeFile, DF, dwellTimeSelector="EVENT"):
    """
    plot event dwell time cumulative event count with events dwell time >0 x

    This code no longer assumes that all the dataframes are filled -- it does check for null columns, but not completely.

    Parameters
    ----------
    allEventsLog : pd.dataframe
        all the single event parameter. Can be read from events.csv
    dwellTimeRank : pd.dataframe
        containing count of events with dwell times shorter than x (in microseconds)
    writeFile : Posix Path
        of file name to save
    DF : pd.dataFrame
        containing fitting parameter
    dwellTimeSelector : str
        options: 'EVENT', 'NOEVENT', 'RISINGEDGE" to select which dwell time will be fitted

    Returns
    -------
    none
    """

    if dwellTimeSelector == "NOEVENT":
        dwellTime = "INTEREVENTDWELLTIME"
    elif dwellTimeSelector == "RISINGEDGE":
        dwellTime = "RISINGEDGEDWELLTIME"
    elif dwellTimeSelector == "EVENT":
        dwellTime = "EVENTDWELLTIME"

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

    if (
        dwellTimeSelector == "EVENT"
        and "A_OFF" in DF.columns
        and "AdS_OFF" in DF.columns
    ):
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
    elif (
        dwellTimeSelector == "NOEVENT"
        and "A_ON" in DF.columns
        and "AdS_OFF" in DF.columns
    ):
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
    elif dwellTimeSelector == "RISINGEDGE" and "A_RISINGEDGE" in DF.columns:
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

    if dwellTimeSelector == "EVENT":
        axsns.set(
            xlabel="Event duration [$\mu sec$]", ylabel="k_off cumulative event count"
        )
        tmp = str(writeFile.stem) + "_K_OFF"
        writeFile = writeFile.parent / tmp
    elif dwellTimeSelector == "NOEVENT":
        axsns.set(
            xlabel="interevent duration [$\mu sec$]",
            ylabel="k_on cumulative interevent count",
        )
        tmp = str(writeFile.stem) + "_K_ON"
        writeFile = writeFile.parent / tmp
    elif dwellTimeSelector == "RISINGEDGE":
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


def lmFIT_strtchtd_exponential(x, y, k_guess, k_upper):
    """
    implementation of weibull distribution fitting function (stretched exponential) with lmfit module

    Parameters
    ----------
    x : np.array
        min to max occuring dwell time bins in [s]
    y : np.array
        number of events with at specific dwell time bin
    k_guess : np.float
        initial guess of k_value based on single exponetial fit


    Returns
    -------
    out : lmfit object

    """

    mod = Model(strtchtd_exponential, nan_policy="omit")
    pars = mod.make_params(A=y[-1], k=k_guess, alpha=0.9)
    pars = mod.make_params(k=k_guess, alpha=0.9)
    pars["A"].set(value=y[-1], min=y[-1] * 0.5, max=y[-1] * 3.0)
    pars["k"].set(value=k_guess, min=1e-6, max=k_upper)
    pars["alpha"].set(value=0.9, min=0.1, max=1)
    out = mod.fit(y, pars, x=x)
    return out


def lmFIT_single(x, y, k_upper):
    """
    implementation of exponential distribution fitting function with lmfit module

    Parameters
    ----------
    x : np.array
        min to max occuring dwell time bins in [s]
    y : np.array
        number of events with at specific dwell time bin

    Returns
    -------
    out : lmfit object

    """
    exp_mod1 = ExponentialModel(prefix="exp1_", nan_policy="omit")
    pars = exp_mod1.guess(y, x=x)
    pars["exp1_amplitude"].set(value=y[-1], min=0, max=y[-1] * 3.0)
    pars["exp1_decay"].set(value=1 / 5, min=1 / k_upper, max=1e6)
    mod = exp_mod1
    out = mod.fit(y, pars, x=x)
    return out


def lmFIT_double(x, y, k_guess, k_upper):
    """
    implementation of double exponential distribution fitting function with lmfit module

    Parameters
    ----------
    x : np.array
        min to max occuring dwell time bins in [s]
    y : np.array
        number of events with at specific dwell time bin
    k_guess : np.float
        initial guess of k_value based on single exponetial fit


    Returns
    -------
    out : lmfit object

    """

    exp_mod1 = ExponentialModel(prefix="exp1_", nan_policy="omit")
    pars = exp_mod1.guess(y, x=x)
    exp_mod2 = ExponentialModel(prefix="exp2_", nan_policy="omit")
    pars.update(exp_mod2.make_params())
    pars["exp1_amplitude"].set(value=y[-1] / 2.0, min=0, max=y[-1] * 3.0)
    pars["exp1_decay"].set(value=1 / k_guess / 10.0, min=1 / k_upper, max=k_upper)
    pars["exp2_amplitude"].set(value=y[-1] / 2.0, min=0, max=y[-1] * 3.0)
    pars["exp2_decay"].set(value=1 / k_guess, min=1 / k_upper, max=1e6)

    mod = exp_mod1 + exp_mod2
    out = mod.fit(y, pars, x=x)
    return out


def lmfit_strtchtd_to_param(out, dwellTimeSelector):
    """
    converts lmfit out object to keys and values for loging in statsDF for the stretched exponential fitting

    Parameters
    ----------
    out : lmfit object
    dwellTimeSelector : str
            options: 'EVENT', 'NOEVENT', 'RISINGEDGE" to select which dwell time will be fitted

    Returns
    -------
    keys : keys of fitting parameter to populate statsDF
    values : values of fitting parameter to populate statsDF

    """

    if dwellTimeSelector == "EVENT":
        keys = [
            "K_OFF_stretchted",
            "A_OFF_stretchted",
            "ALPHA_OFF_stretchted",
            "K_OFF_stretchted_rchi",
            "K_OFF_stretchted_1sigma",
            "K_OFF_stretched_apparent",
        ]
    elif dwellTimeSelector == "NOEVENT":
        keys = [
            "K_ON_stretchted",
            "A_ON_stretchted",
            "ALPHA_ON_stretchted",
            "K_ON_stretchted_rchi",
            "K_ON_stretchted_1sigma",
            "K_ON_stretched_apparent",
        ]
    elif dwellTimeSelector == "RISINGEDGE":
        keys = [
            "K_RISINGEDGE_stretchted",
            "A_RISINGEDGE_stretchted",
            "ALPHA_RISINGEDGE_stretchted",
            "K_RISINGEDGE_stretchted_rchi",
            "K_RISINGEDGE_stretchted_1sigma",
            "K_RISINGEDGE_stretched_apparent",
        ]
    if len(out.best_values.keys()) == 3:
        values = [
            out.best_values["k"],
            out.best_values["A"],
            out.best_values["alpha"],
            out.redchi,
        ]
        if out.errorbars or (not (dwellTimeSelector == "RISINGEDGE")):
            try:
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
            except Exception as e:
                values.extend([np.nan])
        else:
            values.extend([np.nan])
    values.extend([out.best_values["k"] / gamma(1 + 1.0 / out.best_values["alpha"])])
    return keys, values


def lmfit_to_param(out, dwellTimeSelector):
    """
    converts lmfit out object to keys and values for loging in statsDF for single and double exponential fitting.

    Parameters
    ----------
    out : lmfit object
    dwellTimeSelector : str
            options: 'EVENT', 'NOEVENT', 'RISINGEDGE" to select which dwell time will be fitted

    Returns
    -------
    keys : keys of fitting parameter to populate statsDF
    values : values of fitting parameter to populate statsDF

    """

    if dwellTimeSelector == "EVENT":
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
            if out.errorbars:
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
    elif dwellTimeSelector == "NOEVENT":
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
            if out.errorbars:
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
    elif dwellTimeSelector == "RISINGEDGE":
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
            if out.errorbars:
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
                values.extend([np.nan, np.nan])
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

    return keys, values
