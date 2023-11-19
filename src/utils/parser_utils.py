import argparse


def cmd_parser():
    ## Input argument parser
    # General run definitions
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Define number of processes to spawn in parallel. Has to be servicible on slurm or local machine.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Do nothing except print out what data files we'll run on, and exit.",
    )
    parser.add_argument(
        "--ramcheck",
        action="store_true",
        help="Helps with not overloading RAM by checking for free space before asking for a large malloc().  Right now, fails for smFET data.",
    )
    parser.add_argument(
        "--fileNumber",
        type=int,
        default=1000000,
        help="Experiment files with same name are concatenated. FILENUMBER defines the number of files to add to an experiment list, easiest way to reduce analysis amount, for initial results or to save memory.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep results of last run in the DataAndResults/results folder; for subsequent analysis this might be handy. A copy is put to the DataAndResults/resultsArchive folder before running this either way.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Forces a run in serial mode, without threading / parallel execution (overwrites -t THREADS argument) and without explicit calls to spawning new threads. Convenient for debugging in cases where standard output gets mangled or sent somewhere else.",
    )
    parser.add_argument(
        "--expplot",
        action="store_true",
        help="Chimera/Benchvue option. Trace overview plot for the entire experiment. Works mainly for Chimera and Benchvue files.",
    )
    parser.add_argument(
        "--segplot",
        action="store_true",
        help="Default output of trace for each data segment, PSD and cumulative noise. To check sanity of data.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="preprocess the data with baseline drift correction. Stores pickle in original data folder without Iraw.",
    )
    parser.add_argument(
        "--loadpreprocessed",
        action="store_true",
        help="load preprocess data instead of raw data and runs anlysis from there",
    )
    parser.add_argument(
        "--fitTransients",
        action="store_true",
        help="Chimera/Benchvue option. Run the polynomial model RC fitter and exponential fitter on the current trace. For drifting baselines to analyse settling value.",
    )
    parser.add_argument(
        "--fitPore",
        action="store_true",
        help="Chimera to fit pore capacitance and resistance based on FFT.",
    )
    parser.add_argument(
        "--overviewplot",
        action="store_true",
        help="Default output of trace for each data segment, PSD and cumulative noise. To check sanity of data.",
    )
    parser.add_argument(
        "--instrumentType",
        type=str,
        default="Chimera",
        help="Select instrument type for data import. Available options are: INSTRUMENTTYPE = 'Chimera', 'HEKA', 'smFET', 'CNP2'.",
    )
    parser.add_argument(
        "--sliceTime",
        type=float,
        default=float("nan"),
        help="Create individual chunks of one experiment and creates segments. For statistical analysis of long recordings. Experiment is cut into SLICETIME [s], assuming total recording being 600s (no better way found) -- BUG, we can make an Experiment.Length here, based on looking at file length and samplefreq.",
    )
    parser.add_argument(
        "--slicefromfile",
        action="store_true",
        help="Looks for text file with the same name as the experiment, and ordered pairs of start and end points.  Returns slices based on those time segments.  If there's no file, returns the trivial slice.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export full experiment data to csv after filtering.",
    )
    parser.add_argument(
        "--export_slice",
        action="store_true",
        help="Export Slice() data to csv after filtering.",
    )
    parser.add_argument(
        "--picklesegment",
        action="store_true",
        help="Export full current trace data to pickle object after filtering.  Used for the Schnellubsersicht GUI data viewer.",
    )
    parser.add_argument(
        "--channel",
        nargs="+",
        default=[],
        help="Define channels to look at (smFET) platform, can be specified as list: --channel 01 09 11 21  Will reject files by name if they are not in those channels.",
    )

    # Filtering setup
    parser.add_argument(
        "-f",
        "--filter",
        type=float,
        default=5e3,
        help="Lowpass filter cutoff frequency, in Hz.",
    )
    parser.add_argument(
        "--filterType",
        type=str,
        default="FIR",
        help="Definition of filter type. Defaults to FIR. Strongly suggest to use default FIR (finite impulse response) filter for best performance. FILTERTYPE Options: Bessel (10order), FIR (default), Butter (10order). ONLY FIR works <25kHz for Chimera and CNP2 data. ",
    )
    parser.add_argument(
        "--stopBand",
        action="store_true",
        help="Activates filter to supress hump at 200Hz (100-1000Hz)",
    )
    parser.add_argument(
        "--notch",
        action="store_true",
        help="Activates notch filter to supress hump at 60Hz (50-70Hz)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="none",
        help="Definition of wavlet filter mother wavelet. Default 'none' uses lowpass filter as specified. Options are: 'rbio1.1' and all wavelets of pywt",
    )
    parser.add_argument(
        "--waveletlevel",
        type=int,
        default=8,
        help="Level of wavelet decomopisiotn. The higher the stronger filtered... different levels for different instruments required",
    )
    parser.add_argument(
        "--waveletthreshold",
        type=float,
        default=1.0,
        help="threshold parameter mutliplyer, default 1.0 for more filtering increase parameter",
    )

    # for eventfinding
    parser.add_argument(
        "--findevents",
        action="store_true",
        help="Executes iterative event finder on each experimental segment.",
    )
    parser.add_argument(
        "--PDF",
        type=float,
        default=6,
        help="Defines Peak Detection Factor threshold value for event start point, where the thhreshold is PDF * (standard deviation of signal), default is 6.0 (very conservative. Does not work for low SNR).",
    )
    parser.add_argument(
        "--PDFreversal",
        type=float,
        default=2,
        help="Defines Peak Detection Factor threshold value for event end. PDFREVERSAL = sigma value, Threshold is PDFREVERSAL * (standard deviation of signal),default is 2.0",
    )
    parser.add_argument(
        "--eventThreshold",
        type=float,
        default=float('nan'),
        help="fixed threshold parameter for event finder. Will be used if smaller than PDF*std. default value is nan which deactivates the parameter",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Number of passes for iterative event finding, default is 3, for noisy data with a very frequent or long events higher number of iterations give better results",
    )
    parser.add_argument(
        "--avwindow",
        type=float,
        default=3.0,
        help="Defines the window size AVWINDOW [s] for moving average calculation in event finder. Default is AVWINDOW = 3.",
    )
    parser.add_argument(
        "--detrend",
        type=float,
        default=float("nan"),
        help="Detrend data before event finding to improve initialization of event finder. Warning decreases long event depths. Should be used in combination with --HistBaseline argument. Specify detrend time window in [s]",
    )
    parser.add_argument(
        "--mineventlength",
        type=float,
        default=float("nan"),
        help="minimum event length excepted, default is none which will result in 1/filter [s]",
    )
    parser.add_argument(
        "--HistBaseline",
        action="store_true",
        help="Initialize baseline value and standard deviation for event finder with highest probability peak of gaussian mixture model. Does not work for drifiting data.",
    )
    parser.add_argument(
        "--plotevents",
        action="store_true",
        help="Saves a single plot of event trace and event threshold values for each event found. To check event finder, or to look at single events.",
    )
    parser.add_argument(
        "--plotidealevents",
        type=float,
        default=float(
            "nan"
        ),  # Caution -- this should be NAN, becuase then the singleplot events fails TODO
        help="Saves plots with length PLOTIDEALEVENTS [s] with current trace and idealized event traces for the entire segment.  This is probably the one you want for plots of a few seconds of events each.",
    )
    parser.add_argument(
        "--plotundecimated",
        action="store_true",
        help="Saves plots with one second slices of data undecimated in event finder. To check data, baseline, etc.",
    )
    parser.add_argument(
        "--eventType",
        type=str,
        default="both",
        help="Defines type of events to look for. Other events are discarded. Works for both negative and positive currents Options of EVENTTYPE: 'both' (default), 'blockage' (abs(current) < baseline), 'enhance' (abs(current) > baseline)",
    )
    parser.add_argument(
        "--alphaError",
        type=float,
        default=0.05,
        help="Defines alpha error for event detection student - t test. Confidence interval of loglikelyhood test whether observed 'event' mean = baseline mean (H0). Defines confidence value of event mean = baseline mean. Default = 0.05, implying 95% confidence value, by Student's t-test, that the event you detected is actually en event. Good spike/noise rejection for alphaError = 0.000001 BUG no idea who tested that statement. Allows to define lower PDF value to detect long shallow events, but still supress short spikes.",
    )
    parser.add_argument(
        "--endOfEventParameter",
        type=float,
        default=3.0,
        help="Defines endOfEventParameter which corresponds to the length of the forward looking moving average window. This parameter rejects short up/down spikes which could terminate an event early.",
    )
    parser.add_argument(
        "--mineventdepth",
        type=float,
        default=1.0,
        help="Defines minimum event depth for logged events. MINEVENTDEPTH is specified in the base current units -- pA for Chimera or nA for smFET, for example.",
    )
    parser.add_argument(
        "--fakeevents",
        action="store_true",
        help="Adds fake events to data, duration and location is random. You can specify depth and rate in the code.  Used to test the event finder.",
    )
    parser.add_argument(
        "--interpolateBaseline",
        action="store_true",
        help="interpolate between end and start of baseline for the baseline keeper value - not always a good option(for long events not)",
    )
    parser.add_argument(
        "--exportEventTrace",
        action="store_true",
        help="Export current trace and eventrace in plotidealevents slices, to a csv that contains three columns: relative time in sec, filtered current data, and idealized events trace.",
    )
    parser.add_argument(
        "--kvalueanalysis",
        action="store_true",
        help="Enables k value fitting to get rate constants for event duration, interevent duration and event to event duration.",
    )
    parser.add_argument(
        "--hmm",
        type=str,
        default="none",
        help="BUG not implemented fully.  options: 'none', 'hidden', 'two'",
    )
    parser.add_argument(
        "--blcorrectionwhittaker",
        nargs="+",
        default=[],
        help="together with --preprocess activates baseline correction using Whittaker baseline correction https://pybaselines.readthedocs.io/en/stable/algorithms/whittaker.html, parameter control [lam, p, lam_1] which is the smoothness parameter.",
    )
    parser.add_argument(
        "--blcorrectionfreq",
        type=float,
        default=float("nan"),
        help="together with --preprocess activates baseline correction using event finder with filtered data at lowpass frequency 'blcorrectionfreq'",
    )
    parser.add_argument(
        "--twostephmm",
        action="store_true",
        help="does initial HMM fitting to estimate event and baseline level on lower filter cutoff data. FOR NOISY DATA",
    )
    parser.add_argument(
        "--fixedDistributions",
        action="store_true",
        help="only in combination with --twostephmm. Fixes the distribution values for baseline and event state to mean and std values found with low frequency cutoff HMM fit. FOR NOISY DATA",
    )
    parser.add_argument(
        "--gmmdistributionfit",
        action="store_true",
        help="only in combination with --fixedDistributions. Fixes the distribution values for baseline and event state to gmm fit - only wit corrected baseline",
    )
    parser.add_argument(
        "--fitfraction",
        type=float,
        default=1.0,
        help="fraction of data used for fitting hmm (dataSet[0:int(fitfraction * len(dataSet)]). dataSet[int(fitfraction * len(dataSet):] (1-fitfraction) will be used to calculate HMM loglikelyhood",
    )
    parser.add_argument(
        "--saveHMM", action="store_true", help="pickle hmm simulated trace"
    )
    parser.add_argument(
        "--normalizeHMM", action="store_true", help="pickle normalized data trace"
    )
    parser.add_argument(
        "--readHMM",
        action="store_true",
        help="read json from disk and simulate hmm trace",
    )
    parser.add_argument(
        "--highresplots",
        action="store_true",
        help="Saves a high resolution plot of each trace similar to plotidealevents even if no events found.",
    )
    parser.add_argument(
        "--noHmmSpikeFilter", action="store_true", help="no spike filter in hmm model"
    )
    parser.add_argument(
        "--reversepolarity",
        action="store_true",
        help="if used the current trace is inverted upon import. To adress closed/open channel assignment issue for smFET",
    )

    return parser
