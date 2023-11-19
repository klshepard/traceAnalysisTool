import pyqtgraph as pg

# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph.widgets.RemoteGraphicsView
import numpy as np
import pickle
from pathlib import Path
import argparse
import sys
import multiprocessing as mp
import time
import pandas as pd
from datetime import datetime, timedelta
import threading
from queue import Queue
import signal
import pybaselines

try:
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import (
        QApplication,
        QWidget,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QPushButton,
    )

    QT_VERSION = 6
except ModuleNotFoundError:
    # Error handling
    try:
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *  # QGridLayout, QPushButton, QRadioButton
        from PyQt5.QtWidgets import (
            QApplication,
            QWidget,
            QGridLayout,
            QFileDialog,
            QCheckBox,
            QPushButton,
        )

        QT_VERSION = 5
        pass
    except ModuleNotFoundError as err:
        print(err)
        pass

if pg.__version__ >= "0.12.3":
    HIST_LAST_VALUE = 2
else:
    HIST_LAST_VALUE = 1

HIST_UPDATE_INTERVAL = 1
REGION_UPDATE_INTERVAL = 2
REGION_UPDATE_FRACTION = 0.05

sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src" / "utils"))

import data_classes
import filter_utils
import main
import parser_utils
import import_utils
from data_classes import Experiment
from data_classes import Slice

bincount = 100  # how many bins in hist


def read_pckl_data(readFile):
    with open(readFile, "rb") as f:
        seg = pickle.load(f)
        if (hasattr(seg, "idealTrace")) & (hasattr(seg, "compressedTrace")):
            return (
                seg.IaRAW,
                seg.Ia,
                seg.samplerate,
                seg.exp.metaData["SETUP_ADCSAMPLERATE"],
                seg.idealTrace,
                seg.compressedTrace,
            )
        elif hasattr(seg, "hmmTrace"):
            return (
                seg.IaRAW,
                seg.Ia,
                seg.samplerate,
                seg.exp.metaData["SETUP_ADCSAMPLERATE"],
                seg.hmmTrace,
                None,
            )
        elif hasattr(seg, "idealTrace"):
            return (
                seg.IaRAW,
                seg.Ia,
                seg.samplerate,
                seg.exp.metaData["SETUP_ADCSAMPLERATE"],
                seg.idealTrace,
                None,
            )
        elif hasattr(seg, "compressedTrace"):
            return (
                seg.IaRAW,
                seg.Ia,
                seg.samplerate,
                seg.exp.metaData["SETUP_ADCSAMPLERATE"],
                None,
                seg.compressedTrace,
            )
        elif not hasattr(seg, "IaRAW"):
            # k=np.zeros((len(seg[0])))
            # k[:]=seg[0]
            return None, seg[0], seg[1], None, None, None
        else:
            return (
                seg.IaRAW,
                seg.Ia,
                seg.samplerate,
                seg.exp.metaData["SETUP_ADCSAMPLERATE"],
                None,
                None,
            )


def loadData(rc):

    fileName = Path(rc.args.file)
    (
        IaRAW,
        Ia,
        samplerate,
        ADC_sample,
        idealTrace,
        compressedTrace,
    ) = read_pckl_data(fileName)

    if not np.isnan(rc.args.filter):
        if ("none" in rc.args.wavelet) and (rc.args.filter < 4166667):
            Ia = filter_utils.lowpass_filter(
                IaRAW, rc, ADC_sample, order=9, fType=rc.args.filterType, resample=4.0
            )
        elif rc.args.filter > 4166667:
            Ia = IaRAW
        else:
            Ia = filter_utils.wavelet_filter(IaRAW, rc)
            newSampleRate = rc.args.filter * 4
            filteredInt = True
            if newSampleRate < ADC_sample:
                Ia = filter_utils.lowpass_resample(
                    Ia, samplerate, newSampleRate, filteredInt=filteredInt
                )
        samplerate = ADC_sample * np.size(Ia) / np.size(IaRAW)

    if not np.isnan(rc.args.baseline):
        baseLine, param = pybaselines.whittaker.iasls(Ia,lam=rc.args.baseline, p=0.85)
        Ia = Ia - baseLine
        print('doing baseline correction')

    if not np.isnan(rc.args.notch):
        notchQ = 10
        Ia = filter_utils.notch_filter(Ia, rc.args.notch, notchQ, samplerate)
        print('notch filtering f = {}, qualtity factor = {}'.format(rc.args.notch, notchQ))

    del IaRAW
    time = np.arange(0, len(Ia)) / samplerate
    if compressedTrace is not None:
        compressedTrace[:, 0] = compressedTrace[:, 0] / samplerate
    return (
        np.array(Ia, dtype=float),
        time,
        samplerate,
        fileName,
        idealTrace,
        compressedTrace,
    )

class rc2:
    def __init__(self, args):
        self.args = args
        self.k = 0


class args:
    def __init__(self, fs, ftype):
        self.filter = fs
        self.filterType = ftype


## parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--filter",
    type=float,
    default=np.nan,
    help="Lowpass filter cutoff frequency defined by FILTER",
)
parser.add_argument(
    "--baseline",
    type=float,
    default=np.nan,
    help="lambda value for whittaker baseline correction",
)
parser.add_argument(
    "--file", type=str, default="none", help="File name of pickled slice file to load."
)
parser.add_argument(
    "--filterType",
    type=str,
    default="FIR",
    help="Definition of filter type. Defaults to FIR. Strongly suggest to use default FIR (finite impulse response) filter for best performance. FILTERTYPE options: Bessel (10th order), FIR (default), Butterworth (10 order). ONLY FIR works <25kHz for Chimera and CNP2 data. TODO Jakob what's this mean?",
)
parser.add_argument(
    "--wavelet",
    type=str,
    default="none",
    help="Definition of wavelet filter mother wavelet. Default 'none' uses lowpass filter as specified. Options are: 'rbio1.1' and all wavelets of pywt.",
)
parser.add_argument(
    "--waveletlevel",
    type=int,
    default=8,
    help="Level of wavelet decomposition. The higher, the stronger the filtering effect -- different levels for different instruments required.",
)
parser.add_argument(
    "--waveletthreshold",
    type=float,
    default=1.0,
    help="Threshold parameter mutliplier.  Default 1.0 -- for more filtering increase parameter.",
)
parser.add_argument(
    "--notch",
    type=float,
    default=np.nan,
    help="Notch filter frequency, defaults to np.nan which means it is inactive",
)
args = parser.parse_args()
rc = rc2(args)
k = 0


class schnelluebersichtgeraet(QWidget):
    def openFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;Python Files (*.py)",
        )
        if fileName:
            self.fileName = fileName
            self.rc.args.file = self.fileName
            self.p2.clear()
            self.p1.clear()
            del self.compressedTrace, self.Ia, self.idealTrace, self.time, self.p1, self.p2
            self.getData()
            self.closeWidgets()
            self.reloadWidgets()

    def getData(self):
        (
            self.Ia,
            self.time,
            self.samplerate,
            self.fileName,
            self.idealTrace,
            self.compressedTrace,
        ) = loadData(self.rc)

    def reloadWidgets(self):
        self.__init__()

    def closeWidgets(self):
        self.close()

    def updatePlot(self):
        """
        updateSTD the detail plot x range
        """
        self.p2.setXRange(*self.lr.getRegion(), padding=0)
        # self.p2.setYRange()

    def updateRegion(self):
        """
        updates the highlight region (blue shade) in overview (top) plot which is p1
        """
        self.lr.blockSignals(True)
        self.lr.setRegion(self.p2.getViewBox().viewRange()[0])
        self.lr.blockSignals(False)

    def updateSTD(self):
        """
        calculates std and mean on detail plot section of data, which is p2
        """
        start = int(self.p2.getViewBox().viewRange()[0][0] * self.samplerate)
        end = int(self.p2.getViewBox().viewRange()[0][1] * self.samplerate)
        imean = np.mean(self.Ia[start:end])
        istd = np.std(self.Ia[start:end])
        self.p2.setTitle(
            "Zoom on selected region I_mean = {s0:0.3f} pA I_std = {s1:0.3f} pA".format(
                s0=imean, s1=istd
            )
        )

    def regionUpdate(self, stopCalculating, timerRegion):
        """
        calculates the new histogram when queue is full and puts it to queue. has a timer.
        """
        while not stopCalculating.isSet() and not timerRegion.wait(
            REGION_UPDATE_INTERVAL
        ):
            start = int(self.p2.getViewBox().viewRange()[0][0] * self.samplerate)
            end = int(self.p2.getViewBox().viewRange()[0][1] * self.samplerate)
            thresh = REGION_UPDATE_FRACTION * (self.oldEnd - self.oldStart)
            if (np.abs(start - self.oldStart) > thresh) or (
                np.abs(end - self.oldEnd) > thresh
            ):
                self.oldStart = start
                self.oldEnd = end
                self.lr.lineMoveFinished()

    def calculateUpdate(_, valueQueue, resultsQueue, stopCalculating, timerCalculating):
        """
        calculates the new histogram when queue is full and puts it to queue. has a timer.
        """
        while not stopCalculating.isSet() and not timerCalculating.wait(
            HIST_UPDATE_INTERVAL
        ):
            if not valueQueue.empty():
                try:
                    Ia = valueQueue.get(block=True, timeout=0.2)
                    rdmSize = 30000
                    if len(Ia) > rdmSize:
                        y, x = np.histogram(
                            np.random.choice(Ia, rdmSize // 2, replace=False),
                            bins=bincount,
                        )
                    else:
                        y, x = np.histogram(Ia, bins=bincount)
                    resultsQueue.put([y, x])
                except:
                    pass

    def updateWidget(self):
        """
        puts new data to queue for histogram calculations, plots data for the detail plot section, which is p3
        """
        start = int(self.p2.getViewBox().viewRange()[0][0] * self.samplerate)
        end = int(self.p2.getViewBox().viewRange()[0][1] * self.samplerate)
        with self.valueQueue.mutex:
            self.valueQueue.queue.clear()
        self.valueQueue.put(self.Ia[start:end])
        # if not self.resultsQueue.empty():

        # thresh = 0.1 * (self.oldEnd - self.oldStart)
        # if (np.abs(start - self.oldStart) > thresh) or (np.abs(end - self.oldEnd) > thresh) or not self.resultsQueue.empty():
        #     self.oldStart = start
        #     self.oldEnd = end
        self.updateSTD()
        try:
            self.p2.setYRange(
                np.min(self.Ia[start:end]),
                np.max(self.Ia[start:end]),
                padding=0.1,
                update=True,
            )
            y, x = self.resultsQueue.get(timeout=0.01)
            self.p3.clear()
            self.p3item0 = self.p3.plot(
                y, x[0:-HIST_LAST_VALUE], stepMode="center", brush=(0, 0, 255, 150)
            )
            self.p3item1 = self.p3.plot(
                x=np.zeros_like(x[0:-HIST_LAST_VALUE]),
                y=x[0:-HIST_LAST_VALUE],
                pen=pg.mkPen("w", width=0),
            )
            fill = pg.FillBetweenItem(
                curve1=self.p3item1,
                curve2=self.p3item0,
                brush=(0, 0, 255, 150),
                pen=pg.mkPen((0, 0, 255, 150)),
            )
            self.p3.addItem(fill)
        except:
            pass

    def plot_mean(self, Ia, time):
        mean = np.mean(Ia)
        self.p2.addLine(x=None, y=mean, pen=pg.mkPen("b", width=3))

    def plot_upper_std(self, Ia, time):
        mean = np.mean(Ia)
        std = np.std(Ia)
        self.p2.addLine(x=None, y=mean + std, pen=pg.mkPen("g", width=3))

    def plot_lower_std(self, Ia, time):
        mean = np.mean(Ia)
        std = np.std(Ia)
        self.p2.addLine(x=None, y=mean - std, pen=pg.mkPen("g", width=3))

    def mask_current(self, Ia, time, new_time):
        """
        Given a current, and a new_time that is shorter than time, return only the points that correspond to current at the new_time points.
        """
        assert len(time) == len(Ia)
        start = np.where(time <= new_time[0])
        end = np.where(time >= new_time[1])
        return Ia[start[0][-1] : end[0][0]]

    def find_events(self, time_chunk, rc):
        """
        Makes a new Slice() to find the events on, then runs the eventfinder on that Slice()

        Major issue is that the eventfinder params are hardcoded into args_handle{} below, and should be read from a file or something.

        Parameters
        ----------
        time_chunk : List() of length 2
                Start and end times for the segment here, from the beginning of the Slice() that was loaded into Schnelluebersicht.
        rc: int
                RunConstants object from main.py, not the one we define here.

        Returns
        -------
        None.
                Will write files.
        """
        file_id = self.fileName
        seg_id = file_id.stem.replace("_savedTrace", "")
        exp_id = "_".join(file_id.stem.replace("_savedTrace", "").split("_")[1:-2])
        the_index = int(file_id.stem.replace("_savedTrace", "").split("_")[-2])
        local_parser = parser_utils.cmd_parser()
        local_args = local_parser.parse_args()
        npersegPARAM = 240  # cut the whole thing in to this many segments
        noverlapPARAM = 1.2  # 1.1 indicates 10% overlap
        seg_front_crop = 5  # seconds to cut off front
        seg_back_crop = 1  # seconds to cut off back
        allowed_time_lag = timedelta(
            seconds=60
        )  # You must start the Benchvue within this many seconds of starting the Chimera.

        resultsPath = Path.cwd().parent / "DataAndResults/results"
        # resultsPath.mkdir(parents=True, exist_ok=True)
        dataPath = Path()

        dataPath = import_utils.get_dataPath(
            resultsPath.parent
        )  # TODO will fail with no data.txt

        ootd = main.import_utils.get_VA(dataPath)
        local_rc = data_classes.rc(
            npersegPARAM,
            noverlapPARAM,
            local_args,
            seg_front_crop,
            seg_back_crop,
            ootd,
            dataPath,
            resultsPath,
            allowed_time_lag,
        )  ## TODO this requires that the data dir be synchronized -- so make data should return the right files.
        # TODO something above here blocks, and does do so for a long time (160 sec?)  Also, some suspicious behavior, since the button does not change to "were doing a run" upon press... even though htop shows it running....
        the_manager = mp.Manager()
        statsDF = the_manager.list()
        experimentList = import_utils.create_experiment_list_Chimera(
            local_rc
        )  # TODO only Chimera for now.
        the_semaphore = the_manager.Semaphore(local_args.threads)
        statsDF = the_manager.list()
        args_handle = vars(local_args)
        # for thing in args_handle:
        #     print(thing)
        # the above is how you access the args, and change them locally.
        ## TODO these arguements below are the entire set of hyperparameters for eventfinding, taken from utils/parser_utils.py  These are the only ones that affect eventfinding, and they are all set manually here.  This is a complete mess, and the right way to do this is to use something like parseConfig for the events params, but I can't do everything here.
        args_handle["findevents"] = True
        args_handle["PDF"] = 5
        args_handle["PDF_reversal"] = 2
        args_handle["passes"] = 6
        args_handle["avwindow"] = 0.1  # in seconds -- so that's 100 milliseconds.
        args_handle["detrend"] = float("nan")
        args_handle["HistBaseline"] = False
        args_handle[
            "plotevents"
        ] = False  # TODO True fails here, should check the order in which these are called in main.
        args_handle["plotidealevents"] = float(
            "nan"
        )  # TODO, if this is not nan, plotevents will fail
        args_handle["eventType"] = "enhance"
        args_handle["alpha_error"] = 0.05
        args_handle["mineventdepth"] = 1.0
        # TODO this will fail if the slice is too short, as it will take the crops off the ends of the slice -- have to assert that slice length minus crops is longer than zero.
        for exp in experimentList:
            if exp == exp_id:  # make yourself an Experiment here!
                mini_dict = {exp: experimentList[exp]}
                experiment = Experiment(mini_dict[exp], local_rc)
                # TODO this is a little slow.  I wonder if memory-mapped io in the file reader would help here.
                slices = experiment.schneiden()
                for a_slice in slices:
                    if a_slice.i == the_index:  # this is THE ONE you run on.
                        del mini_dict
                        fixed_timestamps = [
                            a_slice.transition[0] + timedelta(seconds=time_chunk[0]),
                            a_slice.transition[0] + timedelta(seconds=time_chunk[1]),
                        ]
                        new_slice = Slice(
                            experiment, fixed_timestamps, a_slice.i, local_rc
                        )
                        # TODO maybe do something more intelligent here with the slice numbering, above.
                        del experiment
                        statsDF.append(
                            main.slice_master_func([new_slice, local_args, local_rc])
                        )
        print("Eventfinding complete.")
        return  # END find-events

    def button_run_handler(self):
        # TODO I split these up into separate functions, so we can figure out how to multiprocess -- something like this https://stackoverflow.com/questions/15675043/multiprocessing-and-gui-updating-qprocess-or-multiprocessing/27036191  Similary, a very explicity way of doing this is given here -- https://www.learnpyqt.com/tutorials/qprocess-external-programs/ -- with the added complication that you have to pass the data to the external process, which means file IO or some command line arg manipulations.  Probably prefer some local temp file IO there.
        self.the_button.setText("Starting up a run.")
        start_time = time.time()
        time_chunk = self.lr.getRegion()
        new_current = self.mask_current(self.Ia, self.time, time_chunk)
        self.p2.clear()
        self.p2.plot(x=self.time, y=self.Ia, pen=pg.mkPen("r", width=0.5))
        self.plot_mean(new_current, self.time)
        self.plot_upper_std(new_current, self.time)
        self.plot_lower_std(new_current, self.time)
        self.find_events(time_chunk, self.rc)
        end_time = time.time()
        run_time = end_time - start_time
        self.the_button.setText(
            "Executed in {s0:0.3f} seconds. Go again, if you want.".format(s0=run_time)
        )

    def toggle_trace(self):
        if self.IdealTraceButton.isChecked():
            self.drawP1()
            self.drawP2()
        elif not self.IdealTraceButton.isChecked():
            self.IdealTraceButton.setChecked(False)
            self.p1.removeItem(self.p1item0)
            self.p1.removeItem(self.p1item1)
            self.p2.removeItem(self.p2item0)
            self.p2.removeItem(self.p2item1)
            self.drawP1()
            self.drawP2()

    def drawP1(self):
        self.p1item0 = self.p1.plot(
            x=self.time, y=self.Ia, pen=pg.mkPen("r", width=0.5)
        )
        if isinstance(self.idealTrace, np.ndarray) & self.IdealTraceButton.isChecked():
            self.p1item1 = self.p1.plot(
                x=self.time,
                y=(np.array(self.idealTrace)),
                pen=pg.mkPen((0, 76, 153, 180), width=4.0),
            )
            self.p1item1.setDownsampling(ds=25)  # , mode="peak")
            self.p1item1.setClipToView(True)

    def drawP2(self):
        self.p2item0 = self.p2.plot(
            x=self.time, y=self.Ia, pen=pg.mkPen("r", width=0.5)
        )
        self.p2.setXRange(*self.lr.getRegion(), padding=0)
        if isinstance(self.idealTrace, np.ndarray) & self.IdealTraceButton.isChecked():
            self.p2item1 = self.p2.plot(
                x=self.time,
                y=(np.array(self.idealTrace)),
                pen=pg.mkPen((0, 76, 153, 180), width=3.0),
            )
            self.p2item1.setDownsampling(ds=5)  # , mode="peak")
            self.p2item1.setClipToView(True)

    def toggle_shading(self):
        if self.EventShadingButton.isChecked():
            self.eventRegion1()
            self.eventRegion2()
        elif not self.EventShadingButton.isChecked():
            self.EventShadingButton.setChecked(False)
            if hasattr(self, "event1"):
                for event in self.event1:
                    self.p1.removeItem(event)
            if hasattr(self, "event2"):
                for event in self.event2:
                    self.p2.removeItem(event)

    def eventRegion1(self):
        if self.compressedTrace is not None:
            if len(self.compressedTrace[:, 0]) > 2:
                self.event1 = []
                for i in np.arange(0, ((len(self.compressedTrace[:, 0]) - 2) // 4), 1):
                    if (
                        self.compressedTrace[i * 4 + 2, 1]
                        < self.compressedTrace[i * 4 + 1, 1]
                    ):
                        lineColor = (0, 204, 153)
                        textPos = 0.10
                    else:
                        lineColor = (204, 0, 255)
                        textPos = 0.75
                    self.event1.append(
                        pg.LinearRegionItem(
                            values=[
                                self.compressedTrace[i * 4 + 2, 0],
                                self.compressedTrace[i * 4 + 3, 0],
                            ],
                            bounds=[
                                self.compressedTrace[i * 4 + 2, 0],
                                self.compressedTrace[i * 4 + 3, 0],
                            ],
                            brush=pg.mkBrush(lineColor + (20,)),
                            pen=pg.mkPen(lineColor + (120,), width=1),
                            movable=False,
                        )
                    )
                    self.event1[i].setZValue(-10)
                    self.p1.addItem(self.event1[i])
                    label = pg.InfLineLabel(
                        self.event1[i].lines[0],
                        "event {}".format(i + 1),
                        color="k",
                        anchors=[(0, 0.75), (0, 0.75)],
                        position=textPos,
                        rotateAxis=(1, 0),
                    )

    def eventRegion2(self):
        if self.compressedTrace is not None:
            if len(self.compressedTrace[:, 0]) > 2:
                self.event2 = []
                for i in np.arange(0, ((len(self.compressedTrace[:, 0]) - 2) // 4), 1):
                    if (
                        self.compressedTrace[i * 4 + 2, 1]
                        < self.compressedTrace[i * 4 + 1, 1]
                    ):
                        lineColor = (0, 204, 153)
                        textPos = 0.10
                    else:
                        lineColor = (204, 0, 255)
                        textPos = 0.75
                    self.event2.append(
                        pg.LinearRegionItem(
                            values=[
                                self.compressedTrace[i * 4 + 2, 0],
                                self.compressedTrace[i * 4 + 3, 0],
                            ],
                            bounds=[
                                self.compressedTrace[i * 4 + 2, 0],
                                self.compressedTrace[i * 4 + 3, 0],
                            ],
                            brush=pg.mkBrush(
                                lineColor + (20,),
                            ),
                            pen=pg.mkPen(lineColor + (120,), width=1.5),
                            movable=False,
                        )
                    )
                    self.event2[i].setZValue(-10)
                    self.p2.addItem(self.event2[i])
                    label = pg.InfLineLabel(
                        self.event2[i].lines[0],
                        "event {}".format(i + 1),
                        color="k",
                        anchors=[(0, 0.75), (0, 0.75)],
                        position=textPos,
                        rotateAxis=(1, 0),
                    )

    def closeWindow(self):
        print("Close button pressed")
        self.stopCalculating.set()
        with self.valueQueue.mutex:
            self.valueQueue.queue.clear()
        with self.resultsQueue.mutex:
            self.resultsQueue.queue.clear()
        self.calculatingProcess.join(0.2)
        # sys.exit(0)

    def __init__(self):
        super(schnelluebersichtgeraet, self).__init__()

        if rc.k == 0:
            self.Ia = [0, 0]
            self.time = [0, 1]
            self.idealTrace = None
            self.compressedTrace = None
            self.samplerate = 2
            self.rc = rc
            fileName = Path("startup.txt")
            self.fileName = str(fileName)
            rc.k = 1

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.resize(1500, 1000)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.p1 = pg.PlotWidget()
        self.p2 = pg.PlotWidget()
        self.oldEnd = 1
        self.oldStart = 0
        self.stopCalculating = threading.Event()
        self.timerCalculating = threading.Event()
        self.valueQueue = Queue()
        self.resultsQueue = Queue()
        self.calculatingProcess = threading.Thread(
            target=self.calculateUpdate,
            args=(
                self.valueQueue,
                self.resultsQueue,
                self.stopCalculating,
                self.timerCalculating,
            ),
            name="calculating_thread",
            daemon=True,
        )
        self.calculatingProcess.start()
        self.timerRegion = threading.Event()
        self.regionProcess = threading.Thread(
            target=self.regionUpdate,
            args=(self.stopCalculating, self.timerRegion),
            name="regionUpdate_thread",
            daemon=True,
        )
        self.regionProcess.start()

        self.IdealTraceButton = QCheckBox("")
        self.IdealTraceButton.setText("Display IdealTrace")
        self.layout.addWidget(self.IdealTraceButton, 3, 1)
        self.IdealTraceButton.setChecked(False)
        self.IdealTraceButton.toggled.connect(self.toggle_trace)

        self.EventShadingButton = QCheckBox("")
        self.EventShadingButton.setText("Display Event Shading")
        self.layout.addWidget(self.EventShadingButton, 4, 1)
        self.EventShadingButton.setChecked(False)
        self.EventShadingButton.toggled.connect(self.toggle_shading)

        # Button to load the data file
        self.loadButton = QPushButton("Select Data File")
        self.layout.addWidget(self.loadButton, 0, 0, 1, 0)
        self.loadButton.clicked.connect(lambda: self.openFileNameDialog())

        # big overview plot in p1
        self.p1.setTitle("Full trace overview")
        self.drawP1()
        self.lr = pg.LinearRegionItem(
            [self.time[-1] / 4.2, self.time[-1] * 2 / 4.2],
            brush=pg.mkBrush((151, 210, 255, 255)),
            pen=pg.mkPen(width=2),
        )
        self.lr.setZValue(-10)
        self.p1.setDownsampling(ds=25, mode="peak")
        self.p1.setClipToView(True)
        self.p1.setDefaultPadding(padding=0.0)
        self.p1.disableAutoRange()
        # self.p1.enableAutoRange(x=False, y=True)
        self.p1.setMouseEnabled(x=True, y=False)
        self.p1.addItem(self.lr)
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        ax0 = self.p1.getAxis("left")
        ax0.setLabel(text="current", units="A", unitPrefix="p")
        ax0.setScale(1e-12)
        # self.p1.autoRange()
        self.p1.setLimits(xMin=self.time[0], xMax=self.time[-1])
        # self.eventRegion1()
        self.layout.addWidget(self.p1, 1, 0, 1, 0)

        # zoomed-in plot in p2
        self.p2.setTitle(
            "Zoom on selected region, I_mean = {s0:0.3f} pA I_std = {s1:0.3f} pA".format(
                s0=np.nan, s1=np.nan
            )
        )

        self.drawP2()
        self.p2.setMouseEnabled(x=True, y=False)
        self.p2.enableAutoRange(x=False, y=True)
        self.p2.disableAutoRange()
        self.p2.setDownsampling(ds=5, auto=True, mode="peak")
        self.p2.setClipToView(True)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p2.setLabel("bottom", text="time", units="s")
        self.p2.setXRange(*self.lr.getRegion(), padding=0)
        self.p2.setLimits(
            xMin=self.time[0],
            xMax=self.time[-1],
            maxXRange=len(self.Ia) * self.samplerate // 4,
        )  # , minXRange = 50 * self.samplerate)
        ax2 = self.p2.getAxis("left")
        ax2.setLabel(text="current", units="A", unitPrefix="p")
        ax2.setScale(1e-12)
        ax2 = self.p2.getAxis("bottom")
        ax2.setLabel(text="time", units="s")
        # self.p2.autoRange()
        # self.p2.setLimits(xMin=self.time[0], xMax=self.time[-1])
        # self.eventRegion2()
        self.layout.addWidget(self.p2, 2, 0)

        # Histogram of zoomed bit in p3
        self.p3 = pg.PlotWidget()
        self.p3.setFixedWidth(250)
        self.p3.setTitle("Histogram of zoom")
        self.updateWidget()
        self.p3.showGrid(x=True, y=True, alpha=0.3)
        ax3 = self.p3.getAxis("left")
        ax3.setLabel(text="current", units="A", unitPrefix="p")
        ax3.setScale(1e-12)
        ax3.setStyle(showValues=False)
        self.p3.setYLink(self.p2)
        ax4 = self.p3.getAxis("bottom")
        ax4.setLabel(text="count", units="points")
        self.layout.addWidget(self.p3, 2, 1)

        # Pushbutton for stuff to be executed on the zoom-selected region.
        self.the_button = QPushButton("Wir schaffen das.")
        self.layout.addWidget(self.the_button, 3, 0)
        self.the_button.resize(100, 32)
        self.the_button.move(50, 0)
        self.the_button.clicked.connect(lambda: self.button_run_handler())
        self.setWindowTitle(str(Path(self.fileName).stem))
        self.lr.sigRegionChanged.connect(self.updatePlot)
        self.p2.sigXRangeChanged.connect(self.updateRegion)
        self.lr.sigRegionChangeFinished.connect(lambda: self.updateWidget())
        self.show()

        app.aboutToQuit.connect(self.closeWindow)


def signal_handler(signal, frame):
    print("exiting")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    app = QApplication([])
    app.setWindowIcon(QIcon(QPixmap("src/media/icon.png")))
    fenster = schnelluebersichtgeraet()  # they'll love us
    if QT_VERSION == 6:
        app.exec()
    else:
        app.exec_()
