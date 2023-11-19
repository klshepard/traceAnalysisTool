import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))
import pytest
import os
import multiprocessing

import import_utils
import parser_utils
import filter_utils
import plot_utils
import fft_utils
import import_utils
import event_utils
import data_classes
import main


@pytest.fixture
def threeterm_data_set():
    os.system("cd ../DataAndResults; git checkout test_ThreeTermData")
    the_manager = multiprocessing.Manager()
    the_semaphore = the_manager.Semaphore(1)
    parser = parser_utils.cmd_parser()
    args = parser.parse_args()
    npersegPARAM = 240  # cut the whole thing in to this many segments
    noverlapPARAM = 1.2  # 1.1 indicates 10% overlap
    seg_front_crop = 1
    seg_back_crop = 0.5
    resultsPath = Path.cwd().parent / "DataAndResults/results"
    resultsPath.mkdir(parents=True, exist_ok=True)
    dataPath = Path()

    if import_utils.check_dataPath(resultsPath.parent):
        dataPath = import_utils.get_dataPath(resultsPath.parent)
    else:
        # BUG -- this is ripe for fuckery.  We should just error out on no data.txt
        logging.warning(
            "No data.txt file to read found in {s1}. Assuming data stored in {s1}/dump folder.".format(
                s1=str(resultsPath.parent)
            )
        )
        dataPath = resultsPath.parent / "dump"

    VA = import_utils.get_VA(dataPath)
    rc = data_classes.rc(
        npersegPARAM,
        noverlapPARAM,
        args,
        seg_front_crop,
        seg_back_crop,
        VA,
        dataPath,
        resultsPath,
    )
    experimentList = import_utils.create_experiment_list_Chimera(rc)
    return main.assemble_data(experimentList, the_manager, the_semaphore, rc, args)


@pytest.fixture
def chimera_data_set():
    os.system("cd ../DataAndResults; git checkout test_ChimeraData")
    the_manager = multiprocessing.Manager()
    the_semaphore = the_manager.Semaphore(1)
    parser = parser_utils.cmd_parser()
    args = parser.parse_args()
    npersegPARAM = 240  # cut the whole thing in to this many segments
    noverlapPARAM = 1.2  # 1.1 indicates 10% overlap
    seg_front_crop = 1
    seg_back_crop = 0.5
    resultsPath = Path.cwd().parent / "DataAndResults/results"
    resultsPath.mkdir(parents=True, exist_ok=True)
    dataPath = Path()

    if import_utils.check_dataPath(resultsPath.parent):
        dataPath = import_utils.get_dataPath(resultsPath.parent)
    else:
        # BUG -- this is ripe for fuckery.  We should just error out on no data.txt
        logging.warning(
            "No data.txt file to read found in {s1}. Assuming data stored in {s1}/dump folder.".format(
                s1=str(resultsPath.parent)
            )
        )
        dataPath = resultsPath.parent / "dump"
    VA = import_utils.get_VA(dataPath)
    rc = data_classes.rc(
        npersegPARAM,
        noverlapPARAM,
        args,
        seg_front_crop,
        seg_back_crop,
        VA,
        dataPath,
        resultsPath,
    )
    experimentList = import_utils.create_experiment_list_Chimera(rc)
    return main.assemble_data(experimentList, the_manager, the_semaphore, rc, args)
    # Could yield and teardown here, to clean up back to right git branch.


@pytest.fixture
def smfet_data_set():
    os.system("cd ../DataAndResults; git checkout test_smFETData")
    the_manager = multiprocessing.Manager()
    the_semaphore = the_manager.Semaphore(1)
    parser = parser_utils.cmd_parser()
    args = parser.parse_args()
    npersegPARAM = 240  # cut the whole thing in to this many segments
    noverlapPARAM = 1.2  # 1.1 indicates 10% overlap
    seg_front_crop = 1
    seg_back_crop = 0.5
    resultsPath = Path.cwd().parent / "DataAndResults/results"
    resultsPath.mkdir(parents=True, exist_ok=True)
    dataPath = Path()

    if import_utils.check_dataPath(resultsPath.parent):
        dataPath = import_utils.get_dataPath(resultsPath.parent)
    else:
        # BUG -- this is ripe for fuckery.  We should just error out on no data.txt
        logging.warning(
            "No data.txt file to read found in {s1}. Assuming data stored in {s1}/dump folder.".format(
                s1=str(resultsPath.parent)
            )
        )
        dataPath = resultsPath.parent / "dump"
    VA = import_utils.get_VA(dataPath)
    rc = data_classes.rc(
        npersegPARAM,
        noverlapPARAM,
        args,
        seg_front_crop,
        seg_back_crop,
        VA,
        dataPath,
        resultsPath,
    )
    experimentList = import_utils.create_experiment_list_smFET(rc)
    return main.assemble_data(experimentList, the_manager, the_semaphore, rc, args)
