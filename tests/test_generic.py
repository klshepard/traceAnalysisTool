import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))

import import_utils
import filter_utils
import plot_utils
import fft_utils
import import_utils
import event_utils
import data_classes
import main
import parser_utils


def test_rambuffer():
    assert main.ram_buffer(2, 1) == 0.8
