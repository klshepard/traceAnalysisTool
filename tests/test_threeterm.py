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


def test_data_benchvue(threeterm_data_set):
    assert threeterm_data_set[0][2].VA > 2300
    assert threeterm_data_set[0][2].VA < 2600
