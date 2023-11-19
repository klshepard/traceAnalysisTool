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


def test_data_dump(chimera_data_set):
    statsDF = (
        []
    )  # note that this is not an mp managedList, which is what is in the actual func.
    for s in chimera_data_set:
        main.exp_master_func([s, statsDF])

    for s in statsDF:
        assert isinstance(s, list)
