from src import schnelluebersicht

print(schnelluebersicht.args)
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import argparse

# see here: https://github.com/pytest-dev/pytest-qt/#pytest-qt


def test_parser(qtbot):
    assert isinstance(schnelluebersicht.args, argparse.Namespace)


def test_button(qtbot):
    widget = schnelluebersicht.schnelluebersichtgeraet()
    widget.Filename = "tests/test.pckl"
    widget.rc.args.file = "tests/test.pckl"
    widget.getData()
    widget.closeWidgets()
    widget.reloadWidgets()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.the_button, Qt.MouseButton.LeftButton)
    assert widget.the_button.text().startswith("Executed in")
