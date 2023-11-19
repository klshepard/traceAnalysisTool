.. schnellstapel documentation master file, created by
   sphinx-quickstart on Thu Jun 18 06:57:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to schnellstapel's documentation!
*****************************************

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
************

.. image:: ../callgraph.png
  :width: 800
  :alt: Callgraph

This codebase is designed to quickly look at data from Chimera Instruments VC100, smFET platform or HEKA USB files.

TODO

Readme File
===========

.. mdinclude:: ../README.md

Architecture Overview
=====================

TODO

Getting started
***************

Try typing "make".

Installation
=============

TODO 

Setup folder
----------------------

Pick a folder on your machine.  This drive will have to contain all the code and all the data, so allocate about a terabyte or so.  If you can dedupe, all the better -- the compression ratio will not be all that high, but worth it.

This will contain two repos: Schnellstapel and DataAndResults.


Clone repositories
----------------------

* strongly recommended to establish ssh key access to github: https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/working-with-ssh-key-passphrases#auto-launching-ssh-agent-on-git-for-windows
* clone data repository: ```git@github.com:schnellstapel/dataAndResults.git```
* clone code repository: ```git@github.com:schnellstapel/schnellstapel.git```

Install Python
---------------

* get latest anaconda installation from https://docs.conda.io/en/latest/miniconda.html
* make sure to have conda installed on local user folder and activated
* create new conda environement:
    * change to parent folder ```cd ./Schnellstapel```
    * run command ```conda create --name schnellstapel --file ./condaSetup/spec-file.txt```

This will give you a conda env named ``schnellstapel`` that you should use to run the code.  As conda does not update after major Python version releases, you should update this manually -- or, as a user, update to the latest, ensure all tests clear, and then push a new spec-file.

Test setup
-----------

We have two types of tests.

* The first is the runtime tests on the data.  To do this, run ```make testData```

* The second are unit tests on the individual pieces of code, down to a function level.  To do this, run ``` make test ```

General structure of code/functionality
====================================================

There are two cardinal abstractions we use, implemented as two classes; the ``Experiment()`` and the ``Slice()``.

An ``Experiment()`` represents a single run of a continuous measurement.  To do this, it concatenates files and syncs them up if the tool writes multiple files.

A ``Slice()`` is fragment of an ``Experiment()`` with steady-state, constant biases on all voltage channels.  This makes eventfinding and averaging over these sensible.

At present, an ``smFET Experiment()`` does not contain all the channels at play in the measurement -- it makes each separate channel a separate ``Experiment()``.  Whether or not this is a bug is debatable.

* code will read all files given in folder which are specified in DataAndResults/data.txt
  * use the git repo https://github.com/schnellstapel/DataAndResults to maintain / specify new data paths

On loading data that's on an anode or cathode channel, the code will filter and downsample it.  This is governed by the ``--freq`` option.  By default, a low-pass FIR filter is used; this can be changed with the ``--filtertype`` option. If filtered lower than device specific cutoff frequency two step filtering is applied -- this comes in at 25 kHz for chimera data, 250 Hz for smFET data.

The data is then downsampled to 4x the sample rate frequency, to drop points that are carry no information.

* Chimera Data:
  * Code will read all files which are present
  * Will assume that all files with the same filename (except of the timestamp in the end) are belonging to one 'Experiment' which is read and processed as one entity
  * Experiments can be sliced up any way -- by default, these are mapped to contstant values using 'benchvue files' for Chimera data, or at constant voltage for IV files on smFET data.
  * one can specify to a subset of files to read only (see cmd line options)

* smFET Data:
  * code will read all files which are present
  * one can specify a channel key word to process further only the respective channel number (see cmd line options)
  * option to slice experiments into slices of given size (--sliceTime XX)

* Experiment()s can be plotted
* Then each experiment is sliced in one or more Slices() (benchvue option, or any other)
* Processing further everything files you have the option to:
  * returns allways aggregated data about each slice
  * plot trace
  * plot powerspectrum and integrated noise
  * find events
  * plot events
  * do hmm modeling (smFET only at this point)

Time-domain analysis
====================

Big numbers


Reduced dataset analysis
========================

In addition to analyzing data that requires the entire time-domain trace, we offer tools for computing over derived values from the time series.

Data Repo
------------

https://github.com/schnellstapel/DataAndResults, contains text files which specify the storage location of data files. Furthermore the repo sets up the appropriate structure for saving the results.


Code
------------

https://github.com/schnellstapel/Schnellstapel

Common workflows
****************


Autodocs
********

Available Command line options
==============================

.. argparse::
    :filename: ../src/utils/parser_utils.py
    :func: cmd_parser
    :prog: src/main


Main
====

.. automodule:: main
    :members:

data_classes
============

.. automodule:: data_classes
    :members:

event_utils
============

.. automodule:: event_utils
    :members:

export_utils
============

.. automodule:: export_utils
    :members:

fft_utils
=========

.. automodule:: fft_utils
    :members:

filter_utils
============

.. automodule:: filter_utils
    :members:

import_utils
============

.. automodule:: import_utils
    :members:

pandas_utils
============

.. automodule:: pandas_utils
    :members:

plot_utils
==========

.. automodule:: plot_utils
    :members:

poreAnalysis_utils
==================

.. automodule:: poreAnalysis_utils
    :members:

pore_current
============

.. automodule:: pore_current
    :members:

setENV_utils
============

.. automodule:: setENV_utils
    :members:

pandas_lib
==========

This is the general library of Pandas utility functions that we rely on for getting the Pandas parts of this to work.

.. automodule:: custom_pandas.pandas_lib
    :members:

pandas_chimera_IV
=================

To do this correctly, we require the input experiments be tagged "IVExp".  The code should then automatically find these and run them to get two-terminal IVs.

.. currentmodule:: custom_pandas.pandas_chimera_IV
.. autofunction:: pandas_the_IV

pandas_chimera_gateIV
=====================

To do this correctly, we require the input experiments be tagged "gatesweep".  The code should then automatically find these and run them to get gate-swept IVs.  TODO

.. currentmodule:: custom_pandas.pandas_chimera_gateIV
.. autofunction:: pandas_the_IV

pandas_chimera_gateeventrate
============================

To do this correctly, we require the input experiments be tagged "DNA".  The code should then automatically find these and run them to get plots of the event rate vs gate voltage. TODO

.. currentmodule:: custom_pandas.pandas_chimera_gateeventrate
.. autofunction:: pandas_the_IV

pandas_chimera_gateeventdwelltime
=================================

To do this correctly, we require the input experiments be tagged "DNA".  The code should then automatically find these and run them to get plots of the dwell time for constant DNA length, presumably, as a function of gate voltage. TODO

.. currentmodule:: custom_pandas.pandas_chimera_gateeventdwelltime
.. autofunction:: pandas_the_IV
  
Contributing
************

TODO

docs

test

black


Coding standards
================
