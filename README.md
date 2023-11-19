# Trace Analysis Tool for smFET and nano pore data.

This tool is for QUICKLY looking at BATCH data from smFET devices
The software was developed in the Bioelectronics Systems Lab of Kenneth Shepard at Columbia University.
It allows to analyse large scale time series data of various measurement platforms to analyze single molecular data.

## Dependencies
### `make docs`
Since `make docs` uses LaTeX for formatting, the following packages are required
(outside of conda env). 

```
sudo apt install latexmk
sudo apt install texlive-full
```

Currently, due to a bug in the conda version of `pyan3`, `pyan3` must be 
installed using pip (in the conda env).
```
pip install -r condaSetup/pip-requirements.txt
```
