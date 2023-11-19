# schnellstapel hello

This tool is for QUICKLY looking at BATCH data from the schnellamt board reader. 

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
