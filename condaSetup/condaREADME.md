## Create new conda environment to run schnellstapel
make sure to have conda installed on local user folder and activated

use conda-forge as channel for installation:
```conda config --add channels conda-forge```
```conda config --set channel_priority strict```

change to parent folder 
```cd ./Schnellstapel```

run command
```conda create --name schnellstapel --file ./condaSetup/spec-file.txt```
