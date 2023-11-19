FREQ=10000

USER=$(shell whoami)

BASHRCPATH=~/.bashrc

ifeq ($(USER), boyan)
EMAIL=bip2102@columbia.edu
BASHRCPATH=/users/boyan/.bashrc
endif

ifeq ($(USER), efy1)
EMAIL=efy1@columbia.edu
BASHRCPATH=/users/efy1/.bashrc
endif

ifeq ($(USER), jakobuchheim)
EMAIL=jb4309@columbia.edu
BASHRCPATH=/users/jakobuchheim/.bash_profile
endif

ifeq ($(USER), dlynall)
EMAIL=dgl2129@columbia.edu
BASHRCPATH=/users/dlynall/.bashrc
endif

ifdef LINRACK # if you would like to run on a specific linrack
	NODE = -w linrack$(LINRACK)
else
	NODE =
endif

ifdef IMEM
	MAXMEM = $(IMEM)
else
	MAXMEM = 200000 # In MB.
endif

ifdef WAIT
	WAITLIST = -d afterany:$(WAIT)
else
	WAITLIST =
endif

ifdef CCMD
	CCMD = $(CCMD)
else
	CCMD = make
endif

ifdef ITIME
	MAXTIME = -t $(ITIME):0
else
	MAXTIME = -t 3600:0
	ITIME = 3600
endif

ifdef ITHREADS
	THREADS = $(ITHREADS)
else
	THREADS = 12
# This is the only place THREADS is set.
endif

MEM = $(shell echo ${MAXMEM} / ${THREADS} | bc)

ifdef BATCH
	RUNBATCH = sbatch\
			-n 1\
			-c $(THREADS)\
			--mem-per-cpu $(MEM)\
			-p charmander\
			--qos ks_distributedcomputing\
			$(WAITLIST)\
			$(NODE)\
			$(MAXTIME)\
			-x linrack3,linrack2,linrack4\
			--job-name schnellstapel\
			-o slurm.txt\
			--mail-type=ALL\
			--mail-user=$(EMAIL)\
			--wrap "source $(BASHRCPATH); conda deactivate; unset PYTHONPATH; conda activate schnellstapel;

	CLOSEBATCH = || exit 91; mv slurm.txt ../DataAndResults/results/ ;\
			mail -s 'Slurm Results of $(SLURM_JOB_ID) on $(SLURMD_NODENAME)' $(EMAIL) < ../DataAndResults/results/slurm.txt;" 

else ifdef BATCHPI
	RUNBATCH = sbatch\
			-n 1\
			-c $(THREADS)\
			--mem-per-cpu $(MEM)\
			-p pikachu\
			--qos ks_lowprio\
			$(WAITLIST)\
			$(NODE)\
			$(MAXTIME)\
			-x linrack3,linrack2,linrack4\
			--job-name schnellstapel\
			-o slurm.txt\
			--mail-type=ALL\
			--mail-user=$(EMAIL)\
			--wrap "source $(BASHRCPATH); conda deactivate; unset PYTHONPATH; conda activate schnellstapel;
	CLOSEBATCH = || exit 91; mv slurm.txt ../DataAndResults/results/ ;\
			mail -s 'Slurm Results of $(SLURM_JOB_ID) on $(SLURMD_NODENAME)' $(EMAIL) < ../DataAndResults/results/slurm.txt;" 

else
		RUNBATCH =
		CLOSEBATCH =
endif

export MAKEFLAGS="-j $(THREADS)"

#These are the invocations to get the fixed mkl issues set -- should also be in .bashrc
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC="FALSE"

PY_FILES := $(shell find `pwd -P`/src -name '*.py')

.PHONY: docs callgraph.png grepcheck

uebersicht:
	python src/schnelluebersicht.py

run:
	$(RUNBATCH) python src/main.py -t $(THREADS) --instrumentType Chimera --segplot --expplot --fitTransients --fitPore -f $(FREQ) $(CLOSEBATCH)

interactive: # get on squirtle with four threads, so you can debug
	echo "This will die after 8 hours."
	srun --pty --x11 -N 1 -c 1 -n 4 --qos ks_interactivecomputing bash

data:
	python src/main.py --data

export:
	$(RUNBATCH) python src/main.py -f $(FREQ) -t $(THREADS) --export_slice --instrumentType smFET $(CLOSEBATCH)

fit:
	python src/main.py -t $(THREADS) --fitTransients --fitPore --keep

segplot:
	$(RUNBATCH) python src/main.py -t $(THREADS) --fitTransients --fitPore --segplot --keep -f $(FREQ) --instrumentType Chimera $(CLOSEBATCH)

expplot:
	$(RUNBATCH) python src/main.py -t $(THREADS) --fitTransients --fitPore --instrumentType Chimera --expplot --keep -f $(FREQ) $(CLOSEBATCH)

dataframe:
	python src/main.py -t $(THREADS)

findevents:
	$(RUNBATCH) python src/main.py --findevents -t $(THREADS) --keep --PDF 6 --PDFreversal 3 --avwindow 0.1 --eventType both --alphaError 0.1 --passes 6 -f $(FREQ) --plotidealevents 5 && python src/main.py --plotevents -t $(THREADS) --keep --PDF 6 --PDFreversal 3 --eventType both --alphaError 0.1 --avwindow 0.1 --passes 6 -f $(FREQ) $(CLOSEBATCH)

pickles: # pickes all the datasets, so you can use Schnellubersicht on all of them.
	$(RUNBATCH) python src/main.py -t $(THREADS) --picklesegment -f $(FREQ) --instrumentType Chimera $(CLOSEBATCH)

fakeevents:
	python src/main.py --testevents -t $(THREADS)
	python src/main.py --plotevents -t $(THREADS) -- keep

grepcheck:
	find . -type d -name __pycache__ -exec rm -r {} \+ # clear all __pycache__
	grep -r "BUG" src 	# Marking potential bugs
	grep -r "BUG?" src 	# Marking potential bugs
	grep -r "pass" src 	# Did you forget to remove a debugging pass?
	grep -r "exit" src 	# Did you forget to remove a debuggin exit?
	grep -r "print" src  	# By convention, we use logging, and not print, so these are "odd"

condalist:
	conda list --explicit > condaSetup/spec-file.txt


test: grepcheck# relies on canonical paths in the DataAndResults repo, so only runs on the servers.
	-rm .coverage
	PYTHONPATH=$(pwd) pytest -v --cov=. -n $(THREADS) --cov-append -x tests/test_generic.py # and add any other generic tests here -- ones that don't rely on state and explicit data.
	make pytest_threeterm
	make pytest_twoterm
	make test_uebersicht


pytest_threeterm:
	cd ../DataAndResults; git fetch --all; git checkout test_ThreeTermData; git pull
	PYTHONPATH=$(pwd) pytest -v --cov=. -n 10 -x tests/test_threeterm.py

pytest_twoterm:
	cd ../DataAndResults; git fetch --all; git checkout test_ChimeraData; git pull
	PYTHONPATH=$(pwd) pytest -v --cov=. -n 10 -x tests/test_twoterm.py

test_uebersicht:
	pytest --cov=src/schnelluebersicht.py -n $(THREADS) tests/test_uebersicht.py

callgraph.png:
	pyan3 $(PY_FILES) --uses --no-defines --grouped --colored --dot > callgraph.dot
	unflatten -f -l 2 callgraph.dot | sfdp -Tpng -Gdpi=600 -Gsize=8,10\! -Gratio=fill -o callgraph.png

snakeviz:
	python -m cProfile -o schnell_profile.prof src/main.py --expplot --segplot --findevents --PDF 4 --PDFreversal 2 --avwindow 0.05 --passes 6 -f $(FREQ) --kvalueanalysis
	snakeviz schnell_profile.prof

profile:
	python -m cProfile -o cprofile.cprof src/main.py --fitTransients --fitPore --keep --serial
	pyprof2calltree -k -i myscript.cprof
	mprof run --include-children --multiprocess src/main.py --fitTransients --fitPore -t $(THREADS) --keep
	mprof plot --output profile.png

profile_all:
	python -m cProfile -o cprofile.cprof src/main.py --fitTransients --fitPore --expplot --segplot --keep --serial
	mprof run --include-children --multiprocess src/main.py --fitTransients --fitPore --expplot --segplot -t $(THREADS) --keep
	mprof plot --output profile.png

std_profile:
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all; git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout test_ChimeraData;
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;
	mprof run --include-children --multiprocess src/main.py -t $(THREADS) --expplot --segplot --findevents --PDF 4 --PDFreversal 2 --avwindow 0.02 --passes 6 -f $(FREQ) --ramcheck --kvalueanalysis
	mprof plot --output ../DataAndResults/results/std_profile.png --backend agg
	mv mprofile_*.dat ../DataAndResults/results/

ipython:
	ipython -i src/main.py --

befehl:
	python src/main.py --help

serial:
	$(RUNBATCH) python src/main.py --serial --segplot --expplot --fitTransients --fitPore $(CLOSEBATCH)

clean:
	rm -rf ./src/utils/__pycache__
	rm -rf ./src/classes/__pycache__
	rm -rf ./src/docs/__pycache__
	rm -rf ./src/custom_pandas/__pycache__
	rm -rf ./tests/__pycache__
	$(MAKE) -C ./docs/ clean

testData: # these are explicit, and not dependencies, since we don't want -j 8 to run them in parallel  Probably a better way of doing all this is to use a cleaner pytest fixture to generate the right state, but this conflicts with the way we manage data files right now.  TODO may be ready to go out
	make testThreeTermData
	make testChimeraData
	make testsmFETData

testThreeTermData:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all; git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout test_ThreeTermData;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	cd ../DataAndResults; git fetch --all; git checkout test_ThreeTermData; git pull
	python src/main.py -t $(THREADS) -f $(FREQ) --filterType Bessel --fitTransients --fitPore --expplot --segplot --findevents --PDF 4 --PDFreversal 2 --avwindow 1.0 --passes 2
	$(CLOSEBATCH)

testChimeraData:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all; git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout test_ChimeraData;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) --expplot --segplot --findevents --PDF 4 --PDFreversal 2 --avwindow 0.02 --passes 6 -f $(FREQ) --ramcheck --kvalueanalysis \
	$(CLOSEBATCH)

testsmFETData:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all; git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout test_smFETData;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t 2 -f 100 --segplot --findevents --alphaError 1e-5 --PDF 2.3 --PDFreversal 0.8 --avwindow 2.0 --eventType blockage --instrumentType smFET --plotidealevents 10 --mineventdepth 10 --passes 4 --HistBaseline --kvalueanalysis \
	$(CLOSEBATCH)

testsmFETHMM:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all; git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout test_smFETData;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t 4 --blcorrectionhmm --twostephmm --fixedDistributions --hmm three -f 100 --instrumentType smFET --sliceTime 60 --plotidealevents 10 --PDF 1.3 --PDFreversal 1.0  \
	$(CLOSEBATCH)

docs: callgraph.png
	$(MAKE) -C docs latex
	$(MAKE) -C docs latexpdf

datarepomake:
	$(RUNBATCH)\
	make THREADS=$(THREADS) FREQ=$(FREQ) -f ../DataAndResults/custom_make $(CCMD) \
	$(CLOSEBATCH)

black:
	black src
	black tests

jupytermake:
	$(RUNBATCH)\
	$(CCMD)\
	$(CLOSEBATCH)

slurmalloc:
	sbatch 	-n 1 \
	-c $(THREADS) \
	--mem-per-cpu $(MEM) \
	-p charmander \
	--qos ks_distributedcomputing\
	$(WAITLIST) \
	$(NODE) \
	$(MAXTIME) \
	--job-name submit \
	-o slurm.txt \
	--mail-type=ALL \
	--mail-user=$(EMAIL) \
	--wrap "source $(BASHRCPATH); conda activate schnellstapel;\
		python src/slurmwait.py --time $(ITIME) \
		|| exit 91; \
		mv slurm.txt ../DataAndResults/results/ ;\
		mail -s 'Slurm Results' $(EMAIL) < ../DataAndResults/results/slurm.txt;"
slurmalloc2:
	$(RUNBATCH)\
	python src/slurmwait.py \
	$(CLOSEBATCH)

#####  BELOW HERE, options are user-specific   ######
#### these makfile invocations are user-specific until we decide to make them official, above the line.

smfet_slicefromfile:
	$(RUNBATCH) python src/main.py -t $(THREADS) -f $(FREQ) --expplot --segplot --instrumentType smFET --slicefromfile $(CLOSEBATCH)

exportPickleData:
	$(RUNBATCH) python src/main.py -t 16 -f 100 --segplot --picklesegment --sliceTime 60 --eventType blockage --instrumentType smFET $(CLOSEBATCH)

PCPW02-G-C6-4_IV:
	python src/main.py -t 8 --segplot --fit -f 100000

yoonheeData:
	$(RUNBATCH) python src/custom_pandas/pandas_dwellTime_analysis_eventData.py -t 16 $(CLOSEBATCH)

batchtest:
	$(RUNBATCH) python src/main.py -t $(THREADS) --segplot --fit -f 100000 $(CLOSEBATCH)

PCPW02-G-C6-4_find:
	$(RUNBATCH)\
	python  src/main.py -t $(THREADS) --segplot -f 25000 --findevents --alphaError 1e-3\
						 --PDF 5.0 --PDFreversal 3.0  --eventType both --instrumentType Chimera\
						 --plotidealevents 1 --passes 4 --avwindow 0.2 --mineventdepth 180 --fileNumber 70 \
	$(CLOSEBATCH)

ovaldetail:
	$(RUNBATCH)\
	python  src/main.py -t $(THREADS) --segplot -f 100000 --findevents --alphaError 1e-3 --fileNumber 20\
						 --PDF 6.0 --PDFreversal 3.0  --eventType enhance --instrumentType Chimera\
						 --passes 4 --avwindow 0.2 --mineventdepth 1000 --plotevents --HistBaseline --plotidealevents 0.25 \
	$(CLOSEBATCH)

PCPW03-B-D1-1:
	$(RUNBATCH)\
	python  src/main.py -t $(THREADS) --segplot -f 200000 --findevents --alphaError 1e-3\
						 --PDF 5.0 --PDFreversal 3.0  --eventType both --instrumentType Chimera\
						 --plotidealevents 2 --passes 4 --avwindow 0.2 --mineventdepth 250 --fileNumber 100 --plotevents \
	$(CLOSEBATCH)

batchfind:
	$(RUNBATCH) make PCPW02-G-C6-4_find;$(CLOSEBATCH)

slurm_batchfind:
	sbatch -n 16 -c 1 --mem-per-cpu 6250 -p bulbasaur -w linrack7 --job-name batchfind -o slurm.txt --mail-type=END --mail-user=$(EMAIL) --wrap "export NUMEXPR_MAX_THREADS=1; python src/main.py --instrumentType smFET --segplot --expplot; mv slurm.txt ../DataAndResults/results/ ; mail -s 'Slurm Results' $(EMAIL) < ../DataAndResults/results/slurm.txt;"

nbatchfind:
	sbatch -n 16 -c 1 --mem-per-cpu 6250 -p bulbasaur -w linrack7 --job-name batchfind -o slurm.txt --mail-type=END --mail-user=$(EMAIL) --wrap "export NUMEXPR_MAX_THREADS=1; make PCPW02-G-C6-4_find; mv slurm.txt ../DataAndResults/results/ ; mail -s 'Slurm Results' $(EMAIL) < ../DataAndResults/results/slurm.txt;"

.PHONY : pandas_model pandas_gates pandas_fits pandas_fits_BC pandas_all

pandas: pandas_fits_BC pandas_gates pandas_fits src/utils/pandas_utils.py pandas_null

pandas_all_files: pandas_fits_BC pandas_null pandas_gates pandas_fits

pandas_gates: pandas_model
	python src/custom_pandas/pandas_plot_gate.py

pandas_fits: pandas_model
	python src/custom_pandas/pandas_plot_fits.py

pandas_fits_BC: pandas_model
	python src/custom_pandas/pandas_plot_fits_BC.py

pandas_all:
	python src/custom_pandas/pandas_plot.py

pandas_null:
	python src/custom_pandas/pandas_plot_null.py

pandas_model:
	python src/custom_pandas/pandas_model.py

pandas_IV:
	python src/custom_pandas/pandas_chimera_IV.py

dataframe_smfet:
	$(RUNBATCH) python src/main.py -t $(THREADS) -f $(FREQ) --instrumentType smFET --segplot --findevents --slicefromfile $(CLOSEBATCH)

smfet_events:
	$(RUNBATCH) python src/main.py -t $(THREADS) -f $(FREQ) --segplot --slicefromfile --findevents --alphaError 1e-3 --PDF 3 --PDFreversal 0.5 --avwindow 5 --eventType both --instrumentType smFET --plotidealevents 10 --mineventdepth 5 --passes 6 --HistBaseline --kvalueanalysis $(CLOSEBATCH)

smfet_events_slices:
	$(RUNBATCH) python src/main.py -t $(THREADS) -f $(FREQ) --sliceTime 60 --segplot --findevents --alphaError 1e-5 --PDF 4 --PDFreversal 2 --avwindow 2.0 --eventType both --instrumentType smFET --plotidealevents 60 --mineventdepth 10 --passes 6 --HistBaseline --kvalueanalysis $(CLOSEBATCH)

waveletd4hmm:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_three_wavelet_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_linear_wavelet_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_four_wavelet_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm slowandfast --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_slowandfast_wavelet_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_two_wavelet_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm sixstate --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_sixstate_wavelet_hmm \
	$(CLOSEBATCH)

waveletd4hmm150s:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_three_wavelet_hmm_150s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_linear_wavelet_hmm_150s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_four_wavelet_hmm_150s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_two_wavelet_hmm_150s \
	$(CLOSEBATCH)

waveletd4hmm58s:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_three_wavelet_hmm_58s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_linear_wavelet_hmm_58s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_four_wavelet_hmm_58s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_two_wavelet_hmm_58s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm slowandfast --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_slowandfast_wavelet_hmm_58s;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm new4 --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_new4_wavelet_hmm_58s \
	$(CLOSEBATCH)

d4hmm58s:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_three_hmm_58s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_linear_hmm_58s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_four_hmm_58s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_two_hmm_58s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm slowandfast --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_slowandfast_hmm_58s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm new4 --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_new4_hmm_58s_200Hz \
	$(CLOSEBATCH)

d4hmm150s:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_three_hmm_150s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_linear_hmm_150s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_four_hmm_150s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_two_hmm_150s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm slowandfast --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_slowandfast_hmm_150s_200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm new4 --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --saveHMM --exportEventTrace --reversepolarity;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_4_new4_hmm_150s_200Hz \
	$(CLOSEBATCH)


B4SIMhmm:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD4HMM3SIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmYoonheeData/simulateHMMData/Device_4_three_SIMULATEhmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD4HMMLSIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmYoonheeData/simulateHMMData/Device_4_linear_SIMULATEhmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD4HMM4SIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmYoonheeData/simulateHMMData/Device_4_four_SIMULATEhmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD4HMM2SIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1000 --wavelet rbio1.1 --waveletthreshold 10 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.0 --avwindow 2.5 --mineventdepth 12\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmYoonheeData/simulateHMMData/Device_4_two_SIMULATEhmm \
	$(CLOSEBATCH)

D5hmm:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device5;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm new4 --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_new4_hmm;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_linear_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_four_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_two_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm slowandfast --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_slowandfast_hmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm sixstate --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.3 --PDFreversal 1.2 --avwindow 0.5 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 150 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_sixstate_hmm \
	$(CLOSEBATCH)
	
D5hmmBW:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device5;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 100 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm100Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm400Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 800 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm800Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 1600 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm1600Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 3200 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm3200Hz;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 6400 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 1.3 --avwindow 0.1 --mineventdepth 14\
			--instrumentType smFET --plotidealevents 10 --sliceTime 300 --fitfraction 0.8 --exportEventTrace --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_hmm6400Hz \
	$(CLOSEBATCH)


B5SIMhmm:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD5HMM3SIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm three --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.5 --PDFreversal 0.9 --avwindow 0.5 --mineventdepth 13\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8 --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_three_SIMULATEhmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD5HMMLSIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm linear --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.5 --PDFreversal 0.9 --avwindow 0.5 --mineventdepth 13\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8 --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_linear_SIMULATEhmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD5HMM4SIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.5 --PDFreversal 0.9 --avwindow 0.5 --mineventdepth 13\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8 --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_four_SIMULATEhmm;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout yD5HMM2SIM;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 400 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.5 --PDFreversal 0.9 --avwindow 0.5 --mineventdepth 13\
			--instrumentType PCKL --plotidealevents 10 --fitfraction 0.8 --saveHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/hmmFINAL/Device_5_two_SIMULATEhmm \
	$(CLOSEBATCH)

d4hmmforManus:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm four --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --picklesegment --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --normalizeHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/HMM_for_manuscript/Device_4_four_hmm_58s_final;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --normalizeHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/HMM_for_manuscript/Device_4_two_hmm_58s_final \
	$(CLOSEBATCH)

d4hmm58s_2s:
	$(RUNBATCH)\
	git --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ fetch --all;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ checkout  -f yoonheeData_device4;\
	git  --git-dir=../DataAndResults/.git --work-tree=../DataAndResults/ pull;\
	python src/main.py -t $(THREADS) -f 200 --kvalueanalysis\
			--hmm two --gmmdistributionfit  --twostephmm --fixedDistributions  --eventType blockage\
			--blcorrectionhmm --PDF 1.4 --PDFreversal 0.9 --avwindow 2.5 --mineventdepth 17\
			--instrumentType smFET --plotidealevents 10 --sliceTime 58 --fitfraction 0.8 --saveHMM --exportEventTrace --normalizeHMM;\
	cp ./slurm.txt ../DataAndResults/results/;\
	mv ../DataAndResults/results/ /proj/jakobuchheim/share/HMM_for_manuscript/Device_4_two_hmm_58s_newBLR \
	$(CLOSEBATCH)