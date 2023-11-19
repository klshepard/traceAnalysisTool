# import custom utils
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src" / "utils"))
sys.path.append(str(Path.cwd() / "src" / "classes"))
sys.path.append(str(Path.cwd() / "src"))
import import_utils
from pandas_utils import *

start_time = time.time()
logging.info("Start run...")

# Clear previous results -- this one always keeps, and just adds to the same place...
logging.info("pandas_model is saving the previous run to resultsArchive")
resultsStoragePath = Path.cwd().parent / "DataAndResults/resultsArchive"
outputPlace = "AnalysisOutput_" + datetime.now().strftime("%Y%m%d_%H%M%S")
distutils.dir_util.copy_tree(
    Path.cwd().parent / "DataAndResults" / "results",
    str(resultsStoragePath / outputPlace),
)

config = []
for row in results_frame["COMMENTS"]:
    if "configA" in row:
        config.append("A")
    elif "configB" in row:
        config.append("B")
    elif "configC" in row:
        config.append("C")
    else:
        config.append(np.nan)

# results_frame here drops into the namespace after from pandas_utils import *
results_frame["config"] = config

for ex in results_frame.groupby(["SAMPLE", "config"]):
    model_frame = pd.DataFrame(columns=["device", "config", "conc", "A", "x0", "B"])
    df = ex[1][["VAC", "I_mean", "conc"]]
    df["I_mean"] = df["I_mean"] / 1000
    df["VAC"] = df["VAC"] / 1000
    df["conc"] = df["conc"].apply(lambda x: str(x) + " M")
    ex_device = ex[0][0]
    for thing in df.groupby(["conc"]):
        ex_conc = str(thing[0])
        fit_frame = pd.DataFrame(thing[1])
        fit_frame = fit_frame.sort_values(by=["VAC"])
        if len(fit_frame) > 3:
            voltages = fit_frame["VAC"].to_numpy()
            amperages = fit_frame["I_mean"].to_numpy()
            fp = fit_single_exponent(voltages, amperages)
            assert len(fp) == 3
            config = ex[0][1]
            model_frame = model_frame.append(
                pd.DataFrame(
                    {
                        "device": [ex_device],
                        "config": config,
                        "conc": [ex_conc],
                        "A": [fp[0]],
                        "x0": [fp[1]],
                        "B": [fp[2]],
                    }
                )
            )
    save_name = ex[0][0] + "_config_" + ex[0][1] + "_exponential_modelframe.csv"
    model_frame.to_csv(Path.cwd().parent / "ChimeraData/results" / save_name)
