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

#### Main loop
if __name__ == "__main__":

    files_for_model = sorted(resultsPath.rglob("*_modelframe.csv"))
    for ex in results_frame[results_frame["exp_name"].str.contains("pindown")].groupby(
        ["SAMPLE", "conc", "COMMENTS", "VAC"]
    ):
        df = ex[1][["VAC", "I_mean", "I_gate", "VAG"]]
        r = df.shape[0]
        if r < 12:
            continue
        df["I_mean"] = df["I_mean"] / 1000
        df["I_gate"] = df["I_gate"] / 1000
        df["VAC"] = df["VAC"] / 1000
        df["VAG"] = df["VAG"] / 1000
        df["I_model"] = np.nan
        ## VOLTFIX REQ
        df["vg_halfcell"] = df["VAC"] / 2 - df["VAG"]  # anode minus gate.
        #        df = df.sort_values(by=['vg_halfcell'])
        ex_conc = str(ex[0][1]) + " M"
        ex_device = ex[0][0]
        for name in files_for_model:
            mod_device = name.stem.split("_")[0]
            if mod_device == ex_device and name.stem.split("_")[2] == "C":
                logging.info("Found a model to load.")
                model = pd.read_csv(name, usecols=[1, 2, 3, 4, 5, 6])
                A = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["A"]
                x0 = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["x0"]
                B = model.loc[
                    (model["device"] == ex_device) & (model["conc"] == ex_conc)
                ]["B"]
                for index, row in df.iterrows():
                    df.at[index, "I_model"] = my_exp(
                        row["vg_halfcell"], A, x0, B
                    ).to_numpy()
        the_fig, the_ax = plt.subplots(figsize=(8, 8))
        the_ax.grid()
        plt.ylim(-30, 10)
        plt.xlim(-0.05, 0.4)
        the_ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        the_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        the_ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        the_ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        the_ax.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        the_ax.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        the_ax.plot(
            df["VAG"], df["I_model"], ".", label="Model $I_{leak}$ [nA]", alpha=0.5
        )
        the_ax.plot(
            df["VAG"],
            df["I_mean"] - df["I_model"],
            ".",
            label="$I_{pore}$ [nA]",
            alpha=0.5,
        )
        the_ax.plot(df["VAG"], df["I_gate"], ".", label="$I_{gate}$ [nA]", alpha=0.5)
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "_MKCl_"
            + ex[0][2]
            + "_VAC_"
            + str(ex[0][3])
            + "_IVg_subtraction.png"
        )
        plt.xlabel(r"$V_{AG}$")
        the_ax.legend()
        the_fig.savefig(Path.cwd().parent / "ChimeraData/results" / fileName, dpi=600)
        plt.close()

    # Save the frame with the addition of computed quantities...
    dataframe_name = (
        "dataframe_pandas_atruntime_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".csv"
    )
    df_name = Path.cwd().parent / "ChimeraData/results" / dataframe_name
    results_frame.to_csv(df_name, encoding="utf-8")

    ## copy data to jakobs home on server if running on /space linracks
    ## only for Jakob - do not remove
    if "space" in str(Path.cwd()):
        distutils.dir_util.copy_tree(
            Path.cwd().parent / "ChimeraData" / "results",
            "/u7/jakobuchheim/Repositories/LinrackData/linrackAnalysisOutput"
            + datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
logging.info("Run complete...")
logging.info("--- runtime was %s seconds ---" % (time.time() - start_time))
