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
        ["sample", "conc", "COMMENTS", "VAC"]
    ):
        df = ex[1][["VAC", "I_mean", "I_gate", "VAG"]]
        r = df.shape[0]
        if r < 12:
            continue
        df["I_mean"] = df["I_mean"] / 1000
        df["I_gate"] = df["I_gate"] / 1000
        df["VAC"] = df["VAC"] / 1000
        df["VAG"] = df["VAG"] / 1000
        df["vg_halfcell_B"] = -df["VAC"] / 2 - df["VAG"]
        df["vg_halfcell_C"] = (
            df["VAC"] / 2 - df["VAG"]
        )  # anode minus gate, in silentside mode
        df["vg_cellsum"] = (
            df["vg_halfcell_C"] - df["vg_halfcell_B"]
        )  # sanity check, these should be VAC
        df["I_model_B"] = np.nan
        df["I_model_C"] = np.nan
        df = df.sort_values(by=["VAG"])
        ex_conc = str(ex[0][1]) + " M"
        ex_device = ex[0][0]
        for name in files_for_model:
            mod_device = name.stem.split("_")[0]
            if mod_device == ex_device and name.stem.split("_")[2] == "B":
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
                    df.at[index, "I_model_B"] = my_exp(
                        row["vg_halfcell_B"], A, x0, B
                    ).to_numpy()
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
                    df.at[index, "I_model_C"] = my_exp(
                        row["vg_halfcell_C"], A, x0, B
                    ).to_numpy()
        the_fig, the_ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
        gs = matplotlib.gridspec.GridSpec(7, 1)
        ax_gate = plt.subplot(gs[0:4, 0])
        ax_anod = plt.subplot(gs[5:7, 0])
        print(df)
        ax_gate.grid()
        ax_gate.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax_gate.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax_gate.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
        ax_gate.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
        ax_gate.set_xlabel(r"$V_{AG}$")
        ax_gate.set_ylabel(r"$I_{G}$")
        ax_anod.grid()
        ax_anod.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax_anod.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax_anod.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax_anod.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        ax_anod.set_xlabel(r"$V_{AG}$")
        ax_anod.set_ylabel(r"$I_{A}$")
        ax_gate.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        ax_gate.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        ax_anod.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        ax_anod.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        ax_gate.plot(df["VAG"], df["I_gate"], "x", label="$I_{gate}$ [nA]", alpha=0.8)
        ax_gate.plot(df["VAG"], df["I_model_B"], ".", label="$I_{B}$ [nA]", alpha=0.8)
        ax_gate.plot(df["VAG"], df["I_model_C"], ".", label="$I_{C}$ [nA]", alpha=0.8)
        ax_gate.plot(
            df["VAG"],
            [b + c for b, c in zip(df["I_model_B"], df["I_model_C"])],
            ".",
            label="Sum $I_{leak}$ [nA]",
            alpha=0.8,
        )
        ax_anod.plot(df["VAG"], df["I_mean"], "x", label="$I_{A}$ [nA]", alpha=0.8)
        ax_anod.plot(
            df["VAG"],
            df["I_mean"] - df["I_model_C"],
            ".",
            label="$I_{A}-I_{leak}$ [nA]",
            alpha=1,
        )
        fileName = (
            ex[0][0]
            + "_salt_"
            + str(ex[0][1])
            + "_MKCl_"
            + ex[0][2]
            + "_VAC_"
            + str(ex[0][3])
            + "_IVg_bothfit.png"
        )

        ax_gate.legend()
        ax_anod.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            borderaxespad=0,
            frameon=False,
        )
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
