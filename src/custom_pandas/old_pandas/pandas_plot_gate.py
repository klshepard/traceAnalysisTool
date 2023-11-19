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

results_frame["config"] = config

#### Main loop
if __name__ == "__main__":

    files_for_model = sorted(resultsPath.rglob("*_exponential_modelframe.csv"))
    for ex in results_frame.groupby(["SAMPLE", "config"]):
        df = ex[1][["VAC", "I_mean", "conc"]]
        df["I_mean"] = df["I_mean"] / 1000
        df["VAC"] = df["VAC"] / 1000
        df["I_model"] = np.nan
        # df['vg'] = df['vg']/1000
        # df['vg_halfcell'] = df['VAC']/2-df['vg'] # anode minus gate.
        df["conc"] = df["conc"].apply(
            lambda x: str(x) + " M"
        )  # moronic automatic rescaling for hue by float
        df = df.sort_values(by=["VAC"])
        ex_device = ex[0][0]
        for (
            index,
            row,
        ) in df.iterrows():  # BUG -- aweful and aweful -- pulls in model EVERY TIME...
            ex_conc = row["conc"]
            for name in files_for_model:
                if (
                    name.stem.split("_")[0] == ex_device
                    and name.stem.split("_")[2] == "C"
                ):
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
                    df.at[index, "I_model"] = my_exp(row["VAC"], A, x0, B).to_numpy()[0]
        the_fig, the_ax = plt.subplots(figsize=(8, 8))
        the_ax.grid()
        the_ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        the_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
        the_ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        the_ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        the_ax.grid(which="minor", linewidth="0.5", color="gray", alpha=0.5)
        the_ax.grid(which="major", linewidth="1", color="gray", alpha=0.5)
        sns.scatterplot(x="VAC", y="I_mean", data=df, hue="conc", ax=the_ax)
        sns.lineplot(x="VAC", y="I_model", data=df, hue="conc", ax=the_ax)
        plt.xlabel(r"$V_{AC}$ [V]")
        plt.ylabel(r"$I_{A}$ [nA]")
        fileName = ex[0][0] + "_config_" + str(ex[0][1]) + "_IV_gatefit.png"
        the_fig.savefig(Path.cwd().parent / "ChimeraData/results" / fileName, dpi=300)
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
