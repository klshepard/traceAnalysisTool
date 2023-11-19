from pandas_lib import enclose_the_pandas

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def pandas_the_IV():
    """
    Plots I(V_G) of two-terminal nanopore data, given a dataframe that's tagged with "gatesweep" in the comments.

    Parameters
    ----------
    None:
        Does read the python env in this file, so this is a little sloppy.

    Returns
    -------
    None:
        Will write to file.
    """
    statsDF["run"] = statsDF["exp_name"].str.split("_").str[-2]
    statsDF["run"] = statsDF["run"].astype("category")

    # sort DF for better plotting:
    #    statsDF.sort_values(by=["exp_name"], inplace=True)
    smallDF = statsDF[statsDF["COMMENTS"].str.contains("gatesweep")]
    sample = str(statsDF["SAMPLE"].values[0])

    smallDF = smallDF[
        ~smallDF.exp_name.str.contains("pindowngatesweep")
    ]  # TODO clean this
    smallDF = smallDF[
        ~smallDF.exp_name.str.contains("pindown2gatesweep")
    ]  # TODO clean this
    smallDF = smallDF[
        ~smallDF.exp_name.str.contains("pindown3gatesweep")
    ]  # TODO clean this
    smallDF = smallDF[
        ~smallDF.exp_name.str.contains("pindown4gatesweep")
    ]  # TODO clean this
    smallDF = smallDF[
        ~smallDF.exp_name.str.contains("pindown6gatesweep")
    ]  # TODO clean this
    smallDF = smallDF[
        ~smallDF.exp_name.str.contains("pindown7gatesweep")
    ]  # TODO clean this

    VAC = smallDF["VAC"].iat[0]
    VA = smallDF["VA"].iat[0]

    # smallDF = smallDF[smallDF["rel_bias"] > 0]  # TODO clean this

    # intention above was to clear out the ones I know didn't get set in the run

    print(smallDF)

    print(smallDF.columns)

    print(smallDF.VAC)
    print(smallDF.run)
    print(smallDF.VAG)

    smallDF = smallDF.rename(columns={"VAC": "V"})

    sns.set()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    IVplot = sns.scatterplot(x="VAG", y="I_mean", data=smallDF, hue="V")
    IVplot.errorbar(
        smallDF.VAG,
        smallDF.I_mean,
        yerr=smallDF.I_std,
        elinewidth=1,  # width of error bar line
        ecolor="blue",  # color of error bar
        capsize=0,  # cap length for error bar
        capthick=1,  # cap thickness for error bar
        fmt="none",
        zorder=50,
        alpha=0.5,
    )

    #    the_text = r"$V$ = {su} mV".format(su=VAC) I set V_AC wrong up there, assuming it's one per frame BUG
    # plt.text(
    #     0.9,
    #     1.05,
    #     the_text,
    #     horizontalalignment="center",
    #     verticalalignment="center",
    #     transform=ax.transAxes,
    # )
    plt.xlabel(r"$V_{AG}$ [mV]")
    plt.ylabel(r"$I_{A}$ " + "[{su}]".format(su=smallDF["unit"].iat[0]))

    fileName = sample + "_gateIV.png"
    fig.savefig(savePath / fileName, dpi=600)


#### Main loop
if __name__ == "__main__":

    resultsPath, savePath, statsDF = enclose_the_pandas()
    pandas_the_IV()
