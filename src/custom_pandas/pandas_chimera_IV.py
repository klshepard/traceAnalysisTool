from pandas_lib import enclose_the_pandas, calculate_conductivity

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def pandas_the_IV():
    """
    Plots I(V_AC) of two-terminal nanopore data, given a dataframe that's tagged with "IVExp" in the comments.

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
    statsDF.sort_values(by=["exp_name"], inplace=True)
    smallDF = statsDF[statsDF["COMMENTS"].str.contains("IVExp")]
    sample = str(statsDF["SAMPLE"].values[0])

    print(smallDF["COMMENTS"])

    sns.set()
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 7))
    IVplot = sns.scatterplot(
        x="VAC", y="I_mean", data=smallDF, hue="COMMENTS", style="conc"
    )
    IVplot.errorbar(
        smallDF.VAC,
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
    V = smallDF.VAC.unique() * 1e-3
    d = 5e-9
    l = 16e-9
    sigma_surf = 0.01
    c = 1
    ideal_voltages, ideal_currents = calculate_conductivity(V, d, l, c, sigma_surf)
    IVplot.plot(ideal_voltages, ideal_currents, "x-", alpha=0.25, color="r")
    del c
    c = 0.1
    ideal_voltages, ideal_currents = calculate_conductivity(V, d, l, c, sigma_surf)
    IVplot.plot(ideal_voltages, ideal_currents, "o-", alpha=0.25, color="r")
    plt.xlabel(r"$V_{AC}$ [mV]")
    plt.ylabel(r"$I_{A}$ " + "[{su}]".format(su=statsDF["unit"].iat[0]))
    fileName = sample + "_baseIV.png"
    IVplot.figure.savefig(savePath / fileName, dpi=600)


#### Main loop
if __name__ == "__main__":
    resultsPath, savePath, statsDF = enclose_the_pandas()
    pandas_the_IV()
