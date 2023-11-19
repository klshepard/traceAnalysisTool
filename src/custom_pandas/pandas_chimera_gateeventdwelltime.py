from pandas_lib import enclose_the_pandas
from pandas_lib import get_events_files
from pandas_lib import make_events_df

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def pandas_the_IV():
    """
    Plots event dwelltimes, colored by gate voltage, for files that are tagged "DNA".

    TODO, check stuff for automatically sorting out DNA length

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
    smallDF = statsDF[statsDF["COMMENTS"].str.contains("DNA")]
    sample = str(statsDF["SAMPLE"].values[0])

    eventsfiles = get_events_files()
    eventsDF = make_events_df(eventsfiles)

    if not "EVENTTYPE" in eventsDF:
        raise ValueError(
            "You have no EVENTS column in this dataframe. Go run the actual eventfinding."
        )

    eventsDF = eventsDF[
        eventsDF["VAG"].notna()
    ]  # TODO -- not, this still adds to the DF incorrecly!  FIX that

    eventsDF = eventsDF[eventsDF.EVENTDWELLTIME < 5e3]

    eventsDF = eventsDF[eventsDF.VAC < 0]
    eventsDF = eventsDF[eventsDF.EVENTDEPTH < 8000]
    eventsDF = eventsDF[eventsDF.EVENTDEPTH > -17000]
    eventsDF = eventsDF[eventsDF.EVENTDWELLTIME > 5e2]
    eventsDF = eventsDF[eventsDF.EVENTDWELLTIME > 5e2]
    eventsDF = eventsDF[eventsDF.EVENTDEPTH < -5000]

    VA = smallDF["VA"].iat[0]

    # TODO run relative gate bias stuff, not this way by hue and style.  Hue and style should go by  DNA length

    sns.set()
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    # Want event depth on vertical, event dwell time on horisontal, hue by gate
    IVplot = sns.scatterplot(
        x="EVENTDEPTH", y="EVENTDWELLTIME", data=eventsDF, hue="VAG", style="VAC"
    )
    IVplot.set_xscale("log")
    # IVplot.errorbar(
    #     smallDF.vg,
    #     smallDF.I_mean,
    #     yerr=smallDF.I_std,
    #     elinewidth=10,  # width of error bar line
    #     ecolor="k",  # color of error bar
    #     capsize=0,  # cap length for error bar
    #     capthick=1,  # cap thickness for error bar
    #     fmt="none",
    #     zorder=50,
    #     alpha=0.5,
    # )
    plt.ylabel(r"Event dwell time [microsec]")
    plt.xlabel(r"Event depth" + " [{su}]".format(su=statsDF["unit"].iat[0]))
    fileName = sample + "_gateeventscloud.png"
    IVplot.figure.savefig(savePath / fileName, dpi=600)
    del IVplot

    plt.figure(figsize=(16, 8))
    IVplot = sns.scatterplot(
        x="VAG",
        y="EVENTDWELLTIME",
        data=eventsDF,
        hue="VAC",
    )
    # IVplot.errorbar(
    #     smallDF.vg,
    #     smallDF.I_mean,
    #     yerr=smallDF.I_std,
    #     elinewidth=10,  # width of error bar line
    #     ecolor="k",  # color of error bar
    #     capsize=0,  # cap length for error bar
    #     capthick=1,  # cap thickness for error bar
    #     fmt="none",
    #     zorder=50,
    #     alpha=0.5,
    # )
    plt.ylabel(r"Event dwell times [microsec]")
    plt.xlabel(r"$V_{AG}$ [mV]")
    fileName = sample + "_gateeventdwelltimes.png"
    IVplot.figure.savefig(savePath / fileName, dpi=600)

    return


#### Main loop
if __name__ == "__main__":
    resultsPath, savePath, statsDF = enclose_the_pandas()
    pandas_the_IV()
