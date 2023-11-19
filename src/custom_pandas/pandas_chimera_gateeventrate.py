from pandas_lib import enclose_the_pandas
from pandas_lib import get_events_files
from pandas_lib import make_events_df

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def pandas_the_IV(resultsPath, savePath, statsDF):
    """
    Plots the gate event rate as a function of gate bias on the device.

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

    del statsDF

    VAC = smallDF["VAC"].iat[0]
    VA = smallDF["VA"].iat[0]

    eventsfiles = get_events_files()
    eventsDF = make_events_df(eventsfiles)

    if not "EVENTTYPE" in eventsDF:
        raise ValueError(
            "You have no EVENTS column in this dataframe. Go run the actual eventfinding."
        )

    # Have to explicitly calculate the event rate.  How to do this?  Event rate is total events over total file length in seconds.
    the_frame = pd.DataFrame(columns=["device", "VAC", "VAG", "time", "events"])
    for a_file in eventsfiles:
        frame = pd.read_csv(a_file)
        the_frame.loc[len(the_frame)] = [
            frame["device"][0],
            frame["VAC"][0],
            frame["VAG"][0],
            frame["TOTALTRACETIME"][0] / 1000000.0,
            frame["NUMBEROFEVENTS"][0],
        ]

    the_frame["rate"] = the_frame["events"] / the_frame["time"]

    the_frame = the_frame[the_frame.rate < 1.5]  # things above 1.5 Hz are odd
    the_frame = the_frame[the_frame.VAC < 0]  # only negative ac biases.

    print(the_frame)

    sns.set()
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))
    # Want event rate on vertical, event dwell time on horisontal, hue by VAC
    IVplot = sns.scatterplot(x="VAG", y="rate", data=the_frame, hue="VAC")
    # IVplot.errorbar(
    #     smallDF.VAC,
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
    plt.xlabel(r"$V_{AG}$ [mV]")
    plt.ylabel(r"Event rate [1/sec]")
    fileName = sample + "_gateeventrate.png"
    IVplot.figure.savefig(savePath / fileName, dpi=600)


#### Main loop
if __name__ == "__main__":
    resultsPath, savePath, statsDF = enclose_the_pandas()
    pandas_the_IV(resultsPath, savePath, statsDF)
