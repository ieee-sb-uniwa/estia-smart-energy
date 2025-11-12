import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from pandas._libs.tslibs.parsing import DateParseError

# relative path
data_path = Path(__file__).parent.parent / "data" / "raw" / "smart_home_energy_consumption_large.csv"

raw_df: pd.DataFrame = pd.read_csv(data_path)
df_columns = raw_df.columns

def plot_frequency_histogram(df: pd.DataFrame, column: str,show=True, save=False, save_path: str | Path ='') -> None:
    """
    Plots frequency of given column as a histogram with instances on the y-axis. Example usage for all columns

    for col in raw_df.columns:
    image_data_path = Path(__file__).parent.parent / "resources" / f"{col.lower()}_frequency.png"
    plot_frequency_histogram(raw_df, col, show=False, save=True, save_path=image_data_path)

    :param df: raw dataframe
    :param column: column of raw dataframe
    :param show: show created plot
    :param save: save created plot in given path,
    :param save_path: path to save created plot
    :return:
    """

    data = df[column]

    plt.figure(figsize=(8, 5))

    if column.lower() == 'time':
        time_series = pd.to_datetime(data, errors='coerce')
        hours = time_series.dt.hour
        print(hours)

        plt.hist(hours, bins=range(25), edgecolor='black', align='left')
        plt.title("Distribution of Observations by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Frequency")
        plt.xticks(range(24))

    elif column.lower() == 'date':
        date_series = pd.to_datetime(data, errors='coerce')
        weeks = date_series.dt.isocalendar().week  # ISO week numbers (1–52)

        plt.hist(weeks, bins=range(1, 54), edgecolor='black', align='left')
        plt.title("Distribution of Observations by Week of Year")
        plt.xlabel("Week Number (1–52)")
        plt.ylabel("Frequency")
        plt.xticks(range(0, 53, 4))

    elif pd.api.types.is_numeric_dtype(data):
        # numberic → histogram
        plt.hist(data, bins=30, edgecolor='black')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

    else:
        # strings-objects → count plot
        counts = data.value_counts().head(20)  #
        counts.plot(kind='bar')
        plt.title(f"Category counts of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")

    plt.tight_layout()

    if save:
        plt.savefig(save_path)

    if show:
        plt.show()


def save_metadata_json(df, json_file) -> None:

    for column in df.columns:

        null_values_count = 0
        unique_values_count: int | None = 0
        unique_values: list | None = [] # in case of numeral or timeseries
        dtype: str = ''
        most_common: str | None = ''
        least_common: str | None = ''
        percent_null: int = 0

        # for numeric
        range: list = [] # min max
        mean: float | int = 0
        median: float | int = 0
        std: float | int = 0

        # for timeseries as seen from figures, data is evenly distributed so i dont do anything

        data = df[column]

        # checking for time series data
        is_timeseries = True
        try:
            _ = pd.to_datetime(data, errors='coerce')
        except DateParseError:
            is_timeseries = False

        if is_timeseries:
            pass
        else:
            pass