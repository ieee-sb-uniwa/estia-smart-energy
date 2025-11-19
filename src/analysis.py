import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import seaborn as sns
from pandas._libs.tslibs.parsing import DateParseError
import numpy as np
from typing import Literal
import warnings
import os


warnings.filterwarnings("ignore")



# relative path
data_path = Path(__file__).parent.parent / "data" / "raw" / "smart_home_energy_consumption_large.csv"

raw_df: pd.DataFrame = pd.read_csv(data_path)
df_columns = raw_df.columns


def plot_frequency_histogram(df: pd.DataFrame, column: str, show=True, save=False, save_path: str | Path = '') -> None:
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


def _make_serializable(obj):
    """

    :param obj: dictionary - json
    :return: serializes - converts int64 to int s.t. can be parsed by json library
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, pd._libs.missing.NAType)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # convert to string
    else:
        return obj


def _get_dtype(df_col: pd.DataFrame) -> Literal['num', 'cat', 'timeseries']:
    """

    :param df_col: dataframe column to be tested
    :return: type of feature. this is a bit hard codded for now, created a function s.t. it can be expanded / fixed
    easily in the future.


    """
    # assuming full data, this can be fixed later
    try:
        _ = len(df_col.iloc[0])
        content = df_col.iloc[0]
        if len(content) == 5 or len(content) == 10:
            return 'timeseries'
        else:
            return 'cat'
    except TypeError:
        return 'num'


def save_metadata_json(df, json_file_path) -> None:
    """

    :param df: arbitrary dataframe containing numerical, categorical and timeseries data.
    :param json_file_path: path for json file to be saved - altered.
    :return: creates and saves a JSON metadata file in path parameter, consisting of:
    for all categories:
        null value count
        data type
        percent_null
    for numeric:
        range difference
        min
        max
        mean
        median
        std
    for categorical:
        unique count
        unique classes
        most frequent class
        least frequent class
    for timeseries:
        to be implemented
    """
    metadata_json = {}
    for column in df.columns:
        # for numeric
        range: list = []  # min max
        mean: float | int = 0
        median: float | int = 0
        std: float | int = 0

        # for cat
        most_common: str | None = ''
        least_common: str | None = ''
        unique_values_count: int | None = 0
        unique_values: list | None = []  # in case of numeral or timeseries

        # for timeseries as seen from figures, data is evenly distributed so do anything

        data = df[column]

        is_num = False
        is_timeseries = False
        is_cat = False

        # checking for time series data

        column_json = {}

        null_count = data.isnull().sum()
        column_json['null_count'] = null_count
        column_json['null_percentage'] = (null_count / len(data)) * 100

        if _get_dtype(data) == 'cat':
            column_json['data_type'] = 'categorical'

            column_json['unique_count'] = data.nunique()  # total number of classes
            column_json['unique_categories'] = data.unique().tolist()  # classes listed.

            counts_of_classes = data.value_counts()
            column_json['most_frequent_class'] = counts_of_classes.idxmax()
            column_json['least_frequent_class'] = counts_of_classes.idxmin()

        if _get_dtype(data) == 'num':
            column_json['data_type'] = 'numerical'

            column_json['range_diff']: float = data.max() - data.min()
            column_json['min']: float = data.min()
            column_json['max']: float = data.max()
            column_json['mean'] = round(data.mean(), 3)
            column_json['median'] = round(data.median(), 3)
            column_json['std'] = round(data.std(), 3)
        if _get_dtype(data) == 'timeseries':
            column_json['data_type'] = 'numerical'

        metadata_json[column] = column_json

    with open(json_file_path, "w") as f:
        json.dump(_make_serializable(metadata_json), f, indent=4)


def plot_energy_relationships(df, target='Energy Consumption (kWh)', save=False, save_folder=''):

    """

    :param df: full arbitrary dataframe.
    :param target: feature to be compared.
    :param save: save
    :param save_path: path to be saved.
    :return:
    """

    # num vs cat features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if target in numerical_features:
        numerical_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    for col in categorical_features:
        N = 20 # number of x labels for timedates
        print(col)
        if col == 'Time':
            # convert to datetime
            df['Time_hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour

            bins = list(range(0, 25))  # [0, 1, 2, ..., 24]
            labels = [str(i) for i in range(24)]  # ['0', '1', '2', ..., '23']
            df['Time_bin'] = pd.cut(df['Time_hour'], bins=bins, labels=labels, right=False)

            avg_energy = df.groupby('Time_bin')[target].mean().sort_index()

        else:
            avg_energy = df.groupby(col)[target].mean()
        plt.figure(figsize=(8, 5))

        sns.barplot(x=avg_energy.index, y=avg_energy.values, palette='viridis')
        plt.title(f"Average {target} by {col}")
        plt.ylabel(f"Average {target}")
        plt.xlabel(col)

        all_ticks = range(len(avg_energy.index))
        tick_positions = all_ticks[::max(1, len(all_ticks) // N)]
        plt.xticks(ticks=tick_positions, labels=[avg_energy.index[i] for i in tick_positions], rotation=45)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            os.makedirs(save_folder, exist_ok=True)

            save_path = os.path.join(save_folder, f'{col}_vs_{target}.png')
            plt.savefig(save_path)
        plt.show()

    for col in numerical_features:

        plt.figure(figsize=(8, 5))
        if df[col].nunique() > 20:

            df['binned'] = pd.cut(df[col], bins=20)
            avg_energy = df.groupby('binned')[target].mean()
            sns.barplot(x=avg_energy.index.astype(str), y=avg_energy.values, palette='coolwarm')
            plt.xticks(rotation=45)
            plt.xlabel(f"Binned {col}")
            plt.ylabel(f"Average {target}")
            plt.title(f"{target} vs {col} (binned)")
            df.drop(columns=['binned'], inplace=True)
        else:
            sns.barplot(x=df[col], y=df[target], ci=None, palette='coolwarm')
            plt.xlabel(col)
            plt.ylabel(target)
            plt.title(f"{target} vs {col}")
        plt.tight_layout()
        if save:
            os.makedirs(save_folder, exist_ok=True)

            save_path = os.path.join(save_folder, f'{col}_vs_{target}.png')
            plt.savefig(save_path)
        plt.show()


plot_energy_relationships(raw_df, save=True, save_folder='resources')