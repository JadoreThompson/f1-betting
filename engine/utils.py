import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import namedtuple
from datetime import date, datetime
from typing import Iterable
from .config import (
    DPATH,
    MPATH,
    TRAINED_MODEL_CLASSES,
    TRAINED_MODEL_FEATURES,
    TRAINED_MODEL,
)


# prediction value, decimal representation of percentage
Prediction = namedtuple("Prediction", ["prediction", "percentage"])


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        ["raceId", "driverId", "constructorId", "position", "year"],
        axis=1,
    )


def parse_quali_times(s: str) -> int:
    """Parse time strings into milliseconds."""
    if pd.isna(s) or s == "\\N":
        return 0

    first_split: list[str] = s.split(":")
    mins, secs, ms = first_split[0], *first_split[1].split(".")

    return int(mins) * 60_000 + int(secs) * 1000 + int(ms)


def parse_times(s: str) -> int:
    """Parse time strings into seconds."""
    if pd.isna(s) or s == "\\N":
        return np.nan

    hour, minute, second = s.split(":")
    return int(hour) * 3600 + int(minute) * 60 + int(second)


def sma(s: pd.Series, df: pd.DataFrame, column: str, window: int = 5) -> float:
    if s.name < window:
        return np.nan

    nums = [df.loc[s.name - i][column] for i in range(1, window + 1)]
    return sum(nums) / len(nums)


def get_position_category(value: str) -> str:
    # # categores:
    # # 0 - top 3
    # # 1 - top 5
    # # 2 - top 10
    # # 3 - top 20 / retired

    # if not value.isdigit():
    #     return "3"

    # val = int(value)

    # if val <= 3:
    #     return "0"
    # if val <= 5:
    #     return "1"
    # return "2"

    # categores:
    # 0 - top 1
    # 1 - top 3
    # 2 - top 5
    # 3 - top 10
    # 4 - top 20
    # 5 - retired

    if not value.isdigit():
        return "5"

    val = int(value)

    if val == 1:
        return "0"
    if val < 3:
        return "1"
    if val < 5:
        return "2"
    if val < 10:
        return "3"
    return "4"


def add_last_n_races(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Add last n race results for each driver to the dataset."""
    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driverId")["positionText"].shift(i)

    df["raceId"] = df["raceId"].astype("int")
    df = df.sort_values("raceId")
    df["raceId"] = df["raceId"].astype("str")
    return df


def get_df(min_year: int, sma_length: int = 4) -> pd.DataFrame:
    """Returns the dataframe without the dropped columns."""
    constructors_df = pd.read_csv(os.path.join(DPATH, "constructors.csv"))[
        ["constructorId", "constructorRef"]
    ]

    qualifying_df = pd.read_csv(os.path.join(DPATH, "qualifying.csv"))[
        ["driverId", "raceId", "position"]
    ]
    # qualifying_df["position"] = qualifying_df["position"].astype("str")

    results_df = pd.read_csv(os.path.join(DPATH, "results.csv"))[
        [
            "raceId",
            "driverId",
            "constructorId",
            # "grid",
            "position",
            "positionText",
        ]
    ]
    results_df["position"] = (
        pd.to_numeric(results_df["position"], errors="coerce").fillna(0).astype("int")
    )

    races_df = pd.read_csv(os.path.join(DPATH, "races-2023.csv"))[["raceId", "year"]]
    races_df = races_df[races_df["year"] >= min_year]

    df = races_df.merge(results_df, on=["raceId"], suffixes=("_race", "_result"))
    # df = df.merge(qualifying_df, on=["raceId", "driverId"], suffixes=("", "_quali"))
    # df = df.merge(constructors_df, on=["constructorId"])

    # df["sma"] = df.apply(lambda x: sma(x, df, "position", sma_length), axis=1)
    df = add_last_n_races(df, sma_length)
    df["positionText"] = df["positionText"].apply(lambda x: get_position_category(x))

    return df.dropna().reset_index().drop("index", axis=1)


def split_df(
    # df: pd.DataFrame, split_size: float = 0.7
    df: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a test and train split of a dataset

    Args:
        df (pd.DataFrame): _description_
        split_size (float): percentage of dataset to be trained on

    Returns:
        pd.DataFrame: _description_
    """

    train_df, test_df = (
        df[df["year"] <= year],
        df[df["year"] > year],
    )

    train_df, test_df = drop_columns(train_df), drop_columns(test_df)
    return train_df, test_df


def get_train_test(
    min_year: int, sma_length: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df = get_df(min_year=min_year, sma_length=sma_length)
    train_df, test_df = split_df(df, 2022)
    train_df.to_csv(os.path.join(DPATH, "train.csv"), index=False)
    test_df.to_csv(os.path.join(DPATH, "test.csv"), index=False)
    return train_df, test_df, len(df["raceId"].unique())


def save_model(model, name: str) -> None:
    fpath = os.path.join(MPATH, name)

    if not os.path.exists(MPATH):
        os.makedirs(MPATH)

    if os.path.exists(fpath):
        os.remove(fpath)

    model.save(fpath)


def interact(data: pd.DataFrame | Iterable) -> tuple[Prediction, ...]:
    if isinstance(data, Iterable):
        d = pd.DataFrame(data, columns=TRAINED_MODEL_FEATURES)
    else:
        d = data

    preds: list[list[float]] = TRAINED_MODEL.predict(d).tolist()

    return tuple(
        Prediction(TRAINED_MODEL_CLASSES[row.index(m := max(row))], m) for row in preds
    )


def plot_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    plt.title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    plt.show()
