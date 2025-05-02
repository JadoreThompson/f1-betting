import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import namedtuple
from typing import Iterable, Literal
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


def add_sma(df: pd.DataFrame, window: int, col: int = "position") -> pd.DataFrame:
    df = df.copy()
    df["raceId"] = pd.to_numeric(df["raceId"])
    df["driverId"] = pd.to_numeric(df["driverId"], errors="coerce").fillna(0)
    df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["driverId", "raceId"])

    df[f"sma_{window}"] = (
        df.groupby("driverId")[col]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values(["raceId"])
    return df


def get_position_category(
    value: str, type_: Literal["tight", "loose", "binary"] = "tight"
) -> str:
    if type_ == "binary":
        if value == "1":
            return "win"
        return "lose"

    if not value.isdigit() or value == "0":
        return "0"

    val = int(value)

    if type_ == "tight":
        if val == 1:
            return "1"
        if val == 2:
            return "2"
        if val == 3:
            return "3"
        if val <= 5:
            return "4"
        if val <= 10:
            return "5"
        return "6"

    if val <= 3:
        return "1"
    if val <= 5:
        return "2"
    if val <= 10:
        return "3"
    return "4"


def add_rolling_position_move(
    df: pd.DataFrame, window: int, min_periods: int = 1
) -> pd.DataFrame:
    """
    Returns a Series of average position change per driver.
    """
    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["raceId"] = df["raceId"].astype("int")
    df = df.dropna(subset=["grid", "position"])

    df = df.sort_values(["raceId"])
    df["rolling_pos_change"] = (
        (df["grid"] - df["position"])
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )
    return df


def get_avg_position_move(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of average position change per driver.
    Negative values indicate a decline in position and positive
    the opposite.
    """
    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    df = df.dropna(subset=["grid", "position"])
    df["position_change"] = df["grid"] - df["position"]
    return df.groupby("driverId")["position_change"].mean()


def add_last_n_races(
    df: pd.DataFrame, lookback: int = 5, col: str = "positionText"
) -> pd.DataFrame:
    """Inserts and returns last n values for positionText for each driver
    to the dataset."""
    df = df.copy()

    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driverId")[col].shift(i)

    df["raceId"] = df["raceId"].astype("int")
    df = df.sort_values("raceId")
    df["raceId"] = df["raceId"].astype("str")
    return df


def get_df(min_year: int, max_year: int = 2026, sma_length: int = 4) -> pd.DataFrame:
    """Returns the dataframe without the dropped columns."""
    qualifying_df = pd.read_csv(os.path.join(DPATH, "qualifying.csv"))[
        ["driverId", "raceId", "q3", "q2", "q1"]
    ]
    qualifying_df["q3"] = qualifying_df["q3"].apply(lambda x: parse_quali_times(x))
    qualifying_df["q2"] = qualifying_df["q2"].apply(lambda x: parse_quali_times(x))
    qualifying_df["q1"] = qualifying_df["q1"].apply(lambda x: parse_quali_times(x))
    qualifying_df = qualifying_df.drop(["q3", "q2", "q1"], axis=1)

    results_df = pd.read_csv(os.path.join(DPATH, "results.csv"))[
        [
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "position",
            "positionText",
        ]
    ]
    results_df["grid"] = results_df["grid"].astype("int")

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[["raceId", "year"]]
    races_df = races_df[(max_year >= races_df["year"]) & (races_df["year"] >= min_year)]

    # Merging
    df = races_df.merge(results_df, on=["raceId"], suffixes=("_race", "_result"))
    df = df.merge(qualifying_df, on=["raceId", "driverId"], suffixes=("", "_quali"))

    # Feature construction
    df["positionText"] = df["positionText"].apply(
        lambda x: get_position_category(x, "loose")
    )
    df = add_last_n_races(df, sma_length, "positionText")

    return df


def split_df(
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
    df = df.copy()
    df["year"] = df["year"].astype("int")

    train_df, test_df = (
        df[df["year"] <= year],
        df[df["year"] > year],
    )

    train_df, test_df = drop_columns(train_df), drop_columns(test_df)
    return train_df, test_df


def get_train_test(
    *, min_year: int, max_year: int, split_year: int, sma_length: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df = get_df(min_year, max_year, sma_length)
    train_df, test_df = split_df(df, split_year)
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


def interact(
    data: pd.DataFrame | Iterable, type_: Literal["multi", "binary"] = "multi"
) -> tuple[Prediction, ...]:
    if isinstance(data, Iterable):
        d = pd.DataFrame(data, columns=TRAINED_MODEL_FEATURES)
    else:
        d = data

    preds: list[list[float] | float] = TRAINED_MODEL.predict(d).tolist()

    if type_ == "multi":
        return tuple(
            Prediction(TRAINED_MODEL_CLASSES[row.index(m := max(row))], m)
            for row in preds
        )

    return tuple(
        Prediction(TRAINED_MODEL_CLASSES[0 if prob < 0.5 else 1], prob)
        for prob in preds
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
