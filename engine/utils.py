import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import namedtuple
from typing import Iterable, Literal

from .config import (
    DPATH,
    MPATH,
    TRAINED_MODEL,
)
from .typing import (
    LoosePositionCategory,
    TightPositionCategory,
)


# prediction value, decimal representation of percentage
Prediction = namedtuple("Prediction", ["prediction", "percentage"])


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


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        ["raceId", "driverId", "constructorId", "position", "year"],
        axis=1,
    )


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
            return TightPositionCategory.FIRST.value
        if val == 2:
            return TightPositionCategory.SECOND.value
        if val == 3:
            return TightPositionCategory.THIRD.value
        if val <= 5:
            return TightPositionCategory.TOP_5.value
        if val <= 10:
            return TightPositionCategory.TOP_10.value
        return TightPositionCategory.TOP_20.value
    else:
        if val <= 3:
            return LoosePositionCategory.TOP_3.value
        if val <= 5:
            return LoosePositionCategory.TOP_5.value
        if val <= 10:
            return LoosePositionCategory.TOP_10.value
        return LoosePositionCategory.TOP_20.value


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


def append_sma(df: pd.DataFrame, window: int, col: int = "position") -> pd.DataFrame:
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


def append_rolling_position_move(
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
    df[f"rolling_pos_change_{window}"] = (
        (df["grid"] - df["position"])
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )
    return df


def append_avg_position(
    df: pd.DataFrame,
    type_: Literal["real", "loose", "tight"] = "real",
    *,
    rolling: bool = False,
    window: int = None,
    weight: float = 1.0,
) -> pd.DataFrame:
    """
    Returns a DataFrame with average position per driver added.
    Optionally applies a rolling average over a specified window.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'driverId' and 'position'.
        type_ (Literal["real", "loose", "tight"], optional):
            Determines how position values are interpreted:
            - "real": use numeric finishing positions,
            - "loose": use the loose category representation,
            - "tight": use the tight category representation.
            Defaults to "real".
        rolling (bool, optional): Whether to apply a rolling average. Defaults to False.
        window (int, optional): Rolling window size. Required if rolling is True.
        weight (float, optional) # TODO

    Returns:
        pd.DataFrame: Original DataFrame with an added column:
            - "avg_position_{type_}": average or rolling average position per driver of type float64.
    """
    if rolling and (window is None or window < 1):
        raise ValueError(
            "window must be greater than or equal to one if rolling is set to true."
        )

    df = df.copy()
    col = "temp"
    name = f"avg_position_{type_}{f"_rolling_{window}" if rolling else ""}"

    if type_ == "real":
        df[col] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype("int")
    else:
        df[col] = (
            df["position"]
            .apply(lambda x: get_position_category(x, type_))
            .astype("int")
        )

    if rolling:
        s = (
            df.groupby("driverId")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .rename(name)
        )
    else:
        s = df.groupby("driverId")[col].mean().rename(name)

    df = df.drop(col, axis=1)
    df = df.merge(s, on="driverId")
    return df


def append_last_n_races(
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


def append_std(df: pd.DataFrame) -> pd.DataFrame:
    def helper(values: list) -> float:
        try:
            avg = sum(values) / len(values)
            diffs = [(v - avg) ** 2 for v in values]
            return math.sqrt(sum(diffs) / len(diffs))
        except ZeroDivisionError:
            return 0.0

    df = df.copy()
    df["position"] = (
        pd.to_numeric(df["position"], errors="coerce").fillna(0).astype("int")
    )
    df = df.merge(
        df.groupby("driverId")["position"]
        .agg(list)
        .rename("std")
        .apply(lambda x: helper(x)),
        on=["driverId"],
    )
    df = df.sort_values("raceId")
    return df


def append_position_propensity(
    df: pd.DataFrame, type_: Literal["loose", "tight"]
) -> pd.DataFrame:
    df["temp_position"] = df["position"].apply(
        lambda x: get_position_category(x, type_)
    )
    pos_props = (
        df.groupby("driverId")["temp_position"]
        .agg(list)
        .rename("position")
        .apply(lambda x: pd.Series(x).value_counts().apply(lambda y: 1 / (len(x) / y)))
    )
    pos_props = pos_props[list(sorted(pos_props.columns))]
    pos_props.columns = [f"propensity_{type_}_{col}" for col in pos_props.columns]
    df = df.merge(pos_props, on="driverId")
    return df.drop("temp_position", axis=1)


def get_df(min_year: int, max_year: int = 2026, sma_length: int = 4) -> pd.DataFrame:
    """Returns the dataframe without the dropped columns."""
    constructors_df = pd.read_csv(os.path.join(DPATH, "constructors.csv"))[
        ["constructorId", "constructorRef"]
    ]

    drivers_df = pd.read_csv(os.path.join(DPATH, "drivers.csv"))[
        ["driverId", "driverRef"]
    ]

    qualifying_df = pd.read_csv(os.path.join(DPATH, "qualifying.csv"))[
        ["driverId", "raceId", "position", "q3", "q2", "q1"]
    ]
    qualifying_df["q3"] = qualifying_df["q3"].apply(lambda x: parse_quali_times(x))
    qualifying_df["q2"] = qualifying_df["q2"].apply(lambda x: parse_quali_times(x))
    qualifying_df["q1"] = qualifying_df["q1"].apply(lambda x: parse_quali_times(x))
    # qualifying_df["fastest_time"] = qualifying_df[["q3", "q2", "q1"]].min(axis=1)
    # qualifying_df["avg_quali_time"] = qualifying_df[["q3", "q2", "q1"]].mean()

    qualifying_df = qualifying_df.drop(["q3", "q2", "q1", "position"], axis=1)

    # qualifying_df["position"] = qualifying_df["position"].astype("str")
    # qualifying_df["position"] = (
    #     qualifying_df["position"]
    #     # .astype("str")
    #     # .apply(lambda x: get_position_category(x, "loose"))
    # )

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
    # results_df["position"] = (
    #     pd.to_numeric(results_df["position"], errors="coerce").fillna(0).astype("int")
    # )
    results_df["grid"] = results_df["grid"].astype("int")

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[["raceId", "year"]]
    races_df = races_df[(max_year >= races_df["year"]) & (races_df["year"] >= min_year)]
    # races_df = races_df.sort_values(["raceId"])

    df = races_df.merge(results_df, on=["raceId"], suffixes=("_race", "_result"))
    df = df.merge(qualifying_df, on=["raceId", "driverId"], suffixes=("", "_quali"))
    # df = df.merge(drivers_df, on=["driverId"])
    # df = df.merge(constructors_df, on=["constructorId"])

    df["positionText"] = df["positionText"].apply(
        lambda x: get_position_category(x, "loose")
    )
    # df = add_sma(df, sma_length, "position")
    df = append_last_n_races(df, sma_length, "positionText")
    # df["avg_pos_move"] = get_avg_position_move(df)
    # df = add_rolling_position_move(df, sma_length)
    df = append_avg_position(df, "real")
    # df = append_avg_position(df, "tight")
    # df = append_std(df)
    # df = append_avg_position(df, "real", rolling=True, window=sma_length)
    # df = append_position_propensity(df, "loose")
    df = df.drop_duplicates(subset=["raceId", "driverId"])
    return df


def split_df(
    df: pd.DataFrame,
    split_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a test and train split of a dataset

    Args:
        df (pd.DataFrame): _description_
        split_size (float): percentage of dataset to be trained on

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and test df
    """
    df = df.copy()
    df["year"] = df["year"].astype("int")

    train_df, test_df = (
        df[df["year"] <= split_year],
        df[df["year"] > split_year],
    )

    return drop_columns(train_df), drop_columns(test_df)


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


def interact(data: pd.DataFrame | Iterable, model=None) -> tuple[Prediction, ...]:
    if model is None:
        model = TRAINED_MODEL

    if isinstance(data, Iterable):

        d = pd.DataFrame(data, columns=model.input_feature_names())
    else:
        d = data
    preds: list[list[float] | float] = model.predict(d).tolist()

    if len(model_classes := model.label_classes()) == 2:
        return tuple(
            Prediction(model_classes[0 if prob < 0.5 else 1], prob) for prob in preds
        )
    else:
        return tuple(
            Prediction(model.label_classes()[row.index(m := max(row))], m)
            for row in preds
        )


def compute_success_rate(dataset: pd.DataFrame, target_label: str, model) -> float:
    predictions = model.predict(dataset)
    success = 0.0

    for i, preds in enumerate(predictions):
        if len(model.label_classes()) == 2:
            pred_index = 0 if preds < 0.5 else 1
        else:
            pred_index = preds.tolist().index(max(preds))

        pred = model.label_classes()[pred_index]

        if pred == dataset.at[i, target_label]:
            success += 1

        # if "1" <= pred < "3" or "1" <= dataset.at[i, TARGET_LABEL] < "3":
        #     total += 1
        #     if pred == dataset.at[i, TARGET_LABEL]:
        #         success += 1

    if success:
        success /= len(predictions)

    return success


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
