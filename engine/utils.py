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
    value: str, pos_cat: Literal["tight", "loose", "binary", "top3"] = "tight"
) -> str:
    if pos_cat == "binary":
        if value == "1":
            return "win"
        return "lose"

    if pos_cat == "top3":
        if not value.isdigit() or value > "3":
            return "0"
        return "1"

    if not value.isdigit() or value == "0":
        return "0"

    val = int(value)

    if pos_cat == "tight":
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

    # if pos_cat == "loose":
    if val <= 3:
        return LoosePositionCategory.TOP_3.value
    if val <= 5:
        return LoosePositionCategory.TOP_5.value
    if val <= 10:
        return LoosePositionCategory.TOP_10.value
    return LoosePositionCategory.TOP_20.value


def append_avg_position_move(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a Series of average position change per driver.
    Negative values indicate a decline in position and positive
    the opposite.
    """
    df = df.copy()
    df["tmp_grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["tmp_position"] = pd.to_numeric(df["position"], errors="coerce")
    df["tmp_race_id"] = pd.to_numeric(df["raceId"], errors="coerce")
    df = df.dropna(subset=["tmp_grid", "tmp_position"])

    df = df.sort_values("tmp_race_id", axis=0)
    df = df.reset_index(drop=True)

    df["tmp_position_change"] = df["tmp_grid"] - df["tmp_position"]
    df = df.merge(
        df.groupby("driverId")["tmp_position_change"]
        .mean()
        .rename("avg_position_move"),
        on="driverId",
    )
    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def append_sma(df: pd.DataFrame, window: int, col: int = "position") -> pd.DataFrame:
    tmp_col = f"tmp_{col}"

    df = df.copy()
    df["tmp_race_id"] = pd.to_numeric(df["raceId"])
    df["tmp_driver_id"] = pd.to_numeric(df["driverId"], errors="coerce").fillna(0)
    df[tmp_col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["tmp_race_id"])
    df[f"sma_{window}"] = df.groupby("driverId")[tmp_col].transform(
        lambda x: x.shift(1).rolling(window=window).mean()
    )

    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def append_rolling_position_move(
    df: pd.DataFrame, window: int, min_periods: int = 0
) -> pd.DataFrame:
    """
    Returns a Series of average position change per driver.
    """
    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["raceId"] = df["raceId"].astype("int")
    df = df.dropna(subset=["grid", "position"])
    df = df.drop_duplicates(subset=["raceId", "driverId"])

    df = df.sort_values("raceId")
    df[f"temp"] = df["grid"] - df["position"]

    df[f"rolling_avg_pos_move_{window}"] = df.groupby("driverId")["temp"].transform(
        lambda x: x.shift(1).rolling(window=window).mean()
    )
    return df.drop("temp", axis=1)


def append_avg_position(
    df: pd.DataFrame,
    pos_cat: Literal["real", "loose", "tight"] = "real",
    *,
    rolling: bool = False,
    window: int = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with average position per driver added.
    Optionally applies a rolling average over a specified window.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'driverId' and 'position'.
        pos_cat (Literal["real", "loose", "tight"], optional):
            Determines how position values are interpreted:
            - "real": use numeric finishing positions,
            - "loose": use the loose category representation,
            - "tight": use the tight category representation.
            Defaults to "real".
        rolling (bool, optional): Whether to apply a rolling average. Defaults to False.
        window (int, optional): Rolling window size. Required if rolling is True.

    Returns:
        pd.DataFrame: Original DataFrame with an added column:
            - "avg_position_{pos_cat}": average or rolling average position per driver of type float64.
    """
    if rolling and (window is None or window < 1):
        raise ValueError(
            "window must be greater than or equal to one if rolling is set to true."
        )

    df = df.copy()
    col = "temp_position_repr"
    name = f"avg_position_{pos_cat}{f"_rolling_{window}" if rolling else ""}"

    if pos_cat == "real":
        df[col] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype("int")
    else:
        df[col] = (
            df["position"]
            .apply(lambda x: get_position_category(x, pos_cat))
            .astype("int")
        )

    if rolling:
        df[name] = (
            df.groupby(["driverId", "year"])[col]
            .transform(lambda x: x.shift(1).rolling(window=window).mean())
            .rename(name)
        )
    else:
        df[name] = df.groupby("driverId")[col].transform("mean").rename(name)

    df = df.drop(col, axis=1)
    return df


def append_last_n_races(
    df: pd.DataFrame, lookback: int = 5, col: str = "positionText"
) -> pd.DataFrame:
    """Inserts and returns last n values for positionText for each driver
    to the dataset."""
    df = df.copy()
    df["tmp_race_id"] = pd.to_numeric(df["raceId"])
    df = df.sort_values("tmp_race_id")

    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driverId")[col].shift(i)

    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


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
    df: pd.DataFrame, pos_cat: Literal["loose", "tight"]
) -> pd.DataFrame:
    df["position"] = df["position"].astype("str")
    df["temp_position"] = df["position"].apply(
        lambda x: get_position_category(x, pos_cat)
    )
    pos_props = (
        df.groupby("driverId")["temp_position"]
        .agg(list)
        .rename("position")
        .apply(lambda x: pd.Series(x).value_counts().apply(lambda y: 1 / (len(x) / y)))
    )
    pos_props = pos_props[list(sorted(pos_props.columns))]
    pos_props.columns = [f"propensity_{pos_cat}_{col}" for col in pos_props.columns]
    df = df.merge(pos_props, on="driverId")
    return df.drop("temp_position", axis=1)


def append_cur_points(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values("raceId").reset_index(drop=True)

    # s = df.groupby()
    # df["tmp_race__id"] =
    # return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def get_df(min_year: int, max_year: int = 2026, sma_length: int = 4) -> pd.DataFrame:
    """Returns the dataframe without the dropped columns."""
    # circuits_df =
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

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[
        ["raceId", "circuitId", "year"]
    ]
    races_df = races_df[(max_year >= races_df["year"]) & (races_df["year"] >= min_year)]
    races_df["circuitId"] = races_df["circuitId"].astype("str")

    df = races_df.merge(results_df, on=["raceId"])

    df["positionText"] = df["positionText"].apply(
        lambda x: get_position_category(x, "top3")
    )
    df = append_last_n_races(df, 3, "positionText")
    df = append_rolling_position_move(df, 10, 1)
    df = append_avg_position(df, rolling=True, window=10)
    append_cur_points(df)
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
    *,
    min_year: int,
    max_year: int,
    split_year: int,
    sma_length: int = 4,
    save: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df = get_df(min_year, max_year, sma_length)
    train_df, test_df = split_df(df, split_year)

    if save:
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


def compute_success_rate(
    dataset: pd.DataFrame, target_label: str, model=None, top_range: bool = False
) -> float:
    success = 0.0
    count = 0

    if model is None:
        model = TRAINED_MODEL

    predictions = model.predict(dataset)
    pred_values = []

    for i, preds in enumerate(predictions):
        if len(model.label_classes()) == 2:
            pred_index = 0 if preds < 0.5 else 1
        else:
            pred_index = preds.tolist().index(max(preds))

        pred = model.label_classes()[pred_index]
        pred_values.append(pred)

        if top_range:
            if "1" <= pred < "3" or "1" <= dataset.iloc[i][target_label] < "3":
                count += 1
                if pred == dataset.iloc[i][target_label]:
                    success += 1
        else:
            if pred == dataset.iloc[i][target_label]:
                success += 1

    dataset["predictions"] = pred_values

    if success:
        if top_range:
            success /= count
        else:
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


if __name__ == "__main__":
    pass
    # df = get_df(2010, 2024, 10)
    # df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0)
    # # print(df.columns)
    # plot_heatmap(df)
