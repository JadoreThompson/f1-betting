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
    Top3PositionCategory,
    WinnerPositionCategory,
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


def drop_temp_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        [
            "raceId",
            "driverId",
            "constructorId",
            "position",
            "year",
            "circuitId",
            "current_standings_position",
        ],
        axis=1,
    )


def get_position_category(
    value: str, pos_cat: Literal["tight", "loose", "winner", "top3"] = "tight"
) -> str:
    """
    Categorizes a race finishing position string into a predefined class based on the selected mode.

    Args:
        value (str): Finishing position as a string.
        pos_cat (Literal["tight", "loose", "winner", "top3"]): Categorization scheme to use.
            - "winner": Returns "1" if first place, else "0".
            - "top3": Returns a label indicating if position is in top 3.
            - "tight": Returns detailed categories like 1st, 2nd, 3rd, top 5, etc.
            - "loose": Returns broader categories like top 3, top 5, etc.

    Returns:
        str: Categorical label representing the finishing position, based on the chosen `pos_cat`:

        **If `pos_cat` is "winner":**
            - "1": Represents 1st place (winner).
            - "0": Represents any position other than 1st.

        **If `pos_cat` is "top3":**
            - "1": Represents positions 1, 2, or 3.
            - "0": Represents positions outside of the top 3.

        **If `pos_cat` is "tight":**
            - "1": Represents 1st place.
            - "2": Represents 2nd place.
            - "3": Represents 3rd place.
            - "4": Represents positions 4 or 5.
            - "5": Represents positions 6 through 10.
            - "6": Represents positions 11 and below.

        **If `pos_cat` is "loose":**
            - "1": Represents positions 1, 2, or 3.
            - "2": Represents positions 4 or 5.
            - "3": Represents positions 6 through 10.
            - "4": Represents positions 11 and below.
    """
    if pos_cat == "winner":
        if value == "1":
            return "1"
        return "0"

    if pos_cat == "top3":
        if not value.isdigit() or value > "3":
            return Top3PositionCategory.NOT_TOP_3.value
        return Top3PositionCategory.TOP3.value

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

    if val <= 3:
        return LoosePositionCategory.TOP_3.value
    if val <= 5:
        return LoosePositionCategory.TOP_5.value
    if val <= 10:
        return LoosePositionCategory.TOP_10.value
    return LoosePositionCategory.TOP_20.value


def append_avg_position_move(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends each driver's average position change across races as a new column.
    Positive values indicate position gains, negative values indicate losses.

    Args:
        df (pd.DataFrame): Input DataFrame containing race data with grid and position columns.

    Returns:
        pd.DataFrame: DataFrame with an added column for average position change per driver.
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

    return drop_temp_cols(df)


def append_rolling_position_move(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Appends a rolling average of position change (grid position minus finishing position)
    for each driver over a specified window of races.

    Args:
        df (pd.DataFrame): Input DataFrame containing race data, with columns 'grid', 'position',
            'raceId', and 'driverId'.
        window (int): Number of races to include in the rolling average window.

    Returns:
        pd.DataFrame: DataFrame with an additional column representing the rolling average
            of position changes for each driver over the specified window.
    """
    df = df.copy()
    df["tmp_grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["tmp_position"] = pd.to_numeric(df["position"], errors="coerce")
    df["tmp_raceId"] = df["raceId"].astype("int")
    df = df.dropna(subset=["tmp_grid", "tmp_position"])
    df = df.drop_duplicates(subset=["tmp_raceId", "driverId"])

    df = df.sort_values("tmp_raceId")
    df["tmp_grid_pos_diff"] = df["tmp_grid"] - df["tmp_position"]

    df[f"rolling_avg_pos_move_{window}"] = df.groupby("driverId")[
        "tmp_grid_pos_diff"
    ].transform(lambda x: x.shift(1).rolling(window=window).mean())

    return drop_temp_cols(df)


def append_sma(df: pd.DataFrame, window: int, col: int = "position") -> pd.DataFrame:
    """
    Appends a simple moving average (SMA) of a specified column for each driver excluding
    the current race.

    Args:
        df (pd.DataFrame): Input DataFrame containing race data.
        window (int): Number of past races to include in the moving average.
        col (str): Column name from which to compute the moving average. Defaults to "position".

    Returns:
        pd.DataFrame: DataFrame with an additional column for the computed SMA.
    """
    tmp_col = f"tmp_{col}"

    df = df.copy()
    df["tmp_race_id"] = pd.to_numeric(df["raceId"])
    df["tmp_driver_id"] = pd.to_numeric(df["driverId"], errors="coerce").fillna(0)
    df[tmp_col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["tmp_race_id"])
    df[f"sma_{window}"] = df.groupby("driverId")[tmp_col].transform(
        lambda x: x.shift(1).rolling(window=window).mean()
    )

    return drop_temp_cols(df)


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
    col = "tmp_position_repr"
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

    return drop_temp_cols(df)


def append_last_n_races(
    df: pd.DataFrame, lookback: int = 5, col: str = "positionText"
) -> pd.DataFrame:
    """
    Appends the last N values of a specified column for each driver as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing driver race data.
        lookback (int): Number of past races to include. Defaults to 5.
        col (str): Column name from which to extract historical values. Defaults to "positionText".

    Returns:
        pd.DataFrame: DataFrame with additional columns for each of the last N values.
    """
    df = df.copy()
    df["tmp_race_id"] = pd.to_numeric(df["raceId"])
    df = df.sort_values("tmp_race_id")

    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driverId")[col].shift(i)

    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def append_std_progressive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the progressive standard deviation of race positions for each driver per season
    and appends it as a new column to the DataFrame.

    For each driver and each year, the function sorts races by raceId and calculates the
    standard deviation of all prior race positions up to (but not including) the current race.
    The result is stored in 'position_std'.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the following columns:
            - 'driverId': Identifier for each driver.
            - 'year': The season year.
            - 'raceId': Identifier for each race (used for chronological ordering).
            - 'position': The driver's race finishing position (can be string or numeric).

    Returns:
        pd.DataFrame: Modified DataFrame with a new column 'position_std' representing the
        progressive standard deviation of prior race positions.
    """

    def helper(values: list) -> float:
        try:
            avg = sum(values) / len(values)
            diffs = [(v - avg) ** 2 for v in values]
            return math.sqrt(sum(diffs) / len(diffs))
        except ZeroDivisionError:
            return 0.0

    df = df.copy()
    df["tmp_position"] = (
        pd.to_numeric(df["position"], errors="coerce").fillna(0).astype("int")
    )

    std_dfs = []
    for _, group in df.groupby(["driverId", "year"]):
        group = group.sort_values("raceId")
        group["tmp_position"] = group["tmp_position"].shift(1)
        values = []
        stds = []

        for _, row in group.iterrows():
            values.append(0 if pd.isna(row["tmp_position"]) else row["tmp_position"])
            stds.append(helper(values))

        group["position_std"] = stds
        std_dfs.append(group)

    df = pd.concat(std_dfs, ignore_index=True)
    return df.drop("tmp_position", axis=1)


def append_position_propensity(
    df: pd.DataFrame, pos_cat: Literal["loose", "tight", "top3", "winner"]
) -> pd.DataFrame:
    """
    Calculates the propensity for each driver to finish within a certain group
    i.e. propensity for finishing with pos_cat = "1" or pos_cat = "0" if "winner"
    is passed as the value for pos_cat.

    Args:
        df (pd.DataFrame): _description_
        pos_cat (Literal[&quot;loose&quot;, &quot;tight&quot;, &quot;top3&quot;, &quot;winner&quot;]): _description_

    Returns:
        pd.DataFrame: DataFrame with the added feature.
    """
    df = df.copy()

    df["tmp_position"] = (
        df["position"]
        .apply(lambda x: str(int(x) if x.isdigit() else 0))
        .apply(lambda x: get_position_category(x, pos_cat))
    )

    prop_dfs = []
    for _, group in df.groupby(["driverId", "year"]):
        props = {
            "loose": {k: 0 for k in LoosePositionCategory._value2member_map_},
            "tight": {k: 0 for k in TightPositionCategory._value2member_map_},
            "top3": {k: 0 for k in Top3PositionCategory._value2member_map_},
            "winner": {k: 0 for k in WinnerPositionCategory._value2member_map_},
        }[pos_cat]
        vals = props.values()
        props_series = {k: [] for k in props}

        group = group.sort_values("raceId")
        group["tmp_position"] = (
            group["tmp_position"]
            .shift(1)
            .apply(lambda x: x if x is not None else pd.NA)
        )
        group = group.dropna(subset=["tmp_position"])

        for _, g in group.iterrows():
            props[g["tmp_position"]] += 1
            s = sum(vals)

            for k in props:
                if props[k] and s:
                    val = props[k] / s
                else:
                    val = 0

                props_series[k].append(val)

        for k, v in props_series.items():
            group[f"propensity_{k}"] = v

        prop_dfs.append(group)

    df = pd.concat(prop_dfs, ignore_index=True)
    df = df.sort_values("raceId", axis=0)
    return drop_temp_cols(df)


def append_current_wins(df: pd.DataFrame):
    """
    Appends the current cumulative amount of wins for each driver
    irrespective of the season.

    Args:
        df (pd.DataFrame)
    """
    df["current_accum_wins"] = (
        df.sort_values("raceId")
        .groupby("driverId")["current_wins"]
        .shift(1)
        .rolling(1)
        .sum()
        .apply(lambda x: 0 if pd.isna(x) else x)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = append_last_n_races(df, 10, "positionText")
    df = append_rolling_position_move(df, 10)
    # # df = append_avg_position(df, rolling=True, window=5)
    # df = append_position_propensity(df, "top3")
    df = append_position_propensity(df, "winner")
    return df


def get_df(min_year: int, max_year: int = 2026, sma_length: int = 4) -> pd.DataFrame:
    """
    Fetches all datasets needed to for composition.

    Args:
        min_year (int)
        max_year (int, optional) Defaults to 2026.
        sma_length (int, optional). Redundant. Defaults to 4.

    Returns:
        pd.DataFrame: Fully comprised dataframe with added features.
    """
    driver_standings_df = pd.read_csv(os.path.join(DPATH, "driver_standings.csv"))[
        ["raceId", "driverId", "points", "position", "wins"]
    ]
    driver_standings_df["points"] = driver_standings_df["points"].shift(1)
    driver_standings_df = driver_standings_df.rename(
        columns={
            "position": "current_standings_position",
            "wins": "current_wins",
            "points": "current_points",
        }
    )

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
    df = df.merge(
        driver_standings_df, on=["raceId", "driverId"], suffixes=("", "_standings")
    )

    df["positionText"] = df["positionText"].apply(
        lambda x: get_position_category(x, "winner")
    )

    return add_features(df)


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

    return drop_features(train_df), drop_features(test_df)


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

    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = get_df(2010, 2024, 10)
    df["positionText"] = df["positionText"].astype("int")
    df = drop_features(df)
    plot_heatmap(df)
