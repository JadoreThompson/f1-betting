from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from ..typing import (
    LoosePositionCategory,
    TightPositionCategory,
    Top3PositionCategory,
    WinnerPositionCategory,
)


PosCat = Literal["tight", "loose", "winner", "top3"]


def drop_temp_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def get_position_category(value: str, pos_cat: PosCat) -> str:
    """
    Categorizes a race finishing position string into a predefined class based on the selected mode.

    Args:
        value (str): Finishing position as a string.
        pos_cat (PosCat): Categorization scheme to use.

    Returns:
        str: Categorical label representing the finishing position, based on the chosen `pos_cat`:
    """
    value = str(value)

    if pos_cat == "winner":
        if value == "1":
            return WinnerPositionCategory.WINNER.value
        return WinnerPositionCategory.NOT_WINNER.value

    if pos_cat == "top3":
        if not value.isdigit() or int(value) > 3:
            return Top3PositionCategory.NOT_TOP_3.value
        return Top3PositionCategory.TOP3.value

    if pos_cat == "tight":
        # print(type(value), value)
        if not value.isdigit() or int(value) > 3:
            return TightPositionCategory.DNF.value
        if value == "1":
            return TightPositionCategory.FIRST.value
        if value == "2":
            return TightPositionCategory.SECOND.value
        return TightPositionCategory.THIRD.value

    if not value.isdigit() or value == "0":
        return LoosePositionCategory.DNF.value

    val = int(value)
    if val <= 3:
        return LoosePositionCategory.TOP_3.value
    if val <= 5:
        return LoosePositionCategory.TOP_5.value
    if val <= 10:
        return LoosePositionCategory.TOP_10.value
    return LoosePositionCategory.TOP_20.value


def append_avg_position_move(
    df: pd.DataFrame,
    *,
    in_season: bool = True,
    progressive: bool = True,
    window: int = 1,
) -> pd.DataFrame:
    """
    Compute and append the average position change for each driver.

    This metric captures how much a driver gains or loses relative to their grid position.
    Supports per-season grouping and rolling/progressive averaging.

    Args:
        df (pd.DataFrame): DataFrame with 'grid', 'position_numeric', 'driverId', and 'year' columns.
        in_season (bool): Whether to compute averages within each season (default: True).
        progressive (bool): Whether to use rolling/expanding mean up to the current race (default: True).
        window (int): Window size for rolling mean if progressive is True. If <1, uses expanding mean.

    Returns:
        pd.DataFrame: Input DataFrame with an added column for average position change.
    """
    if any(key and key not in df.columns for key in ("position_numeric", "grid")):
        raise ValueError("position_numeric and grid must be in dataframe object.")

    df["tmp_position_change"] = df["grid"] - df["position_numeric"]
    final_key = f"avg_position_move{"_in_season" if in_season else ""}{"_progressive" if progressive else ""}_{window}"

    group_cols = ["year", "driverId"] if in_season else ["driverId"]

    pcs = df.groupby(group_cols)["tmp_position_change"]

    if progressive:
        if window < 1:
            s = pcs.apply(lambda x: x.shift(1).expanding().mean())
        else:
            s = pcs.apply(lambda x: x.shift(1).rolling(window=window).mean())
        s = s.reset_index(level=group_cols, drop=True)

    else:
        s = pcs.transform("mean")

    df[final_key] = s
    return drop_temp_cols(df)


def append_sma(
    df: pd.DataFrame,
    col: str = "position_numeric",
    *,
    in_season: bool = True,
    progressive: bool = True,
    window: int = 1,
) -> pd.DataFrame:
    """
    Append a simple moving average (SMA) for a given column, grouped by driver and optionally by season.

    The average excludes the current race via a one-step shift. Supports rolling or expanding averages.

    Args:
        df (pd.DataFrame): DataFrame with at least the specified column and driver/year identifiers.
        col (str): Name of the column to average. Must be numeric or convertible.
        in_season (bool): Whether to compute the average within each season (default: True).
        progressive (bool): If True, compute a rolling or expanding average (default: True).
        window (int): Number of past races to include in the average. If <1, uses expanding mean.

    Returns:
        pd.DataFrame: The input DataFrame with a new SMA column added.
    """
    tmp_col = f"tmp_{col}"
    df[tmp_col] = pd.to_numeric(df[col])

    group_cols = ["year", "driverId"] if in_season else ["driverId"]

    gs = df.groupby(group_cols)[tmp_col]

    if progressive:
        if window < 1:
            s = gs.transform(lambda x: x.shift(1).expanding().mean())
        else:
            s = gs.transform(lambda x: x.shift(1).rolling(window=window).mean())

        s = s.reset_index(level=group_cols, drop=True)
    else:
        s = gs.transform("mean")

    df[f"sma_{col}{"_progressive" if progressive else ""}_{window}"] = s
    return drop_temp_cols(df)


def append_avg_position(
    df: pd.DataFrame,
    col: Literal["position_numeric", "positionText"] = "position_numeric",
    *,
    in_season: bool = True,
    progressive: bool = True,
    window: int = 1,
) -> pd.DataFrame:
    """
    Appends the average finishing position for each driver based on the specified position column.
    Excludes the current race via a one-step shift. Supports per-season grouping and progressive averaging.

    Args:
        df (pd.DataFrame): DataFrame containing race results with driver and season identifiers.
        col (str): Column to use for position ("position_numeric" or "positionText").
        in_season (bool): Whether to compute averages within each season (default: True).
        progressive (bool): Whether to use a rolling or expanding average (default: True).
        window (int): Number of races to include in the moving average. If <1, uses expanding mean.

    Returns:
        pd.DataFrame: DataFrame with an additional column for average position.
    """
    tmp_col = f"tmp_{col}"
    df[tmp_col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["year", "driverId"] if in_season else ["driverId"]
    gs = df.groupby(group_cols)[tmp_col]

    if progressive:
        if window < 1:
            s = gs.transform(lambda x: x.shift(1).expanding().mean())
        else:
            s = gs.transform(lambda x: x.shift(1).rolling(window=window).mean())
    else:
        s = gs.transform("mean")

    colname = f"avg_{col}{"_progressive" if progressive else ""}_{window}"
    df[colname] = s

    return drop_temp_cols(df)


def append_last_n_races(
    df: pd.DataFrame, window: int = 5, col: str = "positionText"
) -> pd.DataFrame:
    """
    Appends the last N values of a specified column for each driver as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing driver race data, including 'raceId' and 'driverId'.
        lookback (int): Number of past races to include. Defaults to 5.
        col (str): Column name from which to extract historical values. Defaults to "positionText".

    Returns:
        pd.DataFrame: DataFrame with additional columns for each of the last N values.
    """
    for i in range(1, window + 1):
        df[f"last_{col}_{i}"] = df.groupby("driverId")[col].shift(i)

    return drop_temp_cols(df)


def append_position_propensity(
    df: pd.DataFrame,
    pos_cat: PosCat,
    *,
    in_season=True,
) -> pd.DataFrame:
    """
    Calculates the propensity for each driver to finish within specified position categories.

    This function adds new columns to the DataFrame that represent the historical probability
    of a driver finishing in each position category (as defined by pos_cat parameter).
    The propensity is calculated based on the driver's previous race results.

    Args:
        df (pd.DataFrame): DataFrame containing race results with columns 'driverId',
            'position_numeric', 'year', and 'raceId'.
        pos_cat (PosCat): Position category type to use.
        in_season (bool, optional): If True, calculates propensity per driver per year.
            If False, calculates across all years. Defaults to True.

    Returns:
        pd.DataFrame: Original DataFrame with added propensity columns (one for each
                     position category value) and temporary columns removed.
    """
    df["tmp_position"] = df["position"].apply(
        lambda x: get_position_category(x, pos_cat)
    )
    category_cls: Enum = {
        "loose": LoosePositionCategory,
        "tight": TightPositionCategory,
        "top3": Top3PositionCategory,
        "winner": WinnerPositionCategory,
    }[pos_cat]
    dfs = []  # storing for concatenation

    for _, group in df.groupby(["driverId", "year"] if in_season else "driverId"):
        counts = {k: 0 for k in category_cls._value2member_map_}
        vals = counts.values()
        prop_series = {k: [] for k in counts}

        group["tmp_position"] = (
            group["tmp_position"].shift(1).apply(lambda x: "0" if x is None else x)
        )

        for _, g in group.iterrows():
            counts[g["tmp_position"]] += 1
            s = sum(vals)

            for k in counts:
                if counts[k] and s:
                    val = counts[k] / s
                else:
                    val = 0

                prop_series[k].append(val)

        for k, v in prop_series.items():
            group[f"propensity_{k}"] = v

        dfs.append(group)

    df = pd.concat(dfs, ignore_index=True)
    return drop_temp_cols(df.sort_values("raceId", axis=0))
