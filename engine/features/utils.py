import numpy as np
import pandas as pd

from enum import Enum
from typing import Literal
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
    window: int = 3,
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

    tpcs = df.groupby(group_cols)["tmp_position_change"]

    if progressive:
        if window < 1:
            s = tpcs.apply(lambda x: x.shift(1).expanding().mean())
        else:
            s = tpcs.apply(lambda x: x.shift(1).rolling(window=window).mean())
        s = s.reset_index(level=group_cols, drop=True)

    else:
        s = tpcs.transform("mean")

    df[final_key] = s
    df = df.dropna(subset=[final_key])
    return drop_temp_cols(df)


def append_sma(
    df: pd.DataFrame,
    col: str = "position_numeric",
    *,
    in_season: bool = True,
    progressive: bool = True,
    window: int = 3,
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
            s = gs.apply(lambda x: x.shift(1).expanding().mean())
        else:
            s = gs.apply(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

        s = s.reset_index(level=group_cols, drop=True)
    else:
        s = gs.transform("mean")

    df[f"sma_{col}{"_progressive" if progressive else ""}_{window}"] = s
    df = df.dropna(
        subset=[f"sma_{col}{"_progressive" if progressive else ""}_{window}"]
    )
    return drop_temp_cols(df)


def append_median_race_position(
    df: pd.DataFrame,
    col: str = "position_numeric",
    *,
    in_season: bool = True,
    progressive: bool = True,
    window: int = 3,
) -> pd.DataFrame:
    """
    Calculates the median value for col passed over the last N races
    provided by window.

    Args:
        df (pd.DataFrame): Input dataframe containing the race data.
        col (str, optional): Column used to calculate the median. Defaults to "position_numeric".
        in_season (bool, optional): Whether to focus on the season or overall
            . Defaults to True.
        progressive (bool, optional): Whether to apply mean across the whole season or gradually
            as races finish. Defaults to True.
        window (int, optional): Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame with the median col appended.
    """
    tmp_col = f"tmp_{col}"
    df[tmp_col] = pd.to_numeric(df[col])

    group_cols = ["year", "driverId"] if in_season else ["driverId"]
    gs = df.groupby(group_cols)[tmp_col]

    if progressive:
        if window < 1:
            s = gs.apply(lambda x: x.shift(1).expanding().median())
        else:
            s = gs.apply(lambda x: x.shift(1).rolling(window=window).median())

        s = s.reset_index(level=group_cols, drop=True)
    else:
        s = gs.transform("mean")

    df[f"median_{col}{"_progressive" if progressive else ""}_{window}"] = s
    df = df.dropna(
        subset=[f"median_{col}{"_progressive" if progressive else ""}_{window}"]
    )
    return drop_temp_cols(df)


def append_last_n_races(
    df: pd.DataFrame,
    col: str,
    *,
    in_season: bool = True,
    window: int = 3,
) -> pd.DataFrame:
    """
    Appends the last N values of a specified column for each driver as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing driver race data, including 'raceId' and 'driverId'.
        lookback (int): Number of past races to include. Defaults to 5.
        col (str): Column name from which to extract historical values.

    Returns:
        pd.DataFrame: DataFrame with additional columns for each of the last N values.
    """
    for i in range(1, window + 1):
        df[f"last_{col}_{i}"] = df.groupby(
            ["year", "driverId"] if in_season else "driverId"
        )[col].shift(i)

    df = df.dropna(subset=[col for col in df.columns if col.startswith("last_")])
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
    df["tmp_position"] = df["positionText"].apply(
        lambda x: get_position_category(x, pos_cat)
    )
    category_cls: Enum = {
        "loose": LoosePositionCategory,
        "tight": TightPositionCategory,
        "top3": Top3PositionCategory,
        "winner": WinnerPositionCategory,
    }[pos_cat]
    dfs: list[pd.DataFrame] = []  # storing for concatenation

    for _, group in df.groupby(["driverId", "year"] if in_season else "driverId"):
        counts: dict[str, int] = {k: 0 for k in category_cls._value2member_map_}
        vals = counts.values()
        prop_series: dict[str, list[int]] = {k: [] for k in counts}
        
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
            group[f"propensity_{pos_cat}{"_in_season" if in_season else ""}_{k}"] = v

        dfs.append(group)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=[col for col in df.columns if col.startswith("propensity_")])
    return drop_temp_cols(df.sort_values("raceId", axis=0).reset_index(drop=True))


def append_last_season_wins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends the number of wins from the previous season for each driver.

    For each race entry, this function adds a new column `prev_season_wins` indicating how many races
    the driver won in the prior season. If no data exists for the previous season, the value is NaN.

    Args:
        df (pd.DataFrame): Input DataFrame with 'driverId', 'year', 'wins', and 'raceId' columns.

    Returns:
        pd.DataFrame: DataFrame with an added `prev_season_wins` column, sorted by 'raceId'.
    """
    dfs: list[pd.DataFrame] = []

    for _, group in df.groupby("driverId"):
        for _, group2 in group.groupby("year"):
            group2["prev_season_wins"] = group[
                (group["year"] == group2["year"].unique()[0] - 1)
                & (group["driverId"] == group2["driverId"].unique()[0])
            ]["wins"].max()
            dfs.append(group2)

    return (
        pd.concat(dfs, ignore_index=True).sort_values("raceId").reset_index(drop=True)
    )


def append_dnf_count(df: pd.DataFrame, *, window: int = 3) -> pd.DataFrame:
    df = df.sort_values(by=["year", "driverId"]).copy()
    col = f"dnf_count_{window}"

    def calc_rolling(group: pd.DataFrame) -> pd.DataFrame:
        dnf_mask = pd.to_numeric(
            (group["statusId"].isin([*range(2, 11)])).shift(1), errors="coerce"
        ).fillna(0)

        if window < 1:
            group[col] = dnf_mask.expanding().sum()
        else:
            group[col] = dnf_mask.rolling(window=window).sum()
        return group

    df = (
        df.groupby(["year", "driverId"])
        .apply(calc_rolling)
        .sort_values("raceId", axis=0)
    )
    return df


def append_field_pos_delta(
    df: pd.DataFrame,
    *,
    in_season: bool = True,
    window: int = 3,
) -> pd.DataFrame:
    """
    Appends the delta between a driver's average finishing position and the field's average per race.

    For each driver and race, this function computes a moving or expanding average of their previous
    finishing positions and compares it to the average of all drivers' averages in that race.
    The resulting delta indicates whether the driver is performing above or below their peers.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'position_numeric', 'driverId',
            'year', and 'raceId' columns.
        in_season (bool): If True, computes moving averages within each season separately.
            If False, computes across all available races per driver (default: True).
        window (int): Size of the rolling window for averaging. If <1, uses expanding average.

    Returns:
        pd.DataFrame: DataFrame with an added column representing delta difference.
    """
    dfs: list[pd.DataFrame] = []

    if in_season:
        group_cols = "raceId"
    else:
        group_cols = ["year", "raceId"]

    if window < 1:
        lf = lambda x: x.shift(1).expanding().mean()
    else:
        lf = lambda x: x.shift(1).rolling(window=window).mean()

    if in_season:  # Calculate average position on year, driverId
        groups: list[pd.DataFrame] = []
        for _, ygroup in df.groupby("year"):
            ygroup = ygroup.copy()
            ygroup["tmp_avg_pos"] = (
                ygroup.sort_values("raceId")
                .groupby("driverId")["position_numeric"]
                .transform(lf)
            )
            ygroup = ygroup.dropna(subset=["tmp_avg_pos"])
            groups.append(ygroup)
        df = pd.concat(groups)

    else:  # Calculate averge position only on driverId
        df["tmp_avg_pos"] = df.groupby("driverId")["position_numeric"].transform(lf)
        df = df.dropna(subset=["tmp_avg_pos"])

    for _, rgroup in df.groupby(group_cols):
        avg = rgroup["tmp_avg_pos"].mean()
        rgroup[f"avg_pos_delta{"_in_season" if in_season else ""}_{window}"] = rgroup[
            "tmp_avg_pos"
        ].apply(lambda x: avg - x)
        dfs.append(rgroup)

    return drop_temp_cols(pd.concat(dfs).sort_values("raceId"))


def append_category_streak(
    df: pd.DataFrame, *, in_season: bool = True, window: 3
) -> pd.DataFrame: ...
