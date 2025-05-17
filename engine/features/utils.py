import math
import time
import numpy as np
import pandas as pd

from enum import Enum
from typing import Callable, Iterable, Literal
from ..typing import (
    LoosePositionCategory,
    TightPositionCategory,
    Top3PositionCategory,
    WinnerPositionCategory,
)


PosCat = Literal["tight", "loose", "winner", "top3"]


def time_it(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        s = time.time()
        res = func(*args, **kwargs)
        print("Total time:", time.time() - s)
        return res

    return wrapper


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
    if pos_cat == "winner":
        if value == "1":
            return WinnerPositionCategory.WINNER.value
        return WinnerPositionCategory.NOT_WINNER.value

    if pos_cat == "top3":
        if not value.isdigit() or int(value) < 1 or int(value) > 3:
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


def append_avg_quali_position_move(
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
        df (pd.DataFrame): DataFrame with 'grid', 'position_quali', 'driverId', and 'year' columns.
        in_season (bool): Whether to compute averages within each season (default: True).
        progressive (bool): Whether to use rolling/expanding mean up to the current race (default: True).
        window (int): Window size for rolling mean if progressive is True. If <1, uses expanding mean.

    Returns:
        pd.DataFrame: Input DataFrame with an added column for average position change.
    """
    if any(key and key not in df.columns for key in ("position_quali", "grid")):
        raise ValueError("position_quali and grid must be in dataframe object.")

    df["tmp_position_change"] = df["position_quali"] - df["positionOrder"]
    final_key = f"avg_quali_position_move{"_in_season" if in_season else ""}{"_progressive" if progressive else ""}_{window}"

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
    df = df.copy()
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
    return drop_temp_cols(df.sort_values(["year", "round"]))


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
    df = df.copy()
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
    return drop_temp_cols(df.sort_values(["year", "round"]))


def append_last_n(
    df: pd.DataFrame,
    col: str,
    *,
    in_season: bool = True,
    window: int = 3,
    typ: str = "str",
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
    if window < 1:
        raise ValueError("window must be >= 1.")

    df = df.sort_values(["year", "round"])
    s = df.groupby(["year", "driverId"] if in_season else "driverId")[col]

    for i in range(1, window + 1):
        df[f"last_{col}_{i}"] = s.shift(i)

    cols = [col for col in df.columns if col.startswith("last_")]
    df = df.dropna(subset=cols)

    for col in cols:
        df[col] = df[col].astype(typ)
    return df


def append_position_propensity(
    df: pd.DataFrame,
    pos_cat: Literal["tight", "loose", "winner", "top3", "real"],
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
        pos_cat (PosCat, "real"): Position category type to use.
        in_season (bool, optional): If True, calculates propensity per driver per year.
            If False, calculates across all years. Defaults to True.

    Returns:
        pd.DataFrame: Original DataFrame with added propensity columns (one for each
                     position category value) and temporary columns removed.
    """
    df = df.copy()

    if pos_cat == "real":
        df["tmp_position"] = df["positionOrder"].astype("str")
        factory: list = df["tmp_position"].unique().tolist()
        factory.append("0")
    else:
        df["tmp_position"] = df["positionText"].apply(
            lambda x: get_position_category(x, pos_cat)
        )
        factory: dict[str, Enum] = {
            "loose": LoosePositionCategory,
            "tight": TightPositionCategory,
            "top3": Top3PositionCategory,
            "winner": WinnerPositionCategory,
        }[pos_cat]._value2member_map_

    dfs: list[pd.DataFrame] = []  # storing for concatenation

    for _, group in df.groupby(["driverId", "year"] if in_season else "driverId"):
        counts: dict[str, int] = {k: 0 for k in factory}
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
    return drop_temp_cols(df.reset_index(drop=True).sort_values(["year", "round"]))


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
        pd.concat(dfs, ignore_index=True)
        .reset_index(drop=True)
        .sort_values(["year", "round"])
    )


def append_dnf_count(
    df: pd.DataFrame, *, in_season: bool = True, window: int = 3
) -> pd.DataFrame:
    df = df.sort_values(["year", "round"])
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

    return (
        df.groupby(["year", "driverId"] if in_season else "driverId")
        .apply(calc_rolling)
        .reset_index(drop=True)
        .sort_values(["year", "round"])
    )


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
    df = df.copy()

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

    return drop_temp_cols(pd.concat(dfs).sort_values(["year", "round"]))


def append_std(df: pd.DataFrame, *, in_season: bool = True, window: 3) -> pd.DataFrame:
    group_cols = ["year", "driverId"] if in_season else ["driverId"]
    col = f"std{'_in_season' if in_season else ''}_{window}"

    df = df.sort_values(by=["year", "round"])

    calc_std: Callable[[Iterable[int]], float] = lambda values: (
        np.nan
        if len(values) < 2 or any(val for val in values == np.nan)
        else np.std(values, ddof=0)
    )

    df["tmp_position_numeric"] = df.groupby(group_cols)["position_numeric"].shift(1)

    if window < 2:
        df[col] = (
            df.groupby(group_cols)["tmp_position_numeric"]
            .expanding()
            .apply(calc_std)
            .reset_index(drop=True)
        )
    else:
        df[col] = (
            df.groupby(group_cols)["tmp_position_numeric"]
            .rolling(window=window)
            .agg(calc_std)
            .reset_index(drop=True)
        )

    return drop_temp_cols(df)


def append_elo(
    df: pd.DataFrame, *, default_elo: float = 1000.0, k: float = 200, p: float = 0.01
) -> pd.DataFrame:
    """
    Computes and appends Elo ratings for drivers across races.

    Elo ratings represent the relative skill levels of drivers, updated after each race
    based on finishing positions. The higher a driver's Elo, the better their historical performance.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'driverId', 'year', 'round',
            and 'positionOrder' columns.
        default_elo (float): Starting Elo rating for all drivers (default: 1000.0).
        k (float): Rating sensitivity factor in likelihood calculation (default: 200).
        p (float): Elo update multiplier controlling the impact of each result (default: 0.01).

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'elo' column representing
            Elo ratings per race per driver.
    """
    df = df.sort_values(["year", "round"])

    df["elo"] = (
        df.groupby("driverId")
        .cumcount()
        .apply(lambda x: default_elo if x == 0 else np.nan)
    )

    current_elos: dict[str, float] = {
        driver_id: default_elo for driver_id in df["driverId"].unique()
    }
    series_list: list[pd.Series] = []

    for _, group in df.groupby(["year", "round"]):
        drivers: list[str] = group["driverId"].unique()
        local_current_elos: dict[str, float] = {d: current_elos[d] for d in drivers}
        positions: dict[str, int] = {
            g["driverId"]: g["positionOrder"] for _, g in group.iterrows()
        }

        for ind, di in enumerate(drivers):
            for dj in drivers[ind + 1 :]:
                if positions[di] < positions[dj]:
                    result = 1
                else:
                    result = 0

                elo_i = current_elos[di]
                elo_j = current_elos[dj]

                # % likelihood of driver i beating driver j
                likelihood = 1 / (1 + 10 ** ((elo_j - elo_i) / k))

                local_current_elos[di] += (elo_j * p) * (result - likelihood)

                if result == 1:
                    local_current_elos[dj] -= (elo_i * p) * (1 - (1 - likelihood))

                else:
                    local_current_elos[dj] += (elo_i * p) * (result - likelihood)

        max_pos = max(positions.values())
        for key, value in local_current_elos.items():
            value = round(value, 2)
            current_elos[key] = value
            s = group[group["driverId"] == key].iloc[0]
            s["elo"] = value
            series_list.append(s)

    final_df = pd.DataFrame(series_list).sort_values(["year", "round"])
    final_df["elo"] = final_df.groupby("driverId")["elo"].shift(1).fillna(default_elo)
    return drop_temp_cols(final_df)


def append_elo_rank_in_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column 'elo_rank_in_race' representing the rank of a driver's Elo
    compared to other drivers in the same race.

    Lower rank means higher Elo (i.e., rank 1 = highest Elo in the race).

    Parameters:
        df (pd.DataFrame): A DataFrame that must include 'raceId' and 'elo'.

    Returns:
        pd.DataFrame: The same DataFrame with 'elo_rank_in_race' column added.
    """
    df = df.copy()
    df["elo_rank_in_race"] = (
        df.groupby("raceId")["elo"]
        .rank(ascending=False, method="min")  # Highest Elo = rank 1
        .astype(int)
    )
    return df


def append_elo_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'elo_percentile' to the DataFrame, representing each driver's
    Elo rating as a proportion of the total Elo ratings within the same race.

    Args:
        df (pd.DataFrame): A DataFrame containing at least the columns 'raceId' and 'elo',
                           where 'raceId' identifies the race and 'elo' is the Elo rating
                           of a driver.

    Returns:
        pd.DataFrame: A copy of the original DataFrame with an additional column
                      'elo_percentile' showing the normalized Elo rating within each race.
    """
    df = df.copy()

    def helper(s: pd.Series) -> float:
        sm = s.sum()
        return s / sm if sm != 0 else 0.0

    df["elo_percentile"] = df.groupby("raceId")["elo"].transform(helper)
    return df


def append_elo_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column 'elo_change' representing the change in Elo rating compared to
    the driver's previous race.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'driverId', 'raceId', and an Elo column.
        elo_col (str): The column name containing Elo ratings.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'elo_change' column.
    """
    df = df.sort_values(["year", "round"])
    df["elo_change"] = df.groupby("driverId")["elo"].diff()
    return df


def append_constructor_encodings(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["constructorRef"], prefix="constructor", dtype=int)
    return pd.concat([df, dummies], axis=1)


def append_nationality_encodings(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["nationality"], prefix="nationality", dtype=int)
    return pd.concat([df, dummies], axis=1)


def append_circuit_encodings(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["circuitRef"], prefix="circuit", dtype=int)
    return pd.concat([df, dummies], axis=1)


def append_drivers_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "round"])
    df["tmp_birth_year"] = df["dob"].apply(lambda x: int(x.split("-")[0]))
    df["age"] = df.apply(lambda x: x["year"] - x["tmp_birth_year"], axis=1)
    return df.drop(columns=["tmp_birth_year"])


def append_confidence(
    df: pd.DataFrame, *, in_season: bool = True, window: int = 3
) -> pd.DataFrame:
    """Appends driver confidence, calculated through
    the average of the reciprocal of the last *window* races.

    Args:
        df (pd.DataFrame): _description_
        in_season (bool, optional): _description_. Defaults to True.
        window (int, optional): _description_. Defaults to 3.

    Returns:
        pd.DataFrame
    """

    def helper(positions: Iterable[int]) -> float:
        if len(positions) < window:
            return np.nan
        return round(sum(1 / pos for pos in positions) / (window or len(positions)), 2)

    dfs: list[pd.DataFrame] = []

    for _, group in df.sort_values(["year", "round"]).groupby(
        ["year", "driverId"] if in_season else "driverId"
    ):
        group[f"confidence_{window}"] = (
            group["positionOrder"].expanding().agg(helper)
            if window < 1
            else group["positionOrder"].rolling(window=window).agg(helper)
        ).shift(1)

        dfs.append(group)

    df = pd.concat(dfs).sort_values(["year", "round"])
    return df


def append_last_n_podiums(
    df: pd.DataFrame, *, in_season: bool = True, window: int = 3
) -> pd.DataFrame:
    """
    Apppends the last n podium finishes.

    Args:
        df (pd.DataFrame)
        in_season (bool, optional): Defaults to True.
        window (int, optional): Defaults to 3.

    Returns:
        pd.DataFrame
    """

    def helper(positions: Iterable[int]) -> int:
        if len(positions) < window:
            return np.nan

        return sum(1 for p in positions if 1 <= p <= 3)

    dfs: list[pd.DataFrame] = []

    for _, group in df.sort_values(["year", "round"]).groupby(
        ["year", "driverId"] if in_season else "driverId"
    ):
        group[f"last_podiums_{window}"] = (
            group["positionOrder"].expanding().apply(helper)
            if window < 1
            else group["positionOrder"].rolling(window=window).apply(helper)
        ).shift(1)
        dfs.append(group)

    return pd.concat(dfs).sort_values(["year", "round"])
