import numpy as np
import pandas as pd

from .clean import get_clean_df


def drop_temp_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop([col for col in df.columns if col.startswith("tmp_")], axis=1)


def append_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'elo' column for each driver in each race based on grid position
    and finishing position. Points are calculated as:
        points = (n_drivers - grid) + (n_drivers - position)

    Args:
        df: DataFrame with columns 'race_id', 'driver_id', 'grid', 'position'

    Returns:
        DataFrame with a new 'elo' column.
    """
    df = df.copy()
    df = df.sort_values(["year", "round"])

    def compute_points(group: pd.DataFrame) -> pd.DataFrame:
        n_drivers = len(group)
        positions = group["position"].apply(
            lambda x: x if x != float("-inf") else n_drivers * 2
        )
        group["elo"] = (n_drivers - group["grid"]) + (n_drivers - positions)
        return group

    df = df.groupby("race_id", group_keys=False).apply(compute_points)
    df["elo"] = df["elo"].shift(1)
    return df


def append_prev_wins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["year", "round"])

    df["win_flag"] = (df["position"] == 1).astype(int)
    df["prev_wins"] = df.groupby("driver_id")["win_flag"].cumsum().shift(1)

    df = df.drop(columns=["win_flag"])
    return df


def append_prev_season_wins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["year", "round"])

    df["win_flag"] = (df["position"] == 1).astype(int)
    df["prev_wins"] = df.groupby(["driver_id", "year"])["win_flag"].cumsum().shift(1)

    df = df.drop(columns=["win_flag"])
    return df


def get_features_df(df: pd.DataFrame | None = None):
    if df is None:
        df = get_clean_df()
        
    df = df[df["year"] >= 2017].copy()

    df = append_elo(df)
    df = append_prev_wins(df)
    df = append_prev_season_wins(df)

    df = df.drop(
        axis=1,
        columns=[
            "constructor_standings_id",
            "constructor_id",
            'constructor_ref',
            "qualify_id",
            "circuit_id",
            "driver_id",
            "race_id",
            "result_id",
            "status_id",
            "driver_ref",
            'laps',
            "time",
            "date",
            "quali_date",
            "sprint_date",
            "q1",
            "q2",
            "q3",
            "fp1_time",
            "fp2_time",
            "fp3_time",
            "fp1_date",
            "fp2_date",
            "fp3_date",
            "fastest_lap",
            "quali_time",
            "sprint_time",
            "milliseconds",
            "points",
            "position",
            "position_order",
            "constructor_points",
            "constructor_position",
            "constructor_position_text",
            "fastest_lap_speed",
            "fastest_lap_time",
            "grid",
            "code",
            "dob",
            "name",
            "forename",
            "surname",
            "nationality",
            "url",
            "rank",
            "number",
        ],
    )
    return df
