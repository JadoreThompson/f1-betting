import numpy as np
import os
import pandas as pd
from .config import DPATH


def apply_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply types to the dataset."""
    df["race_id"] = df["race_id"].astype("str")
    df["driver_id"] = df["driver_id"].astype("str")
    df["year"] = df["year"].astype("str")
    return df


def prepare_train_dataset(min_year: int = 2023) -> pd.DataFrame:
    results_df = pd.read_csv(os.path.join(DPATH, "results.csv"))[
        ["raceId", "driverId", "position", "positionText"]
    ]
    results_df["position"] = (
        pd.to_numeric(results_df["position"], errors="coerce").fillna(0).astype("int")
    )

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[
        [
            "raceId",
            "year",
        ]
    ]
    races_df = races_df[races_df["year"] >= min_year]

    qualifying_df = pd.read_csv(os.path.join(DPATH, "qualifying.csv"))[
        ["raceId", "driverId", "position"]
    ]

    df = races_df.merge(results_df, on=["raceId"], suffixes=("_race", "_result"))
    df = df.merge(qualifying_df, on=["raceId", "driverId"], suffixes=("", "_quali"))
    df.columns = [
        col.replace("Id", "_id").replace("Text", "_text") for col in df.columns
    ]

    df = apply_types(df)
    df = df.dropna()
    return df


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
    return hour * 3600 + minute * 60 + second


def sma(s: pd.Series, df: pd.DataFrame, column: str, window: int = 5) -> float:
    if s.name < window:
        return np.nan

    nums = [df.loc[s.name - i][column] for i in range(1, window + 1)]
    return sum(nums) / len(nums)
