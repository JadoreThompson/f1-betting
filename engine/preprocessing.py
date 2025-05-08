import os
import numpy as np
import pandas as pd
from .config import DPATH


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


def merge_datasets() -> pd.DataFrame:
    driver_standings_df = pd.read_csv(os.path.join(DPATH, "driver_standings.csv"))[
        ["raceId", "driverId", "points", "position", "wins"]
    ]

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

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[
        ["raceId", "circuitId", "year"]
    ]

    df = races_df.merge(results_df, on=["raceId"]).merge(
        driver_standings_df, on=["raceId", "driverId"], suffixes=("", "_standings")
    )

    for key in ("points", "wins", "position_standings"):
        df[f"prev_{key}"] = df.groupby("driverId")[key].shift(1)

    for key in ("driverId", "circuitId", "constructorId"):
        df[key] = df[key].astype("str")

    df["position_numeric"] = pd.to_numeric(df["position"], errors="coerce").fillna(0)
    return df
