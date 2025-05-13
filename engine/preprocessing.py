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
    constructors_df = pd.read_csv(os.path.join(DPATH, "constructors.csv"))[
        ["constructorId", "constructorRef"]
    ]

    constructors_standings_df = pd.read_csv(
        os.path.join(DPATH, "constructor_standings.csv")
    )[["raceId", "constructorId", "points", "position"]]

    drivers_df = pd.read_csv(os.path.join(DPATH, "drivers.csv"))[
        ["driverId", "driverRef", "dob", "nationality"]
    ]

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
            # "statusId",
        ]
    ]

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[
        ["raceId", "circuitId", "year", "round"]
    ]

    df = races_df.merge(results_df, on=["raceId"])
    df = df.merge(
        driver_standings_df,
        on=["raceId", "driverId"],
        suffixes=("", "_driver_standings"),
    )
    df = df.merge(constructors_df, on=["constructorId"])
    df = df.merge(drivers_df, on=["driverId"])
    df = df.merge(
        constructors_standings_df,
        on=["raceId", "constructorId"],
        suffixes=("", "_constructor_standings"),
    )

    df = df.sort_values(["year", "round"])

    for key in (
        "wins",
        "points",
        "position_driver_standings",
    ):
        df[f"prev_{key}"] = df.groupby(["year", "driverId"])[key].transform(
            lambda x: x.shift(1).fillna(0)
        )

    for key in (
        "position_constructor_standings",
        "points_constructor_standings",
    ):
        df[f"prev_{key}"] = df.groupby(["year", "constructorId", "driverId"])[
            key
        ].transform(lambda x: x.shift(1).fillna(0))

    for key in ("driverId", "circuitId", "constructorId"):
        df[key] = df[key].astype("str")

    df["position_numeric"] = pd.to_numeric(df["position"], errors="coerce").fillna(0)
    return df


merge_datasets().to_csv("file.csv", index=False)
