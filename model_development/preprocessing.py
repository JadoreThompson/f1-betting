import os
from typing import Optional
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


def get_datasets() -> dict:
    """
    Loads all racing-related datasets and returns them in a dictionary.

    Returns:
        dict: A dictionary containing all relevant DataFrames.
    """
    datasets = {
        "circuits": pd.read_csv(os.path.join(DPATH, "circuits.csv"))[
            ["circuitId", "circuitRef"]
        ],
        "constructors": pd.read_csv(os.path.join(DPATH, "constructors.csv"))[
            ["constructorId", "constructorRef"]
        ],
        "constructor_standings": pd.read_csv(
            os.path.join(DPATH, "constructor_standings.csv")
        )[["raceId", "constructorId", "points", "position"]],
        "drivers": pd.read_csv(os.path.join(DPATH, "drivers.csv"))[
            ["driverId", "driverRef", "dob", "nationality"]
        ],
        "driver_standings": pd.read_csv(os.path.join(DPATH, "driver_standings.csv"))[
            ["raceId", "driverId", "points", "position", "wins"]
        ],
        "results": pd.read_csv(os.path.join(DPATH, "results.csv"))[
            [
                "raceId",
                "driverId",
                "constructorId",
                "grid",
                "position",
                "positionText",
                "positionOrder",
                "statusId",
            ]
        ],
        "races": pd.read_csv(os.path.join(DPATH, "races.csv"))[
            ["raceId", "circuitId", "year", "round"]
        ],
        "qualifying": pd.read_csv(os.path.join(DPATH, "qualifying.csv"))[
            ["raceId", "driverId", "position"]
        ],
    }

    return datasets


def merge_datasets(datasets: Optional[dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Merges and processes racing-related datasets.

    Args:
        datasets (dict): Dictionary of DataFrames loaded from get_datasets().

    Returns:
        pd.DataFrame: The merged and processed dataset.
    """
    if datasets is None:
        datasets = get_datasets()
        
    df = datasets["races"].merge(datasets["results"], on="raceId")
    df = df.merge(datasets["circuits"], on="circuitId")
    df = df.merge(
        datasets["driver_standings"],
        on=["raceId", "driverId"],
        suffixes=("", "_driver_standings"),
    )
    df = df.merge(datasets["constructors"], on="constructorId")
    df = df.merge(datasets["drivers"], on="driverId")
    df = df.merge(
        datasets["constructor_standings"],
        on=["raceId", "constructorId"],
        suffixes=("", "_constructor_standings"),
    )
    df = df.merge(
        datasets["qualifying"],
        on=["driverId", "raceId"],
        suffixes=("", "_quali"),
    )
    
    df = df.sort_values(["year", "round"])

    for key in ("wins", "points", "position_driver_standings"):
        df[f"prev_{key}"] = df.groupby(["year", "driverId"])[key].transform(
            lambda x: x.shift(1).fillna(0)
        )

    for key in ("position_constructor_standings", "points_constructor_standings"):
        df[f"prev_{key}"] = df.groupby(["year", "constructorId", "driverId"])[
            key
        ].transform(lambda x: x.shift(1).fillna(0))

    for key in ("driverId", "circuitId", "constructorId"):
        df[key] = df[key].astype("str")

    df["position_numeric"] = pd.to_numeric(df["position"], errors="coerce").fillna(0)

    return df
