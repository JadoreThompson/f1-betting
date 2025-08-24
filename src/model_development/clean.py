import os
import pandas as pd
from functools import lru_cache


@lru_cache
def load_datasets():
    folder = os.path.join(os.path.dirname(__file__), "datasets")

    drivers = pd.read_csv(os.path.join(folder, "drivers.csv"))
    qualifying = pd.read_csv(os.path.join(folder, "qualifying.csv"))
    constructor_standings = pd.read_csv(
        os.path.join(folder, "constructor_standings.csv")
    )
    races = pd.read_csv(os.path.join(folder, "races.csv"))
    results = pd.read_csv(os.path.join(folder, "results.csv"))

    # circuits = pd.read_csv(os.path.join(folder, "circuits.csv"))
    # constructor_results = pd.read_csv(os.path.join(folder, "constructor_results.csv"))
    # constructors = pd.read_csv(os.path.join(folder, "constructors.csv"))
    # driver_standings = pd.read_csv(os.path.join(folder, "driver_standings.csv"))
    # lap_times = pd.read_csv(os.path.join(folder, "lap_times.csv"))
    # pit_stops = pd.read_csv(os.path.join(folder, "pit_stops.csv"))
    # seasons = pd.read_csv(os.path.join(folder, "seasons.csv"))
    # sprint_results = pd.read_csv(os.path.join(folder, "sprint_results.csv"))
    # status = pd.read_csv(os.path.join(folder, "status.csv"))

    return {
        "constructor_standings": constructor_standings,
        "drivers": drivers,
        "qualifying": qualifying,
        "races": races,
        "results": results,
    }


def get_clean_df(dfs: dict[str, pd.DataFrame] | None = None):
    if not dfs:
        dfs: dict[str, pd.DataFrame] = load_datasets()

    # Preprocessing
    constructor_standings = dfs["constructor_standings"]
    constructor_standings = constructor_standings.rename(
        columns={
            "position": "constructorPosition",
            "positionText": "constructorPositionText",
            "points": "constructorPoints",
            "wins": "constructorWins",
        }
    )

    drivers = dfs["drivers"]

    qualifying = dfs["qualifying"]
    qualifying = qualifying.rename(columns={"position": "qualifyingPosition"})

    races = dfs["races"]
    results = dfs["results"]

    # Merge
    df = results.merge(drivers, on="driverId", suffixes=("", "_drivers"))
    df = df.merge(qualifying, on=["raceId", "driverId"], suffixes=("", "_qualifying"))
    df = df.merge(races, on=["raceId"], suffixes=("", "_races"))
    df = df.merge(
        constructor_standings,
        on=["constructorId", "raceId"],
        suffixes=("", "_constructors"),
    )
    df = df.sort_values(["year", "round"])

    df["grid"] = df["grid"].astype(int)
    df["position"] = df["position"].apply(
        lambda x: float("-inf") if x == "\\N" else int(x)
    )

    # Clean
    str_cols = ("driverId", "raceId", "constructorId", "statusId", "circuitId")
    for col in str_cols:
        df[col] = df[col].astype(str)

    unique_cols = set(df.columns)
    df = df[[*unique_cols]]

    df = df.drop(
        columns=[
            col
            for col in df.columns
            if any(
                col.endswith(suffix)
                for suffix in (
                    "_results",
                    "_constructor",
                    "_qualifying",
                    "_races",
                    "_drivers",
                )
            )
        ]
    )

    return df
