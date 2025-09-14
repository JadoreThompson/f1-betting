import os
import numpy as np
import pandas as pd
from functools import lru_cache


def parse_column(value: str):
    """Parses camel case to snake case."""
    i = 0
    upper_idxs = []

    for i in range(len(value)):
        if value[i].isupper():
            upper_idxs.append(i)

    if not upper_idxs:
        return value

    upper_idxs.append(len(value))

    parts = []
    prev = 0

    for idx in upper_idxs:
        parts.append(value[prev:idx])
        prev = idx

    return "_".join(p.lower().strip() for p in parts)


@lru_cache
def load_datasets():
    folder = os.path.join(os.path.dirname(__file__), "datasets")

    drivers = pd.read_csv(os.path.join(folder, "drivers.csv"))
    qualifying = pd.read_csv(os.path.join(folder, "qualifying.csv"))
    constructors = pd.read_csv(os.path.join(folder, "constructors.csv"))
    constructor_standings = pd.read_csv(
        os.path.join(folder, "constructor_standings.csv")
    )
    races = pd.read_csv(os.path.join(folder, "races.csv"))
    results = pd.read_csv(os.path.join(folder, "results.csv"))

    # circuits = pd.read_csv(os.path.join(folder, "circuits.csv"))
    # constructor_results = pd.read_csv(os.path.join(folder, "constructor_results.csv"))
    # driver_standings = pd.read_csv(os.path.join(folder, "driver_standings.csv"))
    # lap_times = pd.read_csv(os.path.join(folder, "lap_times.csv"))
    # pit_stops = pd.read_csv(os.path.join(folder, "pit_stops.csv"))
    # seasons = pd.read_csv(os.path.join(folder, "seasons.csv"))
    # sprint_results = pd.read_csv(os.path.join(folder, "sprint_results.csv"))
    # status = pd.read_csv(os.path.join(folder, "status.csv"))

    for df in (
        constructors,
        constructor_standings,
        drivers,
        qualifying,
        races,
        results,
    ):
        df.columns = [parse_column(col) for col in df.columns]

    return {
        "constructors": constructors,
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
    constructors = dfs["constructors"]
    constructor_standings = dfs["constructor_standings"]
    constructor_standings = constructor_standings.rename(
        columns={
            "position": "constructor_position",
            "position_text": "constructor_position_text",
            "points": "constructor_points",
            "wins": "constructor_wins",
        }
    )

    drivers = dfs["drivers"]

    qualifying = dfs["qualifying"]
    qualifying = qualifying.rename(columns={"position": "qualifying_position"})

    races = dfs["races"]
    results = dfs["results"]

    # Merge
    df = results.merge(drivers, on="driver_id", suffixes=("", "_drivers"))
    df = df.merge(qualifying, on=["race_id", "driver_id"], suffixes=("", "_qualifying"))
    df = df.merge(races, on=["race_id"], suffixes=("", "_races"))
    df = df.merge(
        constructor_standings,
        on=["constructor_id", "race_id"],
        suffixes=("", "_constructors_standings"),
    )
    df = df.merge(constructors, on="constructor_id", suffixes=("", "_constructors"))
    df = df.sort_values(["year", "round"])

    # for k, df in dfs.items():
    #     print(k)
    #     print(df.columns)
    #     print("")
    df["grid"] = df["grid"].astype(int)
    df["position"] = df["position"].apply(
        lambda x: (
            float("-inf")
            if x == "\\N" or (isinstance(x, float) and np.isnan(x))
            else int(x)
        )
    )

    # Clean
    str_cols = ("driver_id", "race_id", "constructor_id", "status_id", "circuit_id")
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
                    "_constructors",
                    "_constructors_standings",
                    "_qualifying",
                    "_races",
                    "_drivers",
                )
            )
        ]
    )

    # df.columns = [parse_column(col) for col in df.columns]
    return df
