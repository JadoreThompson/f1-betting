import asyncio
import math
import os
import pandas as pd
import xml.etree.ElementTree as ET

from collections import defaultdict
from httpx import AsyncClient
from typing import Iterable, Literal

from config import BPATH
from engine.typing import LoosePositionCategory, TightPositionCategory
from engine.utils import parse_quali_times, get_position_category
from .typing import (
    Dataset,
    ParsedQualiData,
    ParsedRaceData,
    ConstructedRaceData,
)

ERGAST_BASE = "https://ergast.com/api/f1"
ERGAST_NS = {"mrd": "http://ergast.com/mrd/1.5"}
DPATH = os.path.join(BPATH, "engine", "datasets")

# with open("file.xml", "w") as f:
#     f.write(s)


def parse_quali_results_v1(s: str) -> dict[str, ParsedQualiData]:
    """
    Parses a qualifying XML string into a dictionary of ParsedQualiData objects.

    Args:
        s (str): XML string containing qualifying results.

    Returns:
        dict[str, ParsedQualiData]: Mapping of driver ID to ParsedQualiData.
    """

    root = ET.fromstring(s)
    return {
        result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]: ParsedQualiData(
            position=result.attrib["position"],
            name=result.find("mrd:Driver", ERGAST_NS).attrib["driverId"],
            q3_secs=parse_quali_times(result.findtext("mrd:Q3", namespaces=ERGAST_NS)),
            q2_secs=parse_quali_times(result.findtext("mrd:Q2", namespaces=ERGAST_NS)),
            q1_secs=parse_quali_times(result.findtext("mrd:Q1", namespaces=ERGAST_NS)),
        )
        for result in root.findall(".//mrd:QualifyingResult", ERGAST_NS)
    }


def parse_quali_results_v2(df: pd.DataFrame) -> dict[str, ParsedQualiData]:
    """
    Parses a qualifying DataFrame into a dictionary of ParsedQualiData objects.

    Args:
        df (pd.DataFrame): DataFrame containing qualifying data.

    Returns:
        dict[str, ParsedQualiData]: Mapping of driver ID to ParsedQualiData.
    """

    return {
        row["driverId"]: ParsedQualiData(
            position=row["position"],
            name=row["driverId"],
            q3_secs=parse_quali_times(row["q3"]),
            q2_secs=parse_quali_times(row["q2"]),
            q1_secs=parse_quali_times(row["q1"]),
        )
        for _, row in df.iterrows()
    }


async def fetch_quali_results_v1(
    year: int = 2020, round_: int | None = None
) -> dict[str, ParsedQualiData]:
    """
    Fetches qualifying results from the Ergast API and parses them using XML.

    Args:
        year (int): Season year. Defaults to 2020.
        round_ (int | None): Race round number. If None, fetches latest round.

    Returns:
        dict[str, ParsedQualiData]: Parsed qualifying data.
    """
    async with AsyncClient() as c:
        rsp = await c.get(
            ERGAST_BASE
            + f"/{year}/{f"{round_}/" if round_ is not None else ""}qualifying"
        )
        return parse_quali_results_v1(rsp.text)


def fetch_quali_results_v2(
    year: int = 2020, round_: int = 1
) -> dict[str, ParsedQualiData]:
    """
    Fetches qualifying results from local CSV datasets and parses them.

    Args:
        year (int): Season year.
        round_ (int): Race round number.

    Returns:
        dict[str, ParsedQualiData]: Parsed qualifying data.
    """

    quali_df = pd.read_csv(os.path.join(DPATH, "qualifying.csv"))

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[
        ["raceId", "round", "year"]
    ]

    df = quali_df.merge(races_df, on=["raceId"])
    df = df[
        (df["year"] == year) & (df["round"] == min(max(*df["round"].unique()), round_))
    ]
    return parse_quali_results_v2(df)


async def build_qualifying_dataset(
    rounds: int, years: Iterable[int] = None, version: Literal["v1", "v2"] = "v2"
) -> Dataset[ParsedQualiData]:
    """
    Builds a dataset of qualifying results for multiple years and rounds.

    Args:
        rounds (int): Number of rounds per year.
        years (Iterable[int] | None): List of years. Defaults to [2020].
        version (Literal["v1", "v2"]): Whether to use v1 (API) or v2 (CSV) fetcher.

    Returns:
        Dataset[ParsedQualiData]: A dataset of qualifying results keyed by round index.
    """

    async def helper(year: int) -> Dataset[ParsedRaceData]:
        if version == "v1":
            return {j: await fetch_quali_results_v1(year, j + 1) for j in range(rounds)}
        else:
            return {j: fetch_quali_results_v2(year, j + 1) for j in range(rounds)}

    if years is None:
        years = [2020]

    result = await asyncio.gather(*[helper(years[i]) for i in range(len(years))])

    rtn_value: Dataset[ParsedQualiData] = {}
    for i in range(len(result)):
        for key in result[i]:
            rtn_value[key + (i * (rounds))] = result[i][key]

    return rtn_value


def parse_race_results_v1(
    s: str,
) -> dict[str, ParsedRaceData]:
    """
    Parses a race XML string into a dictionary of ParsedRaceData objects.

    Args:
        s (str): XML string containing race results.

    Returns:
        dict[str, ParsedRaceData]: Mapping of driver ID to ParsedRaceData.
    """

    root = ET.fromstring(s)
    results: dict[str, str] = {}

    for result in root.findall(".//mrd:Result", ERGAST_NS):
        name = result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]
        results[name] = ParsedRaceData(
            grid=result.findtext("mrd:Grid", namespaces=ERGAST_NS),
            position=result.attrib["positionText"],
            constructor_name=result.find(
                "mrd:Constructor", namespaces=ERGAST_NS
            ).attrib["constructorId"],
            name=name,
        )

    return results


def parse_race_results_v2(df: pd.DataFrame) -> dict[str, ParsedRaceData]:
    """
    Parses a race results DataFrame into a dictionary of ParsedRaceData objects.

    Args:
        df (pd.DataFrame): DataFrame containing race results.

    Returns:
        dict[str, ParsedRaceData]: Mapping of driver ID to ParsedRaceData.
    """

    return {
        row["driverId"]: ParsedRaceData(
            position=row["position"],
            grid=row["grid"],
            constructor_name=row["constructorRef"],
            name=row["driverId"],
        )
        for _, row in df.iterrows()
    }


async def fetch_race_results_v1(
    year: int, round_: int | None = None
) -> dict[str, ParsedRaceData]:
    """
    Fetches race results from the Ergast API and parses them using XML.

    Args:
        year (int): Season year.
        round_ (int | None): Race round number. If None, fetches latest round.

    Returns:
        dict[str, ParsedRaceData]: Parsed race data.
    """
    async with AsyncClient() as c:
        rsp = await c.get(
            ERGAST_BASE + f"/{year}/{f"{round_}/" if round_ is not None else ""}results"
        )
        return parse_race_results_v1(rsp.text)


def fetch_race_results_v2(
    year: int = 2020, round_: int = 1
) -> dict[str, ParsedRaceData]:
    """
    Fetches race results from local CSV datasets and parses them.

    Args:
        year (int): Season year. Defaults to 2020.
        round_ (int): Race round number.

    Returns:
        dict[str, ParsedRaceData]: Parsed race data.
    """

    constructors_df = pd.read_csv(os.path.join(DPATH, "constructors.csv"))[
        ["constructorId", "constructorRef"]
    ]

    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))[
        ["raceId", "round", "year"]
    ]

    results_df = pd.read_csv(os.path.join(DPATH, "results.csv"))

    df = results_df.merge(constructors_df, on=["constructorId"])
    df = df.merge(races_df, on=["raceId"])
    df = df[
        (df["year"] == year) & (df["round"] == min(max(*df["round"].unique()), round_))
    ]
    return parse_race_results_v2(df)


def construct_parsed_race_data(
    data: Dataset[ParsedRaceData], last_n_races: int
) -> Dataset[ConstructedRaceData]:
    datasets: Dataset[ConstructedRaceData] = defaultdict(dict)

    for r in data:
        if r < last_n_races:
            continue

        for name in data[r]:
            prev_positions: list[str] = []

            for i in range(1, last_n_races + 1):
                if prev := data[r - i].get(name):
                    prev_positions.append(prev.position)
                else:
                    prev_positions.append("0")

            d = data[r][name]
            datasets[r][name] = ConstructedRaceData(
                prev_positions=prev_positions,
                grid=d.grid,
                real=d.position,
                constructor_name=d.constructor_name,
                name=d.name,
            )

    return datasets


async def build_last_races_dataset(
    rounds: int,
    last_n_races: int,
    years: Iterable[int] = None,
    version: Literal["v1", "v2"] = "v2",
) -> Dataset[ConstructedRaceData]:
    """
    Builds a dataset that includes information about the previous N races for each driver.

    Args:
        rounds (int): Number of rounds per year.
        last_n_races (int): Number of previous races to include.
        years (Iterable[int] | None): List of years. Defaults to [2020].
        version (Literal["v1", "v2"]): Whether to use v1 (API) or v2 (CSV) fetcher.

    Returns:
        Dataset[ConstructedRaceData]: Dataset including historical performance.
    """

    async def helper(year: int) -> Dataset[ParsedRaceData]:
        if version == "v1":
            return {j: await fetch_race_results_v1(year, j + 1) for j in range(rounds)}
        else:
            return {j: fetch_race_results_v2(year, j + 1) for j in range(rounds)}

    if years is None:
        years = [2020]

    result = await asyncio.gather(*[helper(years[i]) for i in range(len(years))])

    race_data: Dataset[ParsedRaceData] = {}

    for i in range(len(result)):
        for key in result[i]:
            race_data[key + (i * rounds)] = result[i][key]

    # datasets: Dataset[ConstructedRaceData] = defaultdict(dict)

    # for r in race_data:
    #     if r < last_n_races:
    #         continue

    #     for name in race_data[r]:
    #         prev_positions: list[str] = []

    #         for i in range(1, last_n_races + 1):
    #             if prev := race_data[r - i].get(name):
    #                 prev_positions.append(prev.position)
    #             else:
    #                 prev_positions.append("0")

    #         d = race_data[r][name]
    #         datasets[r][name] = ConstructedRaceData(
    #             prev_positions=prev_positions,
    #             grid=d.grid,
    #             real=d.position,
    #             constructor_name=d.constructor_name,
    #             name=d.name,
    #         )

    # return datasets
    return construct_parsed_race_data(race_data, last_n_races)


def get_sma(
    data: ConstructedRaceData,
    last_n_races: int,
    type_: Literal["real", "normalised"] = "real",
    category: str = "loose",
) -> float:
    """
    Calculates the simple moving average (SMA) of a driver's positions over the past N races.

    Args:
        data (ConstructedRaceData): Driver's historical race data.
        last_n_races (int): Number of races to average over.
        type_ (Literal["real", "normalised"]): Whether to use raw positions or normalised categories.
        category (str): Normalisation category to use (if applicable).

    Returns:
        float: Simple moving average of the driver's positions.
    """

    if type_ == "real":
        sma_value: float = sum(
            float(p) if p.isdigit() else 0 for p in data.prev_positions
        )
    else:
        sma_value: float = sum(
            float(p) if p.isdigit() else 0
            for p in [
                get_position_category(pos, category) for pos in data.prev_positions
            ]
        )

    if sma_value:
        sma_value /= last_n_races

    return sma_value


def get_avg_position_move(data: Dataset[ConstructedRaceData], name: str) -> float:
    """
    Computes the average position change (grid vs finish) across races for a given driver.

    Args:
        data (Dataset[ConstructedRaceData]): Dataset of races.
        name (str): Driver's name.

    Returns:
        float: Average number of positions gained (positive) or lost (negative).
    """

    total = 0
    count = 0

    for round_data in data.values():
        race_data = round_data.get(name, ConstructedRaceData.construct(1, name))
        total += int(race_data.grid) - int(
            race_data.real if race_data.real.isdigit() else 0
        )
        count += 1

    if total and count:
        total /= count

    return total


def get_rolling_pos_move(
    data: Dataset[ConstructedRaceData], window: int, round: int, name: str
) -> float:
    """
    Calculates rolling average of position gains/losses over a window of races for a driver.

    Args:
        data (Dataset[ConstructedRaceData]): Dataset of race data.
        window (int): Number of races in the rolling window.
        round (int): Current race round.
        name (str): Driver's name.

    Returns:
        float: Average position change over the window.
    """

    if window < 1:
        raise ValueError("window must be greater than or equal to 1.")

    total = 0.0

    for i in range(1, window + 1):
        prev = data.get(round - i, {}).get(name, ConstructedRaceData.construct(1, name))
        gain = int(prev.grid) - int(prev.real if prev.real.isdigit() else 0)
        total += gain

    if total:
        total /= window

    return total


def get_avg_position(
    data: Dataset[ConstructedRaceData],
    name: str,
    *,
    type_: Literal["real", "loose", "tight"] = "real",
    rolling: bool = False,
    round_: int = None,
    window: int = None,
) -> float:
    if rolling and (window is None or window < 1):
        raise ValueError(
            "window must be greater than or equal to one if rolling is set to true."
        )
    if rolling and round_ is None:
        raise ValueError("round_ must be provided if rolling is set to true.")

    total = 0
    count = 0

    if rolling:
        for i in range(1, window + 1):
            if existing := data.get(round_ - i, {}).get(name):
                count += 1
                if type_ == "real":
                    total += int(existing.real if existing.real.isdigit() else 0)
                else:
                    total += int(get_position_category(existing.real, type_))
    else:
        for r in data:
            if existing := data[r].get(name):
                if type_ == "real":
                    total += int(existing.real if existing.real.isdigit() else 0)
                else:
                    total += int(get_position_category(existing.real, type_))
                count += 1

    if total and count:
        total /= count

    return total


def get_std(data: Dataset[ConstructedRaceData], name: str) -> float:
    vals: list[int] = []

    for r in data:
        if existing := data[r].get(name):
            vals.append(int(existing.real if existing.real.isdigit() else 0))

    try:
        avg = sum(vals) / len(vals)
        diffs = [(v - avg) ** 2 for v in vals]
        return math.sqrt(sum(diffs) / len(diffs))
    except ZeroDivisionError:
        return 0.0


def get_position_propensity(
    data: Dataset[ConstructedRaceData], name: str, type_: Literal["tight", "loose"]
) -> list[float]:
    poscat_map = {
        "tight": {key: 0 for key in TightPositionCategory._value2member_map_},
        "loose": {key: 0 for key in LoosePositionCategory._value2member_map_},
    }[type_]
    # vals = []
    count = 0
    for _, round_data in data.items():
        if existing := round_data.get(name):
            poscat_map[get_position_category(existing.real, type_)] += 1
            count += 1

    vals = []

    for v in poscat_map.values():
        try:
            vals.append(1 / (v / count))
        except ZeroDivisionError:
            vals.append(0.0)
    return vals


def get_features(
    *,
    race_data: Dataset[ConstructedRaceData],
    last_n_races: int,
    round_: int,
    name: str,
    quali_data: Dataset[ParsedQualiData] = None,
) -> list:
    """
    Returns list of features to be inputted into the model.

    Args:
        last_n_data (Dataset[ConstructedRaceData])
        quali_data (Dataset[ParsedQualiData])
        LAST_N_RACES (int)
        round (int)
        name (str)

    Returns:
        list: A list of features for a given driver of a given round.
    """
    dataset = []
    specific_last_n_data = race_data[round_].get(
        name,
        ConstructedRaceData.construct(last_n_races, name),
    )
    if quali_data:
        specific_quali_data = quali_data[round_].get(name, ParsedQualiData())

    dataset.append(int(specific_last_n_data.grid))
    # dataset.append(specific_last_n_data.name)
    # dataset.append(specific_last_n_data.constructor_name)
    # dataset.append(get_position_category(specific_quali_data.position, "loose"))
    # dataset.append(int(specific_quali_data.position))
    # dataset.append(get_sma(specific_last_n_data, LAST_N_RACES))
    # dataset.append(
    #     min(
    #         [
    #             specific_quali_data.q3_secs,
    #             specific_quali_data.q2_secs,
    #             specific_quali_data.q1_secs,
    #         ]
    #     )
    # )
    # dataset.append(
    #     sum(
    #         [
    #             specific_quali_data.q3_secs,
    #             specific_quali_data.q2_secs,
    #             specific_quali_data.q1_secs,
    #         ]
    #     )
    #     / 3
    # )
    # dataset.append(specific_quali_data.q3_secs)
    # dataset.append(specific_quali_data.q2_secs)
    # dataset.append(specific_quali_data.q1_secs)
    # quali_t = int
    # if specific_quali_data:
    #     dataset.append(quali_t(specific_quali_data.position))
    # else:
    #     dataset.append(quali_t(0))
    # dataset.extend(specific_last_n_data.prev_positions)
    # dataset.extend(
    #     [float(p) if p.isdigit() else 0.0 for p in specific_last_n_data.prev_positions]
    # )
    dataset.extend(
        [get_position_category(p, "loose") for p in specific_last_n_data.prev_positions]
    )
    # dataset.append(get_avg_position_move(last_n_data, name))
    # dataset.extend(
    #     [np for p in specific_last_n_data.prev_positions for np in get_position_category(p, "tight")]
    # )
    dataset.append(
        get_avg_position(
            race_data,
            name,
            type_="real",
            # rolling=True,
            # window=3,
            # round_=round_,
        )
    )
    # dataset.append(
    #     get_avg_position(
    #         last_n_data,
    #         name,
    #         type_="tight",
    #         # rolling=True,
    #         # window=3,
    #         # round_=round_,
    #     )
    # )
    # dataset.append(get_rolling_pos_move(last_n_data, LAST_N_RACES, round_, name))
    # print(len(dataset))
    # dataset.append(get_std(last_n_data, name))
    # dataset.extend(get_position_propensity(race_data, name, "loose"))
    return dataset
