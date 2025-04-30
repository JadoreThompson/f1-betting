import json
import xml.etree.ElementTree as ET

from typing import Any, Literal, TypedDict, TypeVar
from httpx import AsyncClient

# from .models import ParsedQualiData, ParsedRaceData, ConstructedRaceData
from .typing import Dataset, ParsedQualiData, ParsedRaceData, ConstructedRaceData

ERGAST_BASE = "https://ergast.com/api/f1"
ERGAST_NS = {"mrd": "http://ergast.com/mrd/1.5"}


# with open("file.xml", "w") as f:
#     f.write(s)


class Data(TypedDict):
    real: str  # real position the driver came
    data: list[str] | str


def parse_quali_reuslts(s: str) -> dict[str, ParsedQualiData]:
    root = ET.fromstring(s)
    return {
        result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]: ParsedQualiData(
            position=result.attrib["position"]
        )
        for result in root.findall(".//mrd:QualifyingResult", ERGAST_NS)
    }


async def fetch_quali_results(
    year: int = 2024, round_: int | None = None
) -> dict[str, ParsedQualiData]:
    async with AsyncClient() as c:
        rsp = await c.get(
            ERGAST_BASE
            + f"/{year}/{f"{round_}/" if round_ is not None else ""}qualifying"
        )
        return parse_quali_reuslts(rsp.text)


async def build_qualifying_dataset(
    year: int = 2024, first_round: int = 5, last_round: int = 20
) -> Dataset[ParsedQualiData]:
    return {
        i: await fetch_quali_results(year, i) for i in range(first_round, last_round)
    }

    # quali_data: dict[int, dict[str, str]] = {
    #     i: await fetch_quali_results(year, i) for i in range(first_round, last_round)
    # }
    # return quali_data

    # datasets: dict[int, dict[str, str]] = {}

    # for r in quali_data:
    #     datasets[r] = {}
    #     for name in quali_data[r]:
    #         datasets[r][name] = {"data": quali_data[r][name]}

    # return datasets


def parse_race_results(
    # s: str, key: Literal["positionText", "grid"] = "positionText"
    s: str,
) -> dict[str, ParsedRaceData]:
    # with open("file.xml", "w") as f:
    #     f.write(s)
    root = ET.fromstring(s)
    results: dict[str, str] = {}

    for result in root.findall(".//mrd:Result", ERGAST_NS):
        # if key == "positionText":
        #     val = result.attrib[key]
        # else:
        #     val = result.findtext("mrd://Grid", ERGAST_NS)

        # results[result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]] = val
        # results[result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]] = {
        #     "grid": result.findtext("mrd://Grid", ERGAST_NS),
        #     "position": result.attrib[key]
        # }
        results[result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]] = (
            ParsedRaceData(
                grid=result.findtext("mrd:Grid", namespaces=ERGAST_NS),
                position=result.attrib["positionText"],
                constructor_name=result.find(
                    "mrd:Constructor", namespaces=ERGAST_NS
                ).attrib["constructorId"],
            )
        )

    return results
    # return {
    #     result.find("mrd:Driver", ERGAST_NS).attrib["driverId"]: result.attrib[key]
    #     for result in root.findall(".//mrd:Result", ERGAST_NS)
    # }


async def fetch_race_results(
    year: int, round_: int | None = None
) -> dict[str, ParsedRaceData]:
    async with AsyncClient() as c:
        rsp = await c.get(
            ERGAST_BASE + f"/{year}/{f"{round_}/" if round_ is not None else ""}results"
        )
        return parse_race_results(rsp.text)


# async def fetch_grid_positions(
#     year: int, round_: int | None = None
# ) -> dict[str, RawRaceData]:
#     async with AsyncClient() as c:
#         rsp = await c.get(
#             ERGAST_BASE + f"/{year}/{f"{round_}/" if round_ is not None else ""}results"
#         )
#         return parse_race_results(rsp.text, "grid")


async def build_last_races_dataset(
    year: int = 2024,
    rounds: int = 20,
    last_n_races: int = 5,
) -> Dataset[ConstructedRaceData]:
    race_data: Dataset[ParsedRaceData] = {
        i: await fetch_race_results(year, i + 1) for i in range(rounds)
    }

    datasets: Dataset[ConstructedRaceData] = {}

    for r in race_data:
        if r < last_n_races:
            continue

        datasets[r] = {}

        for name in race_data[r]:
            # datasets[r][name] = {
            #     "data": [
            #         race_data[r - i].get(name, "") for i in range(1, last_n_races + 1)
            #     ],
            #     "real": race_data[r][name],
            # }
            # datasets[r][name] = ParsedRaceData()

            prev_positions: list[str] = []

            for i in range(1, last_n_races + 1):
                if prev := race_data[r - i].get(name):
                    prev_positions.append(prev.position)
                else:
                    prev_positions.append("0")

            datasets[r][name] = ConstructedRaceData(
                prev_positions=prev_positions,
                grid=race_data[r][name].grid,
                real=race_data[r][name].position,
                constructor_name=race_data[r][name].constructor_name,
            )

    # json.dump(datasets, open("file.json", "w"), indent=4)
    return datasets


# def merge_last_n_races_and_quali(
#     last_n_races_data: dict[int, dict[int, dict[str, list[str]]]],
#     quali_data: dict[int, dict[int, dict[str, str]]],
# ) -> None:
#     """
#     Adds the qualifying data to the last n races data. Note: it's up to the developer
#     to ensure each dataset of the same year and contain the same rounds.

#     Args:
#         last_n_races_data (dict[int, dict[int, dict[str, list  |  str]]])
#         quali_data (dict[int, dict[int, dict[str, str]]])

#     Raises:
#         ValueError: Datasets have different lengths
#     """
#     if len(last_n_races_data) != len(quali_data):
#         raise ValueError(
#             f"Both datasets must be of the same length {len(last_n_races_data)}, {len(quali_data)}"
#         )

#     for r in last_n_races_data:
#         for driver in last_n_races_data[r]:
#             last_n_races_data[r][driver]["data"].insert(
#                 0, quali_data[r].get(driver, {}).get("data")
#             )
