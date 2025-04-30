from dataclasses import dataclass
from typing import TypeVar, TypedDict


T = TypeVar("T")
Dataset = dict[int, dict[str, T]]


class Data(TypedDict):
    real: str  # real position the driver came
    data: list[str] | str


@dataclass
class ParsedRaceData:
    grid: str
    position: str
    constructor_name: str
    name: str


@dataclass
class ParsedQualiData:
    position: str
    name: str


@dataclass
class ConstructedRaceData:
    prev_positions: list[str]
    grid: str
    constructor_name: str
    name: str
    real: str
    # sma: float | None


def default_constructed_race_data(last_n_races: int, name: str) -> ConstructedRaceData:
    return (
        ConstructedRaceData(
            prev_positions=["0"] * last_n_races,
            grid="0",
            constructor_name="",
            name=name,
            real="0",
        ),
    )
