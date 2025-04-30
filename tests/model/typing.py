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


@dataclass
class ParsedQualiData:
    position: str


@dataclass
class ConstructedRaceData:
    prev_positions: list[str]
    grid: str
    constructor_name: str
    real: str
