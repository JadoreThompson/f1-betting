from dataclasses import dataclass
from typing import TypeVar, TypedDict



T = TypeVar("T")
Dataset = dict[int, dict[str, T]]


@dataclass
class ParsedRaceData:
    grid: str
    position: str
    constructor_name: str
    name: str


@dataclass
class ParsedQualiData:
    position: str = "0"
    name: str = ""
    q3_secs: int = 0
    q2_secs: int = 0
    q1_secs: int = 0


@dataclass
class ConstructedRaceData:
    prev_positions: list[str]
    grid: str
    constructor_name: str
    name: str
    real: str

    @classmethod
    def construct(cls, last_n_races: int, name: str) -> "ConstructedRaceData":
        return cls(
            prev_positions=["0"] * last_n_races,
            grid="0",
            constructor_name="",
            name=name,
            real="0",
        )
