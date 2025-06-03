from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class Constructor:
    name: str
    nationality: str
    constructor_ref: str
    constructor_id: Optional[int] = None  # To be intiialised by DataPoller


@dataclass
class Time:
    time: str
    millis: Optional[int] = None


@dataclass
class FastestLapTime:
    time: str


@dataclass
class AverageSpeed:
    units: Optional[str] = None
    speed: Optional[float] = None


@dataclass
class FastestLap:
    rank: int
    lap: int
    time: FastestLapTime
    average_speed: AverageSpeed


@dataclass
class Driver:
    driver_ref: str
    permanent_number: str
    code: str
    nationality: str
    dob: datetime
    driver_id: Optional[int] = None  # To be initialised be DataPoller.
    constructor: Optional[Constructor] = (
        None  # Optional: not always tied in non-race result
    )


@dataclass
class QualiResult:
    year: int
    round_: int
    circuit_id: int
    driver: Driver
    position: int
    position_text: str
    q1: Optional[float] = None
    q2: Optional[float] = None
    q3: Optional[float] = None


@dataclass
class SprintResult:
    year: int
    round_: int
    circuit_id: int
    driver: Driver
    grid: int
    points: int
    position: int
    position_text: str


@dataclass
class GrandPrixResult:
    year: int
    round_: int
    circuit_id: int
    number: int
    position: int
    position_text: str
    points: float
    driver: Driver
    grid: int
    laps: int
    status: str
    time: Optional[Time] = None
    fastest_lap: Optional[FastestLap] = None
