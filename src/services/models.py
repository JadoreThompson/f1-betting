from dataclasses import dataclass
from datetime import datetime


@dataclass
class Constructor:
    name: str
    nationality: str
    constructor_ref: str
    constructor_id: int | None = None  # To be initialised by DataPoller


@dataclass
class Time:
    time: str
    millis: int | None = None


@dataclass
class FastestLapTime:
    time: str


@dataclass
class AverageSpeed:
    units: str | None = None
    speed: float | None = None


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
    driver_id: int | None = None  # Drivers DB ID.
    constructor: Constructor | None = None  # Not always tied in non-race result


@dataclass
class QualiResult:
    year: int
    round_: int
    circuit_id: int
    driver: Driver
    position: int
    position_text: str
    q1: float | None = None
    q2: float | None = None
    q3: float | None = None


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
    points: int
    driver: Driver
    grid: int
    laps: int
    status: str
    time: Time | None = None
    fastest_lap: FastestLap | None = None
