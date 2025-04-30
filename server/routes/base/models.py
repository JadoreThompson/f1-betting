from typing import Any, TypedDict
from pydantic import BaseModel, Field


class RaceResult(BaseModel):
    position: int
    position_text: str = Field(alias="positionText")
    driver_id: str = Field(alias="driverId")
    constructor_id: str = Field(alias="constructorId")
    grid: int
    laps: int


class Driver(BaseModel):
    name: str
    number: str


class QualifiedDriver(Driver):
    quali_position: Any
