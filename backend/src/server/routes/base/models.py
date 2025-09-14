from pydantic import BaseModel, field_validator


class Driver(BaseModel):
    driver_name: str
    constructor: str
    nationality: str


class PredictionsResponse(BaseModel):
    driver: Driver
    predicted_position: str
    predicted_probability: float

    @field_validator("predicted_probability")
    def round_prob(cls, v):
        return round(v, 2)
