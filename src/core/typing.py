from typing import Any

from pydantic import BaseModel, field_validator


class F1ModelStats(BaseModel):
    accuracy: float
    precision: float

    @field_validator("accuracy", "precision")
    def round_vals(cls, v):
        return round(v, 3)


class F1ModelConfig(BaseModel):
    params: dict[str, Any]
    features: list[str]
    stats: F1ModelStats
