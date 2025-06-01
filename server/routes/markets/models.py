from typing import Any
from pydantic import BaseModel


class MarketResponse(BaseModel):
    titles: list[str]
    winners: list[list[Any]]
    top3: list[list[Any]]
