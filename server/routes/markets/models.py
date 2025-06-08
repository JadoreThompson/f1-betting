from dataclasses import dataclass
from datetime import datetime
from typing import Any
from pydantic import BaseModel


class MarketResponse(BaseModel):
    titles: list[str]
    winners: list[list[Any]]
    top3: list[list[Any]]


class MarketSummary(BaseModel):
    total_volume: float
    active_bets: int


class NextRace(BaseModel):
    name: str
    datetime: datetime  # with timezone UTC
    round: int


class Overview(BaseModel):
    latest_bet_title: str
    latest_bet_category: str
    latest_bet_amount: float
    most_backed_title: str
    most_backed_category: str
    most_backed_amount: float
