from datetime import datetime
from pydantic import BaseModel
from enums import MarketCategory, Side, TransactionType


class UserSummary(BaseModel):
    total_pos_value: float
    pnl: float
    volume: float
    markets_traded: int


class Pagination(BaseModel):
    has_next: bool
    has_prev: bool
    current_page: int
    total_pages: int
    total_quantity: int


class Position(BaseModel):
    title: str
    category: MarketCategory
    odds: str
    side: Side
    amount: float
    created_at: datetime


class Transaction(BaseModel):
    type: TransactionType
    amount: float
    address: str
    title: str
    category: MarketCategory


class PositionResponse(BaseModel):
    data: list[Position]
    pagination: Pagination


class TransactionResponse(BaseModel):
    data: list[Transaction]
    pagination: Pagination
