from enum import Enum


class MarketCategory(str, Enum):
    TOP3 = "top3"
    WINNER = "winner"


class MarketStatus(int, Enum):
    OPEN = 0
    CLOSED = 1


class Side(str, Enum):
    BACK = "back"
    LAY = "lay"
