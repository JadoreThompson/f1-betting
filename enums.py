from enum import Enum


class MarketCategory(str, Enum):
    TOP3 = "top3"
    WINNER = "winner"


class MarketStatus(int, Enum):
    OPEN = 0
    CLOSED = 1
    SETTLED  = 2


class Side(str, Enum):
    BACK = "back"
    LAY = "lay"
    
class BetStatus(str, Enum):
    """Client facing bet status."""

    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    SETTLED = "settled"

class TransactionType(str, Enum):
    DEPOSIT = "deposit"
    SETTLE = "settle"
    