from enum import Enum


class OrderStatus(str, Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class BetStatus(str, Enum):
    """Client facing bet status."""

    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"
    SETTLED = "settled"



class Topic(int, Enum):
    CREATE = 0
    CLOSE = 1