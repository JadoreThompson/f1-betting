from enum import Enum


class OrderStatus(int, Enum):
    """Order Status to be used within the matching engine."""
    PENDING = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3
    CLOSED = 4


class BetStatus(str, Enum):
    """Client facing bet status."""

    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    SETTLED = "settled"


class Topic(int, Enum):
    CREATE = 0
    CLOSE = 1
    SETTLE = 2