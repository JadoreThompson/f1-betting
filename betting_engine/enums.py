from enum import Enum


class OrderStatus(int, Enum):
    """Order Status to be used within the matching engine."""
    PENDING = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3
    CLOSED = 4


class Topic(int, Enum):
    CREATE = 0
    CLOSE = 1
    SETTLE = 2