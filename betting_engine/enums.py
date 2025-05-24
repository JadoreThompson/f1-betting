from enum import Enum


class OrderStatus(int, Enum):
    PENDING = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CLOSED = 3
    CANCELLED = 4


class Side(int, Enum):
    BID = 0
    ASK = 1
