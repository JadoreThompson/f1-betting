from enum import Enum


class OrderStatus(int, Enum):
    PENDING = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3
    CLOSED = 4


class Topic(int, Enum):
    """
    Defines the action to be partaken
    within the matching engine.
    """

    CREATE = 0
    SETTLE = 2
