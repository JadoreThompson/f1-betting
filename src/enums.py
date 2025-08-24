from enum import Enum


class PredictionStatus(str, Enum):
    OPEN = 'open'
    CLOSED = 'closed'