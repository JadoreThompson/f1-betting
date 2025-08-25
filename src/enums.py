from enum import Enum


class PredictionStatus(str, Enum):
    OPEN = 'open'
    CLOSED = 'closed'

class PredictionOutcome(str, Enum):
    ALL = 'all'
    TOP3 = 'top3'
    WINNER = 'winner'