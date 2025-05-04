from enum import Enum

class LoosePositionCategory(str, Enum):
    DNF = "0"
    TOP_3 = "1"
    TOP_5 = "2"
    TOP_10 = "3"
    TOP_20 = "4"

class TightPositionCategory(str, Enum):
    DNF = "0"
    FIRST = "1"
    SECOND = "2"
    THIRD = "3"
    TOP_5 = "4"
    TOP_10 = "5"
    TOP_20 = "6"
    
class Top3PositionCategory(str, Enum):
    TOP3 = "1"
    NOT_TOP_3 = "0"
