import pandas as pd
import numpy as np
from typing import Literal
from engine.typing import LoosePositionCategory, TightPositionCategory

def parse_quali_times(s: str) -> int:
    """Parse time strings into milliseconds."""
    if pd.isna(s) or s == "\\N":
        return 0

    first_split: list[str] = s.split(":")
    mins, secs, ms = first_split[0], *first_split[1].split(".")

    return int(mins) * 60_000 + int(secs) * 1000 + int(ms)


def parse_times(s: str) -> int:
    """Parse time strings into seconds."""
    if pd.isna(s) or s == "\\N":
        return np.nan

    hour, minute, second = s.split(":")
    return int(hour) * 3600 + int(minute) * 60 + int(second)

def get_position_category(
    value: str, type_: Literal["tight", "loose", "binary"] = "tight"
) -> str:
    if type_ == "binary":
        if value == "1":
            return "win"
        return "lose"

    if not value.isdigit() or value == "0":
        return "0"

    val = int(value)

    if type_ == "tight":
        if val == 1:
            return TightPositionCategory.FIRST.value
        if val == 2:
            return TightPositionCategory.SECOND.value
        if val == 3:
            return TightPositionCategory.THIRD.value
        if val <= 5:
            return TightPositionCategory.TOP_5.value
        if val <= 10:
            return TightPositionCategory.TOP_10.value
        return TightPositionCategory.TOP_20.value
    else:
        if val <= 3:
            return LoosePositionCategory.TOP_3.value
        if val <= 5:
            return LoosePositionCategory.TOP_5.value
        if val <= 10:
            return LoosePositionCategory.TOP_10.value
        return LoosePositionCategory.TOP_20.value