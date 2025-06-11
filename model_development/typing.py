from collections import namedtuple
from dataclasses import dataclass
from typing import Dict

Prediction = namedtuple("Prediction", ("prediction", "percentage"))
Report = Dict[str, Dict[str, float]]