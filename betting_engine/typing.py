from enum import Enum
from typing import Any, TypedDict
from .enums import Topic

Payload = dict[str, str | int]

class EnginePayload(TypedDict):
    topic: Topic
    market: dict[str, Any]
    bet: dict[str, Any]
