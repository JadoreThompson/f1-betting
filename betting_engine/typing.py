from typing import Any, TypedDict
from enums import Side
from .enums import Topic

Payload = dict[str, str | int]

class SettlePayload(TypedDict):
    market_id: int
    winners: Side

class EnginePayload(TypedDict):
    topic: Topic
    market: dict[str, Any]
    bet: dict[str, Any]
    settle_data: SettlePayload
