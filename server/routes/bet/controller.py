from typing import Any, Optional
from betting_engine import Topic
from server.config import MATCHING_ENGINE_QUEUE


def push_to_engine(
    topic: Topic,
    *,
    market: Optional[dict[str, Any]] = None,
    bet: Optional[dict[str, Any]] = None
) -> None:
    payload = {
        "topic": topic,
    }

    if market:
        payload["market"] = market
    if bet:
        payload["bet"] = bet

    MATCHING_ENGINE_QUEUE.put(payload)
