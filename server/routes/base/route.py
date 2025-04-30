import json

from typing import Tuple
from fastapi import APIRouter

from config import REDIS_CLIENT
from .controller import get_predictions

base_route = APIRouter()


@base_route.get("/odds")
async def odds() -> list[Tuple[int, int]]:
    schedule_: bytes | None = await REDIS_CLIENT.get("schedule")
    if schedule:
        schedule = json.loads(schedule_)

    # prev: bytes | None = await REDIS_CLIENT.get("odds")

    # if prev:
    #     return json.loads(prev)

    # prev: bytes | None = await REDIS_CLIENT.get("predictions")

    # if prev:
    #     predictions = json.loads(prev)
    # else:
    #     predictions = fetch_predictions()

    # odds = generate_odds(predictions)
    # await REDIS_CLIENT.set("predictions", json.dumps(predictions))
    # await REDIS_CLIENT.set("odds", json.dumps(odds))

    # return odds
