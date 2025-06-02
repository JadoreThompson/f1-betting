from aiohttp import ClientSession
from datetime import UTC, datetime
from typing import Any, Optional, override
from config import POLLING_BASE_URL


class TargetNotFound(Exception):
    def __init__(self) -> None:
        super().__init__("No target round was found. Season possibly over.")


class BasePoller:
    def __init__(self, sleep_duration: int = 5) -> None:
        self._sleep_duration = sleep_duration

    async def _fetch_schedule(self, session: ClientSession) -> dict[str, Any]:
        async with session.get(POLLING_BASE_URL + f"/current") as rsp:
            if rsp.status != 200:
                raise Exception("Error fetching season schedule.")
            return await rsp.json()

    @override
    async def poll(self) -> Any: ...
