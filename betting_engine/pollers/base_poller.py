from aiohttp import ClientSession
from typing import Any, override
from config import POLLING_BASE_URL
from .exc import APIError


class BasePoller:
    def __init__(self, sleep_duration: int = 5) -> None:
        self._sleep_duration = sleep_duration

    async def _fetch_schedule(
        self, session: ClientSession, year: int
    ) -> dict[str, Any]:
        async with session.get(POLLING_BASE_URL + f"/{year}") as rsp:
            if rsp.status != 200:
                raise APIError(
                    f"Error fetching season schedule. status code: {rsp.status}"
                )
            return await rsp.json()

    @override
    async def poll(self) -> Any: ...
