from dataclasses import asdict
from datetime import datetime
from typing import Any, Iterable, Optional, TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert as ps_insert

from db_models import SprintResults
from utils.db import get_db_session
from utils.utils import get_datetime
from .session_handler import SessionHandler
from ..models import SprintResult

if TYPE_CHECKING:
    from ..session_poller import SessionPoller


class SprintResultHandler(SessionHandler):
    """Handler for sprint race results."""

    api_lookup_key = "SprintResults"
    fetch_path = "/{year}/{round_}/sprint"

    def get_target_event_time(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[tuple[int, datetime]]:
        cur_datetime = get_datetime()
        for d in schedule:
            if "Sprint" in d:
                s_datetime = datetime.fromisoformat(
                    f"{d['Sprint']['date']}T{d['Sprint']['time']}"
                )
                if s_datetime > cur_datetime:
                    return int(d["round"]), s_datetime
        return None

    def _parse_results(
        self,
        poller_cls: "SessionPoller",
        data: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
    ) -> tuple[SprintResult, ...]:
        return tuple(
            SprintResult(
                year=year,
                round_=round_,
                circuit_id=circuit_id,
                driver=poller_cls._parse_driver(d["Driver"], d["Constructor"]),
                grid=int(d["grid"]),
                position=int(d["position"]),
                position_text=d["positionText"],
                points=int(d["points"]),
            )
            for d in data
        )

    async def _persist_results(
        self, poller_cls: "SessionPoller", sprint_results: Iterable[SprintResult]
    ) -> None:
        data = []
        for sr in sprint_results:
            await poller_cls._assert_driver_and_constructor_id(sr.driver)
            d = asdict(sr)
            d.update(
                {
                    "round": sr.round_,
                    "driver_id": sr.driver.driver_id,
                    "constructor_id": sr.driver.constructor.constructor_id,
                }
            )
            d.pop("driver"), d.pop("round_")
            data.append(d)

        async with get_db_session() as sess:
            await sess.execute(
                ps_insert(SprintResults).values(data).on_conflict_do_nothing()
            )
            await sess.commit()

    async def process(
        self,
        poller_cls: "SessionPoller",
        raw_results: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
        circuit_ref: str,
    ) -> None:
        parsed_data = self._parse_results(
            poller_cls, raw_results, year, round_, circuit_id
        )
        await self._persist_results(poller_cls, parsed_data)

        # Post-persist hook: update standings
        await poller_cls._persist_driver_standings(year, round_)
        await poller_cls._persist_constructor_standings(year, round_)
