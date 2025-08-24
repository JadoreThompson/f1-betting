from dataclasses import asdict
from datetime import datetime
from typing import Any, Iterable, Optional, TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert as ps_insert

from db_models import QualiResults
from services.market_generator import MarketGenerator
from utils.db import get_db_session
from utils.utils import get_datetime
from .session_handler import SessionHandler
from ..models import QualiResult

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from ..session_poller import SessionPoller


class QualifyingResultHandler(SessionHandler):
    """Handler for qualifying session results."""

    api_lookup_key = "QualifyingResults"
    fetch_path = "/{year}/{round_}/qualifying"

    def get_target_event_time(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[tuple[int, datetime]]:
        cur_datetime = get_datetime()
        for d in schedule:
            q_datetime = datetime.fromisoformat(
                f"{d['Qualifying']['date']}T{d['Qualifying']['time']}"
            )
            if q_datetime > cur_datetime:
                return int(d["round"]), q_datetime
        return None

    def _parse_results(
        self,
        poller_cls: "SessionPoller",
        data: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
    ) -> tuple[QualiResult, ...]:
        return tuple(
            QualiResult(
                year=year,
                round_=round_,
                circuit_id=circuit_id,
                driver=poller_cls._parse_driver(d["Driver"], d["Constructor"]),
                q1=d.get("Q1"),
                q2=d.get("Q2"),
                q3=d.get("Q3"),
                position=int(d["position"]),
                position_text=d["position"],
            )
            for d in data
        )

    async def _persist_results(
        self, poller_cls: "SessionPoller", quali_results: Iterable[QualiResult]
    ) -> None:
        data = []
        for qr in quali_results:
            await poller_cls._assert_driver_and_constructor_id(qr.driver)
            d = asdict(qr)
            d.update(
                {
                    "round": qr.round_,
                    "driver_id": qr.driver.driver_id,
                    "constructor_id": qr.driver.constructor.constructor_id,
                }
            )
            d.pop("driver"), d.pop("round_")
            data.append(d)

        async with get_db_session() as sess:
            await sess.execute(
                ps_insert(QualiResults).values(data).on_conflict_do_nothing()
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
        await MarketGenerator.generate(year, round_)
