from dataclasses import asdict
from datetime import datetime
from typing import Any, Iterable, Optional, TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert as ps_insert

from db_models import GrandPrixResults
from utils.db import get_db_session
from utils.utils import get_datetime
from .session_handler import SessionHandler
from ..models import (
    GrandPrixResult,
    Time,
    FastestLap,
    FastestLapTime,
    AverageSpeed,
)

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from ..session_poller import SessionPoller


class RaceResultHandler(SessionHandler):
    """Handler for Grand Prix race results."""

    api_lookup_key = "Results"
    fetch_path = "/{year}/{round_}/results"

    def get_target_event_time(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[tuple[int, datetime]]:
        cur_datetime = get_datetime()
        for d in schedule:
            gp_datetime = datetime.fromisoformat(f"{d['date']}T{d['time']}")
            if gp_datetime > cur_datetime:
                return int(d["round"]), gp_datetime
        return None

    def _parse_results(
        self,
        poller_cls: "SessionPoller",
        data: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
    ) -> tuple[GrandPrixResult, ...]:
        results = []
        for d in data:
            driver = poller_cls._parse_driver(d["Driver"], d["Constructor"])

            time_obj = None
            if "Time" in d:
                millis = int(d["Time"]["millis"]) if "millis" in d["Time"] else None
                time_obj = Time(millis=millis, time=d["Time"]["time"])

            fastest_lap_obj = None
            if "FastestLap" in d:
                fl: dict[str, Any] = d["FastestLap"]
                units = fl.get("AverageSpeed", {}).get("units")
                speed = fl.get("AverageSpeed", {}).get("speed")
                fastest_lap_obj = FastestLap(
                    rank=int(fl["rank"]),
                    lap=int(fl["lap"]),
                    time=FastestLapTime(time=fl["Time"]["time"]),
                    average_speed=AverageSpeed(
                        units=units, speed=float(speed) if speed is not None else speed
                    ),
                )

            result = GrandPrixResult(
                year=year,
                round_=round_,
                circuit_id=circuit_id,
                number=int(d["number"]),
                position=int(d["position"]),
                position_text=d["positionText"],
                points=int(d["points"]),
                driver=driver,
                grid=int(d["grid"]),
                laps=int(d["laps"]),
                status=d["status"],
                time=time_obj,
                fastest_lap=fastest_lap_obj,
            )
            results.append(result)
        return tuple(results)

    async def _persist_results(
        self, poller_cls: "SessionPoller", grand_prix_results: Iterable[GrandPrixResult]
    ) -> None:
        data = []
        for gpr in grand_prix_results:
            await poller_cls._assert_driver_and_constructor_id(gpr.driver)
            d = asdict(gpr)
            d.update(
                {
                    "round": gpr.round_,
                    "driver_id": gpr.driver.driver_id,
                    "constructor_id": gpr.driver.constructor.constructor_id,
                    "time_millis": gpr.time.millis if gpr.time else None,
                    "time_str": gpr.time.time if gpr.time else None,
                    "fastest_lap_rank": (
                        gpr.fastest_lap.rank if gpr.fastest_lap else None
                    ),
                    "fastest_lap_number": (
                        gpr.fastest_lap.lap if gpr.fastest_lap else None
                    ),
                    "fastest_lap_time": (
                        gpr.fastest_lap.time.time if gpr.fastest_lap else None
                    ),
                    "fastest_lap_speed": (
                        gpr.fastest_lap.average_speed.speed if gpr.fastest_lap else None
                    ),
                    "fastest_lap_speed_unit": (
                        gpr.fastest_lap.average_speed.units if gpr.fastest_lap else None
                    ),
                }
            )
            for key in ["driver", "round_", "time", "fastest_lap"]:
                d.pop(key)
            data.append(d)

        async with get_db_session() as sess:
            await sess.execute(
                ps_insert(GrandPrixResults).values(data).on_conflict_do_nothing()
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
