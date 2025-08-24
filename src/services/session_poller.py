import asyncio

from aiohttp import ClientSession
from datetime import datetime
from sqlalchemy import insert, select, desc, case, text
from sqlalchemy.dialects.postgresql import insert as ps_insert
from sqlalchemy.sql.functions import sum as sql_sum, coalesce
from typing import Any, Optional

from config import API_BASE_URL
from db_models import (
    Circuits,
    ConstructorStandings,
    Constructors,
    DriverStandings,
    Drivers,
    Results,
    SprintResults,
    Races,
)
from services.exc import APIError
from utils.db import get_db_session
from utils.utils import get_datetime
from .models import Driver, Constructor
from .handlers import (
    SessionHandler,
    QualifyingResultHandler,
    SprintResultHandler,
    RaceResultHandler,
)


class SessionPoller:
    """
    A singleton F1 data poller that fetches and persists session results.

    This class acts as a singleton, using only class methods and state. It
    employs the Strategy pattern to handle different session types (qualifying,
    sprint, race) via a set of configurable handlers. This design follows the
    Open-Closed Principle, allowing for new session types to be added without
    modifying the poller's core logic. It is designed as a long-running,
    highly reliable service.
    """

    _sleep_duration: int = 5
    _db_drivers: dict[str, Driver] = {}
    _handlers: list[SessionHandler] = [
        QualifyingResultHandler(),
        SprintResultHandler(),
        RaceResultHandler(),
    ]

    @classmethod
    async def poll(
        cls,
        production: bool = True,
        *,
        year: Optional[int] = None,
        round_: Optional[int] = None,
    ) -> None:
        """
        Main polling loop that fetches and persists F1 race weekend data.

        In production mode, this method continuously monitors the current F1 season,
        delegating to the appropriate session handler as events become available.
        In non-production mode, it simulates polling for a specified `year` and `round_`.
        """
        if not production and (year is None or round_ is None):
            raise ValueError("Year and round must be set for non-production mode.")

        cur_year = year if not production else datetime.now().date().year
        cur_round = round_ if not production else 1

        async with ClientSession() as sess:
            while True:
                if production:
                    cur_year = datetime.now().date().year

                schedule = await cls._fetch_schedule(sess, cur_year)
                if not schedule:
                    await asyncio.sleep(
                        60 * 60
                    )  # Sleep for an hour if schedule is unavailable
                    continue

                if production:
                    # Find the very next GP to determine the current round
                    race_handler = RaceResultHandler()
                    next_gp_info = race_handler.get_target_event_time(schedule)
                    if next_gp_info is None:
                        print(
                            "Season appears to be over. Waiting for next season's schedule."
                        )
                        await asyncio.sleep(60 * 60 * 24)
                        continue
                    cur_round, _ = next_gp_info
                    print(f"Upcoming round: {cur_round}, Year: {cur_year}")

                circuit_id, circuit_ref = await cls._persist_circuit(
                    schedule, cur_round
                )

                for handler in cls._handlers:
                    if production:
                        target_info = handler.get_target_event_time(schedule)
                        if target_info is None or target_info[0] != cur_round:
                            continue

                        _, time_of_event = target_info
                        sleep_time = (
                            time_of_event.timestamp() - get_datetime().timestamp()
                        )
                        if sleep_time > 0:
                            print(
                                f"Sleeping until {time_of_event} for {handler.api_lookup_key}"
                            )
                            await asyncio.sleep(sleep_time)

                    fetch_path = handler.fetch_path.format(
                        year=cur_year, round_=cur_round
                    )
                    success, fetched_data = await cls._fetch_results(sess, fetch_path)

                    if not success or not fetched_data:
                        continue

                    # Traverse the nested JSON to get to the results list
                    raw_results = fetched_data
                    keys_to_traverse = [
                        "MRData",
                        "RaceTable",
                        "Races",
                        0,
                        handler.api_lookup_key,
                    ]
                    try:
                        for key in keys_to_traverse:
                            raw_results = raw_results[key]
                    except (KeyError, IndexError):
                        print(
                            f"Could not find results for {handler.api_lookup_key} at {fetch_path}"
                        )
                        continue

                    await handler.process(
                        cls, raw_results, cur_year, cur_round, circuit_id, circuit_ref
                    )

                print(
                    f"Polling cycle for round {cur_round} complete. Sleeping for {cls._sleep_duration} seconds."
                )
                await asyncio.sleep(cls._sleep_duration)

                if not production:
                    if cur_round == len(schedule):
                        cur_year += 1
                        cur_round = 1
                    else:
                        cur_round += 1
                    print(
                        f"Advancing to non-production round: {cur_round}, Year: {cur_year}"
                    )

    @classmethod
    async def _fetch_schedule(
        cls, session: ClientSession, year: int
    ) -> list[dict[str, Any]] | None:
        async with session.get(API_BASE_URL + f"/{year}") as rsp:
            if rsp.status != 200:
                return
            data = await rsp.json()
            return data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

    @classmethod
    async def _fetch_results(
        cls, session: ClientSession, path: str
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        attempts, max_attempts = 0, 10
        while attempts < max_attempts:
            async with session.get(API_BASE_URL + path) as rsp:
                if rsp.status == 200:
                    return True, await rsp.json()
            attempts += 1
            await asyncio.sleep(2**attempts)
        return False, None

    @classmethod
    async def _assert_driver_and_constructor_id(cls, driver: Driver) -> None:
        if driver.constructor.constructor_id is None:
            driver.constructor.constructor_id = await cls._persist_constructor(
                driver.constructor
            )
        if driver.driver_id is None:
            driver.driver_id = await cls._persist_driver(driver)

    @classmethod
    def _parse_driver(
        cls, driver_data: dict[str, Any], constructor_data: dict[str, Any]
    ) -> Driver:
        driver_ref = driver_data["driverId"]
        if driver_ref in cls._db_drivers:
            d = cls._db_drivers[driver_ref]
            if d.constructor.constructor_ref == constructor_data.get("constructorId"):
                return d

        d = Driver(
            driver_ref=driver_ref,
            permanent_number=driver_data.get("permanentNumber"),
            code=driver_data.get("code"),
            nationality=driver_data.get("nationality"),
            dob=datetime.strptime(driver_data["dateOfBirth"], "%Y-%m-%d"),
            constructor=Constructor(
                name=constructor_data.get("name"),
                nationality=constructor_data.get("nationality"),
                constructor_ref=constructor_data["constructorId"],
            ),
        )
        cls._db_drivers[driver_ref] = d
        return d

    @classmethod
    async def _persist_circuit(
        cls, data: list[dict[str, Any]], round_: int
    ) -> tuple[int, str]:
        circuit_data = data[round_ - 1]["Circuit"]
        circuit_ref = circuit_data["circuitId"]
        async with get_db_session() as sess:
            res = await sess.execute(
                select(Circuits.circuit_id, Circuits.circuit_ref).where(
                    Circuits.circuit_ref == circuit_ref
                )
            )
            if db_circuit := res.first():
                return db_circuit

            stmt = (
                ps_insert(Circuits)
                .values(
                    circuit_ref=circuit_ref,
                    name=circuit_data["circuitName"],
                    location=f"{circuit_data['Location']['locality']} {circuit_data['Location']['country']}",
                    country=circuit_data["Location"]["country"],
                    lat=float(circuit_data["Location"]["lat"]),
                    lng=float(circuit_data["Location"]["long"]),
                )
                .on_conflict_do_nothing()
                .returning(Circuits.circuit_id, Circuits.circuit_ref)
            )

            res = await sess.execute(stmt)
            db_circuit = res.first()
            await sess.commit()
        return db_circuit

    @classmethod
    async def _persist_constructor(cls, constructor: Constructor) -> int:
        async with get_db_session() as sess:
            res = await sess.execute(
                select(Constructors.constructor_id).where(
                    Constructors.constructor_ref == constructor.constructor_ref
                )
            )
            if constructor_id := res.scalar_one_or_none():
                return constructor_id

            res = await sess.execute(
                insert(Constructors)
                .values(
                    constructor_ref=constructor.constructor_ref,
                    name=constructor.name,
                    nationality=constructor.nationality,
                )
                .returning(Constructors.constructor_id)
            )
            constructor_id = res.scalar_one()
            await sess.commit()
            return constructor_id

    @classmethod
    async def _persist_driver(cls, driver: Driver) -> int:
        async with get_db_session() as sess:
            res = await sess.execute(
                select(Drivers.driver_id).where(Drivers.driver_ref == driver.driver_ref)
            )
            if driver_id := res.scalar_one_or_none():
                return driver_id

            res = await sess.execute(
                insert(Drivers)
                .values(
                    driver_ref=driver.driver_ref,
                    number=driver.permanent_number,  # Fixed: was permanent_number
                    code=driver.code,
                    nationality=driver.nationality,
                    dob=driver.dob,
                )
                .returning(Drivers.driver_id)
            )
            driver_id = res.scalar_one()
            await sess.commit()
        return driver_id

    @classmethod
    async def _persist_driver_standings(cls, year: int, round_: int) -> None:
        async with get_db_session() as sess:
            # Get the race_id for this year and round to properly link standings
            race_res = await sess.execute(
                select(Races.race_id).where(Races.year == year, Races.round == round_)
            )
            race_id = race_res.scalar_one_or_none()
            if race_id is None:
                print(f"No race found for year {year}, round {round_}")
                return

            # Check if standings for this race already exist
            existing_res = await sess.execute(
                select(DriverStandings.driver_standings_id)
                .where(DriverStandings.race_id == race_id)
                .limit(1)
            )
            if existing_res.scalar_one_or_none():
                return

            # Subquery for total GP wins in the season up to this round
            wins_subq = (
                select(
                    Results.driver_id,
                    coalesce(
                        sql_sum(case((Results.position == 1, 1), else_=0)), 0
                    ).label("total_wins"),
                )
                .select_from(Results.join(Races, Results.race_id == Races.race_id))
                .where(Races.year == year, Races.round <= round_)
                .group_by(Results.driver_id)
                .subquery()
            )

            # Subquery for total GP points up to this round
            gp_pts_subq = (
                select(
                    Results.driver_id,
                    coalesce(sql_sum(Results.points), 0).label("gp_pts"),
                )
                .select_from(Results.join(Races, Results.race_id == Races.race_id))
                .where(Races.year == year, Races.round <= round_)
                .group_by(Results.driver_id)
                .subquery()
            )

            # Subquery for total sprint points up to this round
            sp_pts_subq = (
                select(
                    SprintResults.driver_id,
                    coalesce(sql_sum(SprintResults.points), 0).label("sp_pts"),
                )
                .select_from(
                    SprintResults.join(Races, SprintResults.race_id == Races.race_id)
                )
                .where(Races.year == year, Races.round <= round_)
                .group_by(SprintResults.driver_id)
                .subquery()
            )

            standings_query = (
                select(
                    Drivers.driver_id,
                    (
                        coalesce(gp_pts_subq.c.gp_pts, 0)
                        + coalesce(sp_pts_subq.c.sp_pts, 0)
                    ).label("total_points"),
                    coalesce(wins_subq.c.total_wins, 0).label("total_wins"),
                )
                .select_from(Drivers)
                .outerjoin(wins_subq, Drivers.driver_id == wins_subq.c.driver_id)
                .outerjoin(gp_pts_subq, Drivers.driver_id == gp_pts_subq.c.driver_id)
                .outerjoin(sp_pts_subq, Drivers.driver_id == sp_pts_subq.c.driver_id)
                .where(
                    Drivers.driver_id.in_(
                        select(Results.driver_id)
                        .select_from(
                            Results.join(Races, Results.race_id == Races.race_id)
                        )
                        .where(Races.year == year, Races.round <= round_)
                        .distinct()
                    )
                )
                .order_by(desc("total_points"), desc("total_wins"))
            )

            res = await sess.execute(standings_query)
            standings_data = [
                {
                    "race_id": race_id,
                    "driver_id": d.driver_id,
                    "points": float(d.total_points),
                    "position": i + 1,
                    "position_text": str(i + 1),
                    "wins": d.total_wins,
                }
                for i, d in enumerate(res.all())
            ]

            if standings_data:
                await sess.execute(insert(DriverStandings).values(standings_data))
                await sess.commit()

    @classmethod
    async def _persist_constructor_standings(cls, year: int, round_: int) -> None:
        async with get_db_session() as sess:
            # Get the race_id for this year and round
            race_res = await sess.execute(
                select(Races.race_id).where(Races.year == year, Races.round == round_)
            )
            race_id = race_res.scalar_one_or_none()
            if race_id is None:
                print(f"No race found for year {year}, round {round_}")
                return

            # Check if constructor standings for this race already exist
            existing_res = await sess.execute(
                select(ConstructorStandings.constructor_standings_id)
                .where(ConstructorStandings.race_id == race_id)
                .limit(1)
            )
            if existing_res.scalar_one_or_none():
                return

            # Calculate constructor points by summing their drivers' points up to this round
            # Subquery for GP points
            gp_points_subq = (
                select(
                    Results.constructor_id,
                    coalesce(sql_sum(Results.points), 0).label("gp_points"),
                )
                .select_from(Results.join(Races, Results.race_id == Races.race_id))
                .where(Races.year == year, Races.round <= round_)
                .group_by(Results.constructor_id)
                .subquery()
            )

            # Subquery for sprint points
            sprint_points_subq = (
                select(
                    SprintResults.constructor_id,
                    coalesce(sql_sum(SprintResults.points), 0).label("sprint_points"),
                )
                .select_from(
                    SprintResults.join(Races, SprintResults.race_id == Races.race_id)
                )
                .where(Races.year == year, Races.round <= round_)
                .group_by(SprintResults.constructor_id)
                .subquery()
            )

            # Subquery for constructor wins
            wins_subq = (
                select(
                    Results.constructor_id,
                    coalesce(
                        sql_sum(case((Results.position == 1, 1), else_=0)), 0
                    ).label("total_wins"),
                )
                .select_from(Results.join(Races, Results.race_id == Races.race_id))
                .where(Races.year == year, Races.round <= round_)
                .group_by(Results.constructor_id)
                .subquery()
            )

            standings_query = (
                select(
                    Constructors.constructor_id,
                    (
                        coalesce(gp_points_subq.c.gp_points, 0)
                        + coalesce(sprint_points_subq.c.sprint_points, 0)
                    ).label("total_points"),
                    coalesce(wins_subq.c.total_wins, 0).label("total_wins"),
                )
                .select_from(Constructors)
                .outerjoin(
                    gp_points_subq,
                    Constructors.constructor_id == gp_points_subq.c.constructor_id,
                )
                .outerjoin(
                    sprint_points_subq,
                    Constructors.constructor_id == sprint_points_subq.c.constructor_id,
                )
                .outerjoin(
                    wins_subq, Constructors.constructor_id == wins_subq.c.constructor_id
                )
                .where(
                    Constructors.constructor_id.in_(
                        select(Results.constructor_id)
                        .select_from(
                            Results.join(Races, Results.race_id == Races.race_id)
                        )
                        .where(Races.year == year, Races.round <= round_)
                        .distinct()
                    )
                )
                .order_by(desc("total_points"), desc("total_wins"))
            )

            res = await sess.execute(standings_query)
            standings_data = [
                {
                    "race_id": race_id,
                    "constructor_id": d.constructor_id,
                    "points": float(d.total_points),
                    "position": i + 1,
                    "position_text": str(i + 1),
                    "wins": d.total_wins,
                }
                for i, d in enumerate(res.all())
            ]

            if standings_data:
                await sess.execute(insert(ConstructorStandings).values(standings_data))
                await sess.commit()
