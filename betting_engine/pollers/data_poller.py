import asyncio

from aiohttp import ClientSession
from asyncio import sleep
from dataclasses import asdict
from datetime import datetime
from sqlalchemy import insert, select, desc, case, text
from sqlalchemy.dialects.postgresql import insert as ps_insert
from sqlalchemy.sql.functions import sum as sql_sum, coalesce
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar

from config import POLLING_BASE_URL
from db_models import (
    Circuits,
    ConstructorStandings,
    Constructors,
    DriverStandings,
    Drivers,
    GrandPrixResults,
    QualiResults,
    SprintResults,
)
from utils.db import get_db_session
from utils.utils import get_datetime
from .base_poller import BasePoller
from .models import (
    AverageSpeed,
    Driver,
    FastestLap,
    FastestLapTime,
    GrandPrixResult,
    QualiResult,
    SprintResult,
    Constructor,
    Time,
)
from ..market_pipeline import MarketPipeline

T = TypeVar("T")


class DataPoller(BasePoller):
    """Formula 1 data poller that fetches and persists qualifying, sprint, and grand prix results.

    Inherits from BasePoller and implements async polling functionality to fetch F1 race data
    from the Formula 1 API and persist it to a database. Handles qualifying sessions,
    sprint races, and grand prix results.
    """

    def __init__(self, sleep_duration: int = 5) -> None:
        """Initialize the DataPoller with specified sleep duration.

        Args:
            sleep_duration (int, optional): Time to sleep between polling cycles in seconds.
                                          Defaults to 5.
        """
        super().__init__(sleep_duration)
        self._db_drivers: dict[str, Driver] = {}  # driver_ref => Driver
        self._market_pipeline = MarketPipeline()

    async def _assert_driver_and_constructor_id(self, driver: Driver) -> None:
        """Ensures driver and constructor have database IDs, persisting them if necessary.

        Adds the database representation of the constructor_id and driver_id
        if it isn't set on the driver object. This method modifies the driver
        object in-place by setting the missing IDs.

        Args:
            driver (Driver): Driver object that may need database IDs populated.
        """
        if driver.constructor.constructor_id is None:
            driver.constructor.constructor_id = await self._persist_constructor(
                driver.constructor
            )

        if driver.driver_id is None:
            driver.driver_id = await self._persist_driver(driver)

    def _dumper(self, obj: dict) -> dict[str, Any]:
        """Converts non-serializable objects to serializable format for debugging.

        Recursively processes dictionary objects to convert datetime objects to strings
        and handles nested dictionaries. Primarily used for JSON debugging output.

        Args:
            obj (dict): Dictionary that may contain non-serializable objects.

        Returns:
            dict[str, Any]: Dictionary with all objects converted to serializable format.
        """
        for k, v in obj.items():
            if isinstance(v, datetime):
                obj[k] = str(v)
            elif isinstance(v, dict):
                obj[k] = self._dumper(v)
        return obj

    def _get_configs(
        self,
    ) -> tuple[
        tuple[
            Callable[[list[dict[str, Any]]], Optional[tuple[int, datetime]]],
            str,
            str,
            Callable[[list[dict[str, Any]], int, int, int], tuple[T, ...]],
            Callable[[Iterable[T]], None],
        ],
        ...,
    ]:
        """Returns configuration tuples for different F1 race event types.

        Provides the necessary function references for processing qualifying,
        sprint, and grand prix events. Each configuration tuple contains:
        - Target function to find next event
        - API response key name
        - Fetch function for API data
        - Parse function for raw data
        - Persist function for database storage

        Returns:
            tuple: Configuration tuples for quali, sprint, and grand prix processing.
        """
        return (
            (
                self._get_target_quali,
                "QualifyingResults",
                "/{year}/{round_}/qualifying",
                self._parse_quali_results,
                self._persist_quali_result,
            ),
            (
                self._get_target_sprint,
                "SprintResults",
                "/{year}/{round_}/sprint",
                self._parse_sprint_results,
                self._persist_sprint_result,
            ),
            (
                self._get_target_gp,
                "Results",
                "/{year}/{round_}/results",
                self._parse_grand_prix_results,
                self._persist_grand_prix_result,
            ),
        )

    async def poll(
        self,
        production: bool = True,
        *,
        year: Optional[int] = None,
        round_: Optional[int] = None,
    ) -> None:
        """Main polling loop that fetches and persists F1 race weekend data.

        In production mode, this method continuously monitors the current F1 season,
        polling qualifying, sprint, and Grand Prix results as they become available.
        It processes and persists event data into the database, handles standings updates,
        and triggers a market prediction pipeline after qualifying. The loop waits
        until event times or a configured interval between iterations.

        In non-production mode (i.e., development/testing), the method instead simulates
        polling historical data from a specified `year` and `round_`, iterating forward
        without real-time delays.

        Args:
            production (bool, optional): Whether to run in production mode.
                In production mode, the poller fetches current-season live data,
                sleeping until events occur. If False, the poller operates deterministically
                and without delay. Defaults to True.

            year (Optional[int], keyword-only): The starting season year for non-production mode.
                Must be provided if `production` is False.

            round_ (Optional[int], keyword-only): The starting race round for non-production mode.
                Must be provided if `production` is False.

        Raises:
            ValueError: If `production` is False and either `year` or `round_` is not specified.
        """
        configs = self._get_configs()

        if not production:
            if year is None or round_ is None:
                raise ValueError(
                    "Year and round must be set if production is set to False."
                )
            year = year
            cur_round = round_

        async with ClientSession() as sess:
            while True:
                # Initialisation
                if production:
                    year = datetime.now().date().year

                schedule = await super()._fetch_schedule(sess, year)
                schedule = schedule["MRData"]["RaceTable"]["Races"]

                if production:
                    next_gp_info = self._get_target_gp(schedule)
                    if next_gp_info is None:
                        await asyncio.sleep(60 * 60 * 24)
                        continue
                    
                    print("Upcoming round", next_gp_info[0], "date", next_gp_info[1])
                    cur_round, _ = next_gp_info

                circuit_id, circuit_ref = await self._persist_circuit(
                    schedule, cur_round
                )

                # Fetching
                for (
                    target_func,
                    lookup_key,
                    fetch_path,
                    parse_func,
                    persist_func,
                ) in configs:
                    if production:
                        next_gp_info = target_func(schedule)
                        if next_gp_info is None or next_gp_info[0] != cur_round:
                            continue

                        _, time_of_event = next_gp_info
                        print("Sleeping until", time_of_event, "for", lookup_key)
                        await sleep(
                            time_of_event.timestamp() - get_datetime().timestamp()
                        )

                    fetched_data = await self._fetch_results(
                        sess, fetch_path.format(year=year, round_=cur_round)
                    )
                    if fetched_data is None:
                        continue  # For sprint, possibly missing data

                    success, fetched_data = await self._fetch_results(
                        sess, fetch_path.format(year=year, round_=cur_round)
                    )

                    if not success:
                        continue

                    leave = False
                    raw_results = fetched_data
                    for key_ in ["MRData", "RaceTable", "Races", 0, lookup_key]:
                        try:
                            raw_results = raw_results[key_]
                        except IndexError:
                            leave = True
                            break

                    if leave:
                        continue

                    parsed_data = parse_func(raw_results, year, cur_round, circuit_id)
                    await persist_func(parsed_data)

                    if lookup_key != "QualifyingResults":
                        await self._persist_driver_standings(year, cur_round)
                        await self._persist_constructor_standings(year, cur_round)
                    else:
                        await self._market_pipeline.run(
                            year,
                            cur_round,
                            circuit_id,
                            circuit_ref,
                            [q.driver.driver_id for q in parsed_data],
                            [q.position for q in parsed_data],
                        )

                print(f"Sleeping for {self._sleep_duration} seconds...")
                await sleep(self._sleep_duration)

                # Debugging
                if not production:
                    if cur_round == int(schedule[-1]["round"]):
                        year += 1
                        cur_round = 1
                    else:
                        cur_round += 1

                print("Fetching round:", cur_round, "year:", year)

    def _get_target_quali(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[Tuple[int, datetime]]:
        """Finds the next upcoming qualifying session in the race schedule.

        Searches through the provided schedule to find the next qualifying session
        that occurs after the current datetime.

        Args:
            schedule (list[dict[str, Any]]): List of race weekend data from F1 API.

        Returns:
            Optional[Tuple[int, datetime]]: Tuple of (round_number, qualifying_datetime)
                                          if found, None otherwise.
        """
        cur_datetime = get_datetime()

        for d in schedule:
            q_datetime = datetime.fromisoformat(
                f"{d['Qualifying']['date']}T{d['Qualifying']['time']}"
            )
            if q_datetime > cur_datetime:
                return (int(d["round"]), q_datetime)

    def _get_target_sprint(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[Tuple[int, datetime]]:
        """Finds the next upcoming sprint race in the race schedule.

        Searches through the provided schedule to find the next sprint race
        that occurs after the current datetime. Skips weekends without sprint races.

        Args:
            schedule (list[dict[str, Any]]): List of race weekend data from F1 API.

        Returns:
            Optional[Tuple[int, datetime]]: Tuple of (round_number, sprint_datetime)
                                          if found, None otherwise.
        """
        cur_datetime = get_datetime()

        for d in schedule:
            if "Sprint" not in d:
                continue

            s_datetime = datetime.fromisoformat(
                f"{d['Sprint']['date']}T{d['Sprint']['time']}"
            )
            if s_datetime > cur_datetime:
                return (int(d["round"]), s_datetime)

    def _get_target_gp(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[Tuple[int, datetime]]:
        """Finds the next upcoming grand prix race in the race schedule.

        Searches through the provided schedule to find the next grand prix race
        that occurs after the current datetime.

        Args:
            schedule (list[dict[str, Any]]): List of race weekend data from F1 API.

        Returns:
            Optional[Tuple[int, datetime]]: Tuple of (round_number, race_datetime)
                                          if found, None otherwise.
        """
        cur_datetime = get_datetime()

        for d in schedule:
            gp_datetime = datetime.fromisoformat(f"{d['date']}T{d['time']}")
            if gp_datetime > cur_datetime:
                return (int(d["round"]), gp_datetime)

    async def _fetch_results(
        self, session: ClientSession, path: str
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            rsp = await session.get(POLLING_BASE_URL + path)

            if rsp.status == 200:
                return (True, await rsp.json())

            attempts += 1
            await asyncio.sleep(2**attempts)

        return (False, None)

    def _parse_quali_results(
        self,
        data: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
    ) -> Tuple[QualiResult, ...]:
        """Parses raw qualifying results data into QualiResult objects.

        Converts JSON qualifying data from the F1 API into structured QualiResult
        objects with driver information and qualifying times.

        Args:
            data (list[dict[str, Any]]): Raw qualifying results from API.
            year (int): The racing season year.
            round_ (int): The race round number.
            circuit_id (int): Database ID of the circuit.

        Returns:
            Tuple[QualiResult, ...]: Tuple of parsed qualifying result objects.
        """
        return tuple(
            QualiResult(
                year=year,
                round_=round_,
                circuit_id=circuit_id,
                driver=self._parse_driver(d["Driver"], d["Constructor"]),
                q1=d.get("Q1"),
                q2=d.get("Q2"),
                q3=d.get("Q3"),
                position=int(d["position"]),
                position_text=d["position"],
            )
            for d in data
        )

    def _parse_sprint_results(
        self,
        data: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
    ) -> Tuple[SprintResult, ...]:
        """Parses raw sprint race results data into SprintResult objects.

        Converts JSON sprint race data from the F1 API into structured SprintResult
        objects with driver information and finishing positions.

        Args:
            data (list[dict[str, Any]]): Raw sprint results from API.
            year (int): The racing season year.
            round_ (int): The race round number.
            circuit_id (int): Database ID of the circuit.

        Returns:
            Tuple[SprintResult, ...]: Tuple of parsed sprint result objects.
        """
        return tuple(
            SprintResult(
                year=year,
                round_=round_,
                circuit_id=circuit_id,
                driver=self._parse_driver(d["Driver"], d["Constructor"]),
                grid=int(d["grid"]),
                position=int(d["position"]),
                position_text=d["positionText"],
                points=int(d["points"]),
            )
            for d in data
        )

    def _parse_grand_prix_results(
        self,
        data: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
    ) -> tuple[GrandPrixResult, ...]:
        """Parses raw grand prix results data into GrandPrixResult objects.

        Converts JSON grand prix data from the F1 API into structured GrandPrixResult
        objects with comprehensive race information including times, fastest laps,
        and finishing positions.

        Args:
            data (list[dict[str, Any]]): Raw grand prix results from API.
            year (int): The racing season year.
            round_ (int): The race round number.
            circuit_id (int): Database ID of the circuit.

        Returns:
            tuple[GrandPrixResult, ...]: Tuple of parsed grand prix result objects.
        """
        results = []
        for d in data:
            driver = self._parse_driver(d["Driver"], d["Constructor"])

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
                        units=units,
                        speed=float(speed) if speed is not None else speed,
                    ),
                )

            result = GrandPrixResult(
                year=year,
                round_=round_,
                circuit_id=circuit_id,
                number=int(d["number"]),
                position=int(d["position"]),
                position_text=d["positionText"],
                points=float(d["points"]),
                driver=driver,
                grid=int(d["grid"]),
                laps=int(d["laps"]),
                status=d["status"],
                time=time_obj,
                fastest_lap=fastest_lap_obj,
            )
            results.append(result)

        return tuple(results)

    def _parse_driver(
        self, driver_data: dict[str, Any], constructor_data: dict[str, Any]
    ) -> Driver:
        """Parses raw driver and constructor data into a Driver object.

        Converts JSON driver and constructor data from the F1 API into a structured
        Driver object with associated Constructor information.

        Args:
            driver_data (dict[str, Any]): Raw driver data from API.
            constructor_data (dict[str, Any]): Raw constructor data from API.

        Returns:
            Driver: Parsed driver object with constructor information.
        """
        driver_ref = driver_data["driverId"]
        if driver_ref in self._db_drivers:
            d = self._db_drivers[driver_ref]
            if d.constructor.name == constructor_data.get("constructor_ref", ""):
                return d

        d = Driver(
            driver_ref=driver_ref,
            permanent_number=driver_data.get("permanentNumber", ""),
            code=driver_data.get("code", ""),
            nationality=driver_data.get("nationality", ""),
            dob=datetime.strptime(driver_data["dateOfBirth"], "%Y-%m-%d"),
            constructor=Constructor(
                name=constructor_data.get("name", ""),
                nationality=constructor_data.get("nationality", ""),
                constructor_ref=constructor_data["constructorId"],
            ),
        )
        self._db_drivers[driver_ref] = d
        return d

    async def _persist_circuit(
        self, data: list[dict[str, Any]], round_: int
    ) -> tuple[int, str]:
        """Creates or retrieves a circuit record from the database.

        Creates a new circuit record if the circuit at the specified round is a new entrant,
        otherwise returns the existing circuit ID. Uses upsert logic with conflict handling.

        Args:
            data (list[dict[str, Any]]): Race schedule data containing circuit information.
            round_ (int): The race round number to extract circuit data for.

        Returns:
            int: Database circuit ID for the specified circuit.
        """
        circuit_data = data[round_ - 1]["Circuit"]

        async with get_db_session() as sess:
            r = await sess.execute(
                select(Circuits.circuit_id, Circuits.circuit_ref).where(
                    Circuits.circuit_ref == circuit_data["circuitId"]
                )
            )
            db_circuit_data = r.first()

            if not db_circuit_data:
                r = await sess.execute(
                    ps_insert(Circuits)
                    .values(
                        circuit_ref=circuit_data["circuitId"],
                        name=circuit_data["circuitName"],
                        location=f"{circuit_data["Location"]["locality"]} {circuit_data["Location"]["country"]}",
                        country=circuit_data["Location"]["country"],
                        lat=float(circuit_data["Location"]["lat"]),
                        lng=float(circuit_data["Location"]["long"]),
                    )
                    .on_conflict_do_nothing()
                    .returning(Circuits.circuit_id, Circuits.circuit_ref)
                )
                db_circuit_data = r.first()

                await sess.commit()

        return db_circuit_data

    async def _persist_constructor(self, constructor: Constructor) -> int:
        """Creates or retrieves a constructor record from the database.

        Inserts a new constructor record if it doesn't exist, otherwise returns
        the existing constructor ID.

        Args:
            constructor (Constructor): Constructor object to persist.

        Returns:
            int: Database constructor ID for the specified constructor.
        """
        async with get_db_session() as sess:
            r = await sess.execute(
                select(Constructors.constructor_id).where(
                    Constructors.name == constructor.name
                )
            )
            constructor_id = r.scalar_one_or_none()

            if constructor_id is None:
                r = await sess.execute(
                    insert(Constructors)
                    .values(
                        constructor_ref=constructor.constructor_ref,
                        name=constructor.name,
                        nationality=constructor.nationality,
                    )
                    .returning(Constructors.constructor_id)
                )
                constructor_id = r.scalar_one()

            await sess.commit()

        return constructor_id

    async def _persist_driver(self, driver: Driver) -> int:
        """Creates or retrieves a driver record from the database.

        Inserts a new driver record if it doesn't exist, otherwise returns
        the existing driver ID.

        Args:
            driver (Driver): Driver object to persist.

        Returns:
            int: Database driver ID for the specified driver.
        """
        async with get_db_session() as sess:
            r = await sess.execute(
                select(Drivers.driver_id).where(Drivers.driver_ref == driver.driver_ref)
            )
            driver_id = r.scalar_one_or_none()

            if driver_id is None:
                r = await sess.execute(
                    insert(Drivers)
                    .values(
                        driver_ref=driver.driver_ref,
                        permanent_number=driver.permanent_number,
                        code=driver.code,
                        constructor_id=driver.constructor.constructor_id,
                        dob=driver.dob,
                        nationality=driver.nationality,
                    )
                    .returning(Drivers.driver_id)
                )
                driver_id = r.scalar_one_or_none()

            await sess.commit()

        return driver_id

    async def _persist_quali_result(
        self,
        quali_results: Iterable[QualiResult],
    ) -> None:
        """Persists qualifying results to the database.

        Processes a collection of QualiResult objects, ensures all related drivers
        and constructors exist in the database, then bulk inserts the qualifying
        results using conflict handling to avoid duplicates.

        Args:
            quali_results (Iterable[QualiResult]): Collection of qualifying results to persist.
        """
        data = []

        for qr in quali_results:
            await self._assert_driver_and_constructor_id(qr.driver)

            d = asdict(qr)

            d["round"] = qr.round_
            d["driver_id"] = qr.driver.driver_id
            d["constructor_id"] = qr.driver.constructor.constructor_id

            d.pop("driver")
            d.pop("round_")

            data.append(d)

        async with get_db_session() as sess:
            await sess.execute(
                ps_insert(QualiResults).values(data).on_conflict_do_nothing()
            )
            await sess.commit()

    async def _persist_sprint_result(
        self, sprint_results: Iterable[SprintResult]
    ) -> None:
        """Persists sprint race results to the database.

        Processes a collection of SprintResult objects, ensures all related drivers
        and constructors exist in the database, then bulk inserts the sprint
        results using conflict handling to avoid duplicates.

        Args:
            sprint_results (Iterable[SprintResult]): Collection of sprint results to persist.
        """
        data = []

        for sr in sprint_results:
            await self._assert_driver_and_constructor_id(sr.driver)

            d = asdict(sr)

            d["round"] = sr.round_
            d["driver_id"] = sr.driver.driver_id
            d["constructor_id"] = sr.driver.constructor.constructor_id

            d.pop("driver")
            d.pop("round_")

            data.append(d)

        async with get_db_session() as sess:
            await sess.execute(
                ps_insert(SprintResults).values(data).on_conflict_do_nothing()
            )
            await sess.commit()

    async def _persist_grand_prix_result(
        self, grand_prix_results: Iterable[GrandPrixResult]
    ) -> None:
        """Persists grand prix race results to the database.

        Processes a collection of GrandPrixResult objects, ensures all related drivers
        and constructors exist in the database, flattens complex nested objects
        (time, fastest_lap) into database columns, then bulk inserts the results
        using conflict handling to avoid duplicates.

        Args:
            grand_prix_results (Iterable[GrandPrixResult]): Collection of grand prix results to persist.
        """
        data = []

        for gpr in grand_prix_results:
            await self._assert_driver_and_constructor_id(gpr.driver)

            d = asdict(gpr)

            d["round"] = gpr.round_
            d["driver_id"] = gpr.driver.driver_id
            d["constructor_id"] = gpr.driver.constructor.constructor_id

            if gpr.time is not None:
                d["time_millis"] = gpr.time.millis
                d["time_str"] = gpr.time.time
            else:
                d["time_millis"] = None
                d["time_str"] = None

            if gpr.fastest_lap is not None:
                d["fastest_lap_rank"] = gpr.fastest_lap.rank
                d["fastest_lap_number"] = gpr.fastest_lap.lap
                d["fastest_lap_time"] = gpr.fastest_lap.time.time
                d["fastest_lap_speed"] = gpr.fastest_lap.average_speed.speed
                d["fastest_lap_speed_unit"] = gpr.fastest_lap.average_speed.units
            else:
                d["fastest_lap_rank"] = None
                d["fastest_lap_number"] = None
                d["fastest_lap_time"] = None
                d["fastest_lap_speed"] = None
                d["fastest_lap_speed_unit"] = None

            d.pop("driver")
            d.pop("round_")
            d.pop("time")
            d.pop("fastest_lap")

            data.append(d)

        async with get_db_session() as sess:
            await sess.execute(
                ps_insert(GrandPrixResults).values(data).on_conflict_do_nothing()
            )
            await sess.commit()

    async def _persist_driver_standings(self, year: int, round_: int) -> None:
        # total GP wins per driver
        driver_wins_subq = (
            select(
                GrandPrixResults.driver_id,
                coalesce(
                    sql_sum(case((GrandPrixResults.position == 1, 1), else_=0)), 0
                ).label("total_wins"),
            )
            .where(GrandPrixResults.year == year)
            .group_by(GrandPrixResults.driver_id)
            .subquery()
        )

        # total grand prix points
        gp_points_subq = (
            select(
                GrandPrixResults.driver_id,
                coalesce(sql_sum(GrandPrixResults.points), 0).label("gp_point_sum"),
            )
            .where(GrandPrixResults.year == year)
            .group_by(GrandPrixResults.driver_id)
            .subquery()
        )

        # total sprint points
        sp_points_subq = (
            select(
                SprintResults.driver_id,
                coalesce(sql_sum(SprintResults.points), 0).label("sp_point_sum"),
            )
            .where(SprintResults.year == year)
            .group_by(SprintResults.driver_id)
            .subquery()
        )

        async with get_db_session() as sess:
            r = await sess.execute(
                text(
                    "SELECT 1 FROM driver_standings WHERE year = :year AND round = :round"
                ),
                {"year": year, "round": round_},
            )
            if r.first():
                return

            r = await sess.execute(
                select(
                    driver_wins_subq.c.driver_id,
                    Drivers.constructor_id,
                    (
                        coalesce(gp_points_subq.c.gp_point_sum, 0)
                        + coalesce(sp_points_subq.c.sp_point_sum, 0)
                    ).label("total_points"),
                    driver_wins_subq.c.total_wins,
                )
                .select_from(
                    driver_wins_subq.join(
                        Drivers, Drivers.driver_id == driver_wins_subq.c.driver_id
                    )
                    .outerjoin(
                        gp_points_subq,
                        gp_points_subq.c.driver_id == driver_wins_subq.c.driver_id,
                    )
                    .outerjoin(
                        sp_points_subq,
                        sp_points_subq.c.driver_id == driver_wins_subq.c.driver_id,
                    )
                )
                .order_by(desc("total_points"))
            )

            await sess.execute(
                insert(DriverStandings).values(
                    [
                        {
                            "year": year,
                            "round": round_,
                            "driver_id": driver_id,
                            "constructor_id": constructor_id,
                            "position": ind + 1,
                            "points": total_points,
                            "wins": total_wins,
                        }
                        for ind, (
                            driver_id,
                            constructor_id,
                            total_points,
                            total_wins,
                        ) in enumerate(r.all())
                    ]
                )
            )

            await sess.commit()

    async def _persist_constructor_standings(self, year: int, round_: int) -> None:
        async with get_db_session() as sess:
            r = await sess.execute(
                select(ConstructorStandings.constructor_id).where(
                    ConstructorStandings.year == year,
                    ConstructorStandings.round == round_,
                )
            )

            if r.first():
                return

            r = await sess.execute(
                select(
                    sql_sum(DriverStandings.points).label("constructor_points"),
                    DriverStandings.constructor_id,
                )
                .where(DriverStandings.year == year, DriverStandings.round == round_)
                .group_by(DriverStandings.constructor_id)
                .order_by(desc("constructor_points"))
            )

            await sess.execute(
                insert(ConstructorStandings).values(
                    [
                        {
                            "year": year,
                            "round": round_,
                            "constructor_id": constructor_id,
                            "points": points,
                            "position": ind + 1,
                        }
                        for ind, (points, constructor_id) in enumerate(r.all())
                    ]
                )
            )

            await sess.commit()
