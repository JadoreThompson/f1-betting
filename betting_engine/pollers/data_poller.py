import json

from aiohttp import ClientSession
from asyncio import sleep
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Iterable, Optional, Tuple, TypeVar
from sqlalchemy import insert, select, desc
from sqlalchemy.dialects.postgresql import insert as ps_insert
from sqlalchemy.sql.functions import sum as sql_sum, coalesce

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

T = TypeVar("T")


# TODO:
# - Save drivers and info from quali in mem as it can be reused.
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
            Callable[[list[dict[str, Any]]], None],
            str,
            Callable[[ClientSession, int, int], Awaitable[dict[str, Any]]],
            Callable[[list[dict[str, Any]], int, int, int], tuple[T, ...]],
            Callable[[Iterable[T]], None],
        ]
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
                self._fetch_quali_results,
                self._parse_quali_results,
                self._persist_quali_result,
            ),
            (
                self._get_target_sprint,
                "SprintResults",
                self._fetch_sprint_results,
                self._parse_sprint_results,
                self._persist_sprint_result,
            ),
            (
                self._get_target_gp,
                "Results",
                self._fetch_grand_prix_results,
                self._parse_grand_prix_results,
                self._persist_grand_prix_result,
            ),
        )

    async def poll(self) -> None:
        """Main polling loop that fetches and persists F1 race data.

        Continuously polls for qualifying, sprint and grand prix results.
        For each race weekend, it:
        1. Fetches the current season schedule
        2. Identifies the next upcoming events
        3. Fetches results data for completed events
        4. Parses and persists the data to the database
        5. Sleeps for the configured duration before repeating

        The method will return if no upcoming grand prix is found.
        """
        configs = self._get_configs()
        cur_round = 0

        async with ClientSession() as sess:
            while True:
                print(1)
                # Initialisation
                schedule = await super()._fetch_schedule(sess)
                print(2)
                schedule = schedule["MRData"]["RaceTable"]["Races"]
                # next_gp_info = self._get_target_gp(schedule)

                cur_round += 1

                # if next_gp_info is None:
                #     return

                year = datetime.now().date().year - 1
                # cur_round, _ = next_gp_info
                circuit_id = await self._persist_circuit(schedule, cur_round)
                print(3)
                # Fetching
                for (
                    target_func,
                    lookup_key,
                    fetch_func,
                    parse_func,
                    persist_func,
                ) in configs:
                    # next_gp_info = target_func(schedule)
                    # if next_gp_info is None or next_gp_info[0] != cur_round:
                    #     continue

                    # Commented for testing
                    # Sleep until the event
                    # _, time_of_event = next_gp_info
                    # await sleep(round_time.timestamp() - datetime.now(UTC).timestamp())

                    fetched_data = await fetch_func(sess, cur_round, year)
                    # print("Fetched")
                    if fetched_data is None:
                        continue  # For sprint, possibly missing data

                    # Dump for debugging
                    json.dump(fetched_data, open(f"{lookup_key}.json", "w"), indent=4)

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

                    # Dump for debugging
                    # json.dump(
                    # [self._dumper(asdict(p)) for p in parsed_data],
                    # open(f"{lookup_key}-parsesd.json", "w"),
                    # indent=4,
                    # )

                    await persist_func(parsed_data)

                    if lookup_key != "QualifyingResults":
                        await self._persist_driver_standings(year, cur_round)
                        await self._persist_constructor_standings(year, cur_round)

                print(f"Sleeping for {self._sleep_duration} seconds...")
                await sleep(self._sleep_duration)

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
        cur_datetime = datetime.now(UTC)

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
        cur_datetime = datetime.now(UTC)

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
        cur_datetime = datetime.now(UTC)

        for d in schedule:
            gp_datetime = datetime.fromisoformat(f"{d['date']}T{d['time']}")
            if gp_datetime > cur_datetime:
                return (int(d["round"]), gp_datetime)

    async def _fetch_quali_results(
        self, session: ClientSession, target_round: int, year: int
    ) -> dict[str, Any]:
        """Fetches qualifying results from the Formula 1 API.

        Makes an HTTP request to retrieve qualifying session results for a specific
        race round and year.

        Args:
            session (ClientSession): Aiohttp client session for making requests.
            target_round (int): The race round number to fetch results for.
            year (int): The racing season year.

        Returns:
            dict[str, Any]: JSON response containing qualifying results data.

        Raises:
            Exception: If the API request returns a non-200 status code.
        """
        endpoint = f"/{year}/{target_round}/qualifying"
        rsp = await session.get(POLLING_BASE_URL + endpoint)
        if rsp.status != 200:
            raise Exception(f"{endpoint} threw status code: {rsp.status}")
        return await rsp.json()

    async def _fetch_sprint_results(
        self, session: ClientSession, target_round: int, year: int
    ) -> dict[str, Any]:
        """Fetches sprint race results from the Formula 1 API.

        Makes an HTTP request to retrieve sprint race results for a specific
        race round and year.

        Args:
            session (ClientSession): Aiohttp client session for making requests.
            target_round (int): The race round number to fetch results for.
            year (int): The racing season year.

        Returns:
            dict[str, Any]: JSON response containing sprint race results data.

        Raises:
            Exception: If the API request returns a non-200 status code.
        """
        endpoint = f"/{year}/{target_round}/sprint"
        rsp = await session.get(POLLING_BASE_URL + endpoint)
        if rsp.status != 200:
            raise Exception(f"{endpoint} threw status code: {rsp.status}")
        return await rsp.json()

    async def _fetch_grand_prix_results(
        self, session: ClientSession, target_round: int, year: int
    ) -> dict[str, Any]:
        """Fetches grand prix race results from the Formula 1 API.

        Makes an HTTP request to retrieve grand prix race results for a specific
        race round and year.

        Args:
            session (ClientSession): Aiohttp client session for making requests.
            target_round (int): The race round number to fetch results for.
            year (int): The racing season year.

        Returns:
            dict[str, Any]: JSON response containing grand prix results data.

        Raises:
            Exception: If the API request returns a non-200 status code.
        """
        endpoint = f"/{year}/{target_round}/results"
        rsp = await session.get(POLLING_BASE_URL + endpoint)
        if rsp.status != 200:
            raise Exception(f"{endpoint} threw status code: {rsp.status}")
        return await rsp.json()

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

            # time_obj = None
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
        return Driver(
            driver_ref=driver_data["driverId"],
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

    async def _persist_circuit(self, data: list[dict[str, Any]], round_: int) -> int:
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
                select(Circuits.circuit_id).where(
                    Circuits.circuit_ref == circuit_data["circuitId"]
                )
            )
            circuit_id = r.scalar_one_or_none()

            if not circuit_id:
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
                    .returning(Circuits.circuit_id)
                )
                circuit_id = r.scalar_one_or_none()

            await sess.commit()

        return circuit_id

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

            d["time_millis"] = gpr.time.millis
            d["time_str"] = gpr.time.time

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
        sp_subq = (
            select(
                sql_sum(SprintResults.points).label("sp_total_points"),
                SprintResults.driver_id,
            )
            .where(SprintResults.year == year)
            .group_by(SprintResults.driver_id)
        ).subquery()

        gp_subq = (
            select(
                sql_sum(GrandPrixResults.points).label("gp_total_points"),
                GrandPrixResults.driver_id,
            )
            .where(GrandPrixResults.year == year)
            .group_by(GrandPrixResults.driver_id)
        ).subquery()

        dc_subq = select(Drivers.driver_id, Drivers.constructor_id).subquery()

        async with get_db_session() as sess:
            r = await sess.execute(
                select(DriverStandings.driver_id).where(
                    DriverStandings.year == year, DriverStandings.round == round_
                )
            )

            if r.first():
                return

            r = await sess.execute(
                select(
                    gp_subq.c.driver_id,
                    dc_subq.c.constructor_id,
                    (
                        coalesce(gp_subq.c.gp_total_points, 0)
                        + coalesce(sp_subq.c.sp_total_points, 0)
                    ).label("total_points"),
                )
                .select_from(
                    gp_subq.outerjoin(
                        sp_subq, gp_subq.c.driver_id == sp_subq.c.driver_id
                    ).join(dc_subq, gp_subq.c.driver_id == dc_subq.c.driver_id)
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
                            "points": points,
                            "position": ind + 1,
                        }
                        for ind, (driver_id, constructor_id, points) in enumerate(
                            r.all()
                        )
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
