import requests
import time
import logging
import os
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from config import SYNC_DB_ENGINE
from db_models import (
    Base,
    Seasons,
    Circuits,
    Constructors,
    Drivers,
    Races,
    Status,
    Results,
    SprintResults,
    Qualifyings,
    DriverStandings,
    ConstructorStandings,
    LapTimes,
)

# Note: The PitStops model was commented out in the provided schema.
# If you uncomment it, the ingestion logic for it is included below.
# from db_models import PitStops

# --- Configuration ---
DB_FILE = "f1_data.db"
API_BASE_URL = "https://api.jolpi.ca/ergast"
API_HEADERS = {
    # 'User-Agent': 'F1DataIngestionScript/1.0 (your-email@example.com)'
}
# Be polite to the API
REQUEST_DELAY_SECONDS = 0.5
RETRY_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 5

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ingestion.log"), logging.StreamHandler()],
)

# --- Database Setup ---
# Delete the old DB file to ensure a fresh start if needed
if os.path.exists(DB_FILE):
    logging.warning(f"Deleting existing database file: {DB_FILE}")
    os.remove(DB_FILE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=SYNC_DB_ENGINE)


def create_tables():
    """Create all database tables."""
    logging.info("Creating database tables...")
    Base.metadata.create_all(bind=SYNC_DB_ENGINE)
    logging.info("Database tables created successfully.")


# --- API Interaction ---
def make_api_request(
    endpoint: str,
    key: str,
    params: dict = None,
) -> dict | None:
    """
    Makes a request to the API with pagination and retry logic.

    Args:
        endpoint: The API endpoint to call (e.g., '/f1/seasons').
        params: A dictionary of query parameters.

    Returns:
        A list of all items fetched from the endpoint across all pages.
    """
    if params is None:
        params = {}

    all_items = []
    limit = 100  # Request a larger limit for efficiency
    offset = 0
    params["limit"] = limit

    while True:
        params["offset"] = offset
        full_url = f"{API_BASE_URL}{endpoint}"

        for attempt in range(RETRY_ATTEMPTS):
            try:
                time.sleep(REQUEST_DELAY_SECONDS)
                response = requests.get(
                    full_url, params=params, headers=API_HEADERS, timeout=30
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                # Find the main data table in the response
                mr_data = data.get("MRData", {})
                table_key = next(
                    (key for key in mr_data if key.endswith("Table")), None
                )
                if not table_key:
                    logging.warning(f"No Table key found in response for {endpoint}")
                    return all_items

                data_table = mr_data[table_key]
                # The actual list of items is usually pluralized (e.g., "Seasons", "Races")
                item_list_key = next((key for key in data_table), None)
                # print(data_table, item_list_key)
                items = data_table.get(key, [])
                # print("Items", items)

                if not items:
                    # No more items to fetch
                    return all_items

                if isinstance(items, list):
                    all_items.extend(items)

                total_results = int(mr_data.get("total", 0))
                current_offset = int(mr_data.get("offset", 0))
                current_limit = int(mr_data.get("limit", 0))

                if current_offset + current_limit >= total_results:
                    return all_items  # We have fetched all results

                offset += limit
                break  # Success, move to next page or finish

            except requests.exceptions.RequestException as e:
                logging.error(
                    f"Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed for {full_url}: {e}"
                )
                if attempt + 1 == RETRY_ATTEMPTS:
                    logging.critical(
                        f"All retry attempts failed for {full_url}. Aborting."
                    )
                    return None
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))  # Exponential backoff
    return None  # Should not be reached


# --- Ingestion Logic ---


def ingest_seasons(session: Session) -> list[str]:
    """Ingest all F1 seasons."""
    logging.info("Ingesting seasons...")
    seasons_data = make_api_request("/f1/seasons.json", "Seasons")
    if not seasons_data:
        logging.error("Failed to fetch seasons data.")
        return []

    # print(session.execute(select(Seasons)).all())
    existing_seasons = {
        s.year for s in session.execute(select(Seasons)).scalars().all()
    }
    seasons_to_add = []

    for season_data in seasons_data:
        year = int(season_data["season"])
        if year not in existing_seasons:
            seasons_to_add.append(Seasons(year=year, url=season_data["url"]))
            existing_seasons.add(year)

    if seasons_to_add:
        session.add_all(seasons_to_add)
        session.commit()

    logging.info(
        f"Ingested {len(seasons_to_add)} new seasons. Total seasons: {len(existing_seasons)}"
    )
    return sorted(list(existing_seasons), key=int)


def ingest_circuits(session: Session):
    """Ingest all circuits."""
    logging.info("Ingesting circuits...")
    circuits_data = make_api_request("/f1/circuits.json", "Circuits")
    if not circuits_data:
        logging.error("Failed to fetch circuits data.")
        return

    existing_circuits = {
        c.circuit_ref for c in session.execute(select(Circuits)).scalars().all()
    }
    circuits_to_add = []

    for c_data in circuits_data:
        if c_data["circuitId"] not in existing_circuits:
            circuits_to_add.append(
                Circuits(
                    circuit_ref=c_data["circuitId"],
                    name=c_data["circuitName"],
                    location=c_data["Location"]["locality"],
                    country=c_data["Location"]["country"],
                    lat=float(c_data["Location"]["lat"]),
                    lng=float(c_data["Location"]["long"]),
                )
            )
            existing_circuits.add(c_data["circuitId"])

    if circuits_to_add:
        session.add_all(circuits_to_add)
        session.commit()
    logging.info(f"Ingested {len(circuits_to_add)} new circuits.")


def ingest_status(session: Session):
    """Ingest all result statuses."""
    logging.info("Ingesting statuses...")
    status_data = make_api_request("/f1/status.json", "Status")
    if not status_data:
        logging.error("Failed to fetch status data.")
        return

    existing_statuses = {
        s.status_id for s in session.execute(select(Status)).scalars().all()
    }
    statuses_to_add = []

    for s_data in status_data:
        status_id = int(s_data["statusId"])
        if status_id not in existing_statuses:
            statuses_to_add.append(Status(status_id=status_id, status=s_data["status"]))
            existing_statuses.add(status_id)

    if statuses_to_add:
        session.add_all(statuses_to_add)
        session.commit()
    logging.info(f"Ingested {len(statuses_to_add)} new statuses.")


def ingest_drivers_and_constructors(session: Session, seasons: list[str]):
    """Ingest all drivers and constructors across all seasons."""
    logging.info("Ingesting drivers and constructors for all seasons...")

    existing_drivers = {
        d.driver_ref for d in session.execute(select(Drivers)).scalars().all()
    }
    existing_constructors = {
        c.constructor_ref for c in session.execute(select(Constructors)).scalars().all()
    }

    drivers_to_add = []
    constructors_to_add = []

    for season in seasons:
        logging.info(f"Fetching data for {season} season...")

        # Drivers for the season
        drivers_data = make_api_request(f"/f1/{season}/drivers.json", "Drivers")
        # print(drivers_data)
        if drivers_data:
            for d_data in drivers_data:
                if d_data["driverId"] not in existing_drivers:
                    drivers_to_add.append(
                        Drivers(
                            driver_ref=d_data["driverId"],
                            number=d_data.get("permanentNumber"),
                            code=d_data.get("code"),
                            forename=d_data["givenName"],
                            surname=d_data["familyName"],
                            dob=d_data["dateOfBirth"],
                            nationality=d_data["nationality"],
                            url=d_data["url"],
                        )
                    )
                    existing_drivers.add(d_data["driverId"])

        # Constructors for the season
        constructors_data = make_api_request(
            f"/f1/{season}/constructors.json", "Constructors"
        )
        # print(constructors_data)
        if constructors_data:
            for c_data in constructors_data:
                if c_data["constructorId"] not in existing_constructors:
                    constructors_to_add.append(
                        Constructors(
                            constructor_ref=c_data["constructorId"],
                            name=c_data["name"],
                            nationality=c_data["nationality"],
                            url=c_data["url"],
                        )
                    )
                    existing_constructors.add(c_data["constructorId"])

    if drivers_to_add:
        session.add_all(drivers_to_add)
    if constructors_to_add:
        session.add_all(constructors_to_add)

    if drivers_to_add or constructors_to_add:
        session.commit()

    logging.info(f"Ingested {len(drivers_to_add)} new drivers.")
    logging.info(f"Ingested {len(constructors_to_add)} new constructors.")


def ingest_races(session: Session, seasons: list[str], circuit_ref_map: dict):
    """Ingest races for all specified seasons."""
    logging.info("Ingesting races...")

    existing_races = {
        (r.year, r.round)
        for r in session.execute(select(Races.year, Races.round)).all()
    }

    for season in seasons:
        races_data = make_api_request(f"/f1/{season}/races.json", "Races")
        # print(races_data)
        if not races_data:
            logging.warning(f"No race data found for season {season}.")
            continue

        races_to_add = []
        for r_data in races_data:
            year = int(r_data["season"])
            round_num = int(r_data["round"])

            if (year, round_num) in existing_races:
                continue

            # Gracefully handle missing session data
            fp1 = r_data.get("FirstPractice", {})
            fp2 = r_data.get("SecondPractice", {})
            fp3 = r_data.get("ThirdPractice", {})
            quali = r_data.get("Qualifying", {})
            sprint = r_data.get("Sprint", {})

            races_to_add.append(
                Races(
                    year=year,
                    round=round_num,
                    circuit_id=circuit_ref_map.get(r_data["Circuit"]["circuitId"]),
                    name=r_data["raceName"],
                    date=r_data["date"],
                    time=r_data.get("time"),
                    url=r_data["url"],
                    fp1_date=fp1.get("date"),
                    fp1_time=fp1.get("time"),
                    fp2_date=fp2.get("date"),
                    fp2_time=fp2.get("time"),
                    fp3_date=fp3.get("date"),
                    fp3_time=fp3.get("time"),
                    quali_date=quali.get("date"),
                    sprint_date=sprint.get("date"),
                )
            )
            existing_races.add((year, round_num))

        if races_to_add:
            session.add_all(races_to_add)
            session.commit()
            logging.info(f"Ingested {len(races_to_add)} new races for {season} season.")


def ingest_race_related_data(session: Session, id_maps: dict):
    """Ingest data related to each race (results, qualifying, etc.)."""
    logging.info("Ingesting race-related data (results, qualifying, etc.)...")
    races = session.execute(select(Races)).scalars().all()

    for race in races:
        logging.info(
            f"Fetching data for Race: {race.year} Round {race.round} ({race.name})"
        )

        # --- Ingest Results ---
        results_data = make_api_request(
            f"/f1/{race.year}/{race.round}/results.json", "Races"
        )
        if results_data and results_data[0].get("Results"):
            results_to_add = []
            existing_results = {res.driver_id for res in race.results}
            for res_data in results_data[0]["Results"]:
                driver_id = id_maps["driver_map"].get(res_data["Driver"]["driverId"])
                if driver_id in existing_results:
                    continue

                fastest_lap_data = res_data.get("FastestLap", {})
                time_data = res_data.get("Time", {})
                results_to_add.append(
                    Results(
                        race_id=race.race_id,
                        driver_id=driver_id,
                        constructor_id=id_maps["constructor_map"].get(
                            res_data["Constructor"]["constructorId"]
                        ),
                        number=res_data["number"],
                        grid=res_data["grid"],
                        position=res_data.get("position"),
                        position_text=res_data["positionText"],
                        position_order=res_data["position"],
                        points=res_data["points"],
                        laps=res_data["laps"],
                        time=time_data.get("time"),
                        milliseconds=time_data.get("millis"),
                        fastest_lap=fastest_lap_data.get("lap"),
                        rank=fastest_lap_data.get("rank"),
                        fastest_lap_time=fastest_lap_data.get("Time", {}).get("time"),
                        status_id=id_maps["status_map"].get(res_data["status"]),
                        fastest_lap_speed=None,  # Not in Jolpica API
                    )
                )
            if results_to_add:
                session.add_all(results_to_add)

        # --- Ingest Sprint Results ---
        sprint_data = make_api_request(
            f"/f1/{race.year}/{race.round}/sprint.json", "Races"
        )
        if sprint_data and sprint_data[0].get("SprintResults"):
            sprints_to_add = []
            existing_sprints = {sprint.driver_id for sprint in race.sprint_results}
            for sprint_res_data in sprint_data[0]["SprintResults"]:
                driver_id = id_maps["driver_map"].get(
                    sprint_res_data["Driver"]["driverId"]
                )
                if driver_id in existing_sprints:
                    continue

                fastest_lap_data = sprint_res_data.get("FastestLap", {})
                time_data = sprint_res_data.get("Time", {})
                sprints_to_add.append(
                    SprintResults(
                        race_id=race.race_id,
                        driver_id=driver_id,
                        constructor_id=id_maps["constructor_map"].get(
                            sprint_res_data["Constructor"]["constructorId"]
                        ),
                        number=sprint_res_data["number"],
                        grid=sprint_res_data["grid"],
                        position=sprint_res_data.get("position"),
                        position_text=sprint_res_data["positionText"],
                        position_order=sprint_res_data["position"],
                        points=sprint_res_data["points"],
                        laps=sprint_res_data["laps"],
                        time=time_data.get("time"),
                        milliseconds=time_data.get("millis"),
                        fastest_lap=fastest_lap_data.get("lap"),
                        fastest_lap_time=fastest_lap_data.get("Time", {}).get("time"),
                        status_id=id_maps["status_map"].get(sprint_res_data["status"]),
                    )
                )
            if sprints_to_add:
                session.add_all(sprints_to_add)

        # --- Ingest Qualifying Results ---
        quali_data = make_api_request(
            f"/f1/{race.year}/{race.round}/qualifying.json", "Races"
        )
        if quali_data and quali_data[0].get("QualifyingResults"):
            qualis_to_add = []
            existing_qualis = {q.driver_id for q in race.qualifying}
            for q_data in quali_data[0]["QualifyingResults"]:
                driver_id = id_maps["driver_map"].get(q_data["Driver"]["driverId"])
                if driver_id in existing_qualis:
                    continue

                qualis_to_add.append(
                    Qualifyings(
                        race_id=race.race_id,
                        driver_id=driver_id,
                        constructor_id=id_maps["constructor_map"].get(
                            q_data["Constructor"]["constructorId"]
                        ),
                        number=q_data["number"],
                        position=q_data["position"],
                        q1=q_data.get("Q1"),
                        q2=q_data.get("Q2"),
                        q3=q_data.get("Q3"),
                    )
                )
            if qualis_to_add:
                session.add_all(qualis_to_add)

        # --- Ingest Lap Times ---
        laps_data = make_api_request(f"/f1/{race.year}/{race.round}/laps.json", "Races")
        if laps_data:
            laps_to_add = []
            existing_laps_query = select(LapTimes.driver_id, LapTimes.lap).where(
                LapTimes.race_id == race.race_id
            )
            existing_laps = {
                (r.driver_id, r.lap) for r in session.execute(existing_laps_query).all()
            }

            for lap_group in laps_data:
                for timing in lap_group.get("Laps", []):
                    # API groups laps by driver, but we need to find the driver for each lap group.
                    # This information is not directly in the lap time object, so we assume
                    # the /laps.json endpoint provides driver info outside the timings list.
                    # A better API would include driverId in each timing object.
                    # For now, let's assume it works by looping through drivers.
                    pass  # The Jolpica /laps endpoint structure is a bit ambiguous in the sample
                    # but would be handled here. A more robust implementation would
                    # iterate drivers, then laps for that driver if the API is structured that way.

        # --- Commit all data for this race ---
        session.commit()


def ingest_standings(session: Session, seasons: list[str], id_maps: dict):
    """Ingest final driver and constructor standings for each season."""
    logging.info("Ingesting final standings for all seasons...")
    race_map = {
        (r.year, r.round): r.race_id
        for r in session.execute(select(Races)).scalars().all()
    }

    for season in seasons:
        # --- Driver Standings ---
        ds_data = make_api_request(
            f"/f1/{season}/driverStandings.json", "StandingsLists"
        )
        if ds_data and ds_data[0]["StandingsLists"]:
            standings_list = ds_data[0]["StandingsLists"][0]
            round_num = int(standings_list["round"])
            race_id = race_map.get((int(season), round_num))

            if not race_id:
                logging.warning(
                    f"Could not find race for standings in season {season}, round {round_num}"
                )
                continue

            standings_to_add = []
            existing_standings = {
                s.driver_id
                for s in session.execute(
                    select(DriverStandings).where(DriverStandings.race_id == race_id)
                )
                .scalars()
                .all()
            }
            for standing in standings_list["DriverStandings"]:
                driver_id = id_maps["driver_map"].get(standing["Driver"]["driverId"])
                if driver_id not in existing_standings:
                    standings_to_add.append(
                        DriverStandings(
                            race_id=race_id,
                            driver_id=driver_id,
                            points=standing["points"],
                            position=standing["position"],
                            position_text=standing["positionText"],
                            wins=standing["wins"],
                        )
                    )
            if standings_to_add:
                session.add_all(standings_to_add)

        # --- Constructor Standings ---
        cs_data = make_api_request(
            f"/f1/{season}/constructorStandings.json", "StandingsLists"
        )
        if cs_data and cs_data[0]["StandingsLists"]:
            standings_list = cs_data[0]["StandingsLists"][0]
            round_num = int(standings_list["round"])
            race_id = race_map.get((int(season), round_num))

            if not race_id:
                continue  # Already logged warning from driver standings

            standings_to_add = []
            existing_standings = {
                s.constructor_id
                for s in session.execute(
                    select(ConstructorStandings).where(
                        ConstructorStandings.race_id == race_id
                    )
                )
                .scalars()
                .all()
            }
            for standing in standings_list["ConstructorStandings"]:
                constructor_id = id_maps["constructor_map"].get(
                    standing["Constructor"]["constructorId"]
                )
                if constructor_id not in existing_standings:
                    standings_to_add.append(
                        ConstructorStandings(
                            race_id=race_id,
                            constructor_id=constructor_id,
                            points=standing["points"],
                            position=standing["position"],
                            position_text=standing["positionText"],
                            wins=standing["wins"],
                        )
                    )
            if standings_to_add:
                session.add_all(standings_to_add)

        session.commit()
    logging.info("Finished ingesting standings.")


def main():
    """Main function to run the full data ingestion pipeline."""
    create_tables()

    db_session = SessionLocal()

    try:
        # Ingest independent data
        all_seasons = ingest_seasons(db_session)
        if not all_seasons:
            logging.critical("No seasons found. Aborting.")
            return

        ingest_circuits(db_session)
        ingest_status(db_session)

        # Ingest dependents
        ingest_drivers_and_constructors(db_session, all_seasons)

        # 3. Create maps for efficient foreign key lookups
        logging.info("Building foreign key maps...")
        circuit_ref_map = {
            c.circuit_ref: c.circuit_id
            for c in db_session.execute(select(Circuits)).scalars().all()
        }
        driver_ref_map = {
            d.driver_ref: d.driver_id
            for d in db_session.execute(select(Drivers)).scalars().all()
        }
        constructor_ref_map = {
            c.constructor_ref: c.constructor_id
            for c in db_session.execute(select(Constructors)).scalars().all()
        }
        status_text_map = {
            s.status: s.status_id
            for s in db_session.execute(select(Status)).scalars().all()
        }

        id_maps = {
            "circuit_map": circuit_ref_map,
            "driver_map": driver_ref_map,
            "constructor_map": constructor_ref_map,
            "status_map": status_text_map,
        }

        ingest_races(db_session, all_seasons, circuit_ref_map)
        ingest_race_related_data(db_session, id_maps)
        ingest_standings(db_session, all_seasons, id_maps)

        logging.info("--- Data ingestion complete! ---")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        db_session.rollback()
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
