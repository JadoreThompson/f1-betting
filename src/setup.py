# setup.py
import os
import numpy as np
from datetime import datetime, time
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect

from config import SYNC_DB_ENGINE
from db_models import (
    Base,
    Circuits,
    Drivers,
    Qualifyings,
    Constructors,
    ConstructorStandings,
    Races,
    Results,
    Seasons,
    ConstructorResults,
    DriverStandings,
    LapTimes,
    SprintResults,
    Status,
)


folder = os.path.join(os.path.dirname(__file__), "model_development", "datasets")


csv_files = {
    "seasons": "seasons.csv",
    "circuits": "circuits.csv",
    "drivers": "drivers.csv",
    "qualifying": "qualifying.csv",
    "constructors": "constructors.csv",
    "constructor_standings": "constructor_standings.csv",
    "races": "races.csv",
    "results": "results.csv",
    "constructor_results": "constructor_results.csv",
    "driver_standings": "driver_standings.csv",
    "lap_times": "lap_times.csv",
    "sprint_results": "sprint_results.csv",
    "status": "status.csv",
}

dataframes = {
    name: pd.read_csv(os.path.join(folder, fname)) for name, fname in csv_files.items()
}


def parse_column(value: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    if not isinstance(value, str):
        return value
    parts = []
    prev = 0
    for i, char in enumerate(value):
        if char.isupper() and i != 0:
            parts.append(value[prev:i].lower())
            prev = i
    parts.append(value[prev:].lower())
    return "_".join(parts)


def clean_chunk(chunk: dict) -> dict:
    for col in chunk:
        if chunk[col] == "\\N" or (
            isinstance(chunk[col], float) and np.isnan(chunk[col])
        ):
            chunk[col] = None
    return chunk


def parse_date(value: str):
    return datetime.fromisoformat(value).date() if value and value != "\\N" else None


def parse_time(value: str) -> time | None:
    return datetime.fromisoformat(value).time() if value and value != "\\N" else None


def filter_columns(df: pd.DataFrame, model) -> pd.DataFrame:
    model_cols = {c.key for c in inspect(model).mapper.column_attrs}
    return df[[c for c in df.columns if c in model_cols]].copy()


for name, df in dataframes.items():
    df.columns = [parse_column(col) for col in df.columns]

Base.metadata.create_all(bind=SYNC_DB_ENGINE)
smaker = sessionmaker(bind=SYNC_DB_ENGINE, autoflush=False, autocommit=False)


def main():
    with smaker.begin() as session:
        try:
            # Seasons
            for chunk in filter_columns(dataframes["seasons"], Seasons).to_dict(
                orient="records"
            ):
                session.add(Seasons(**clean_chunk(chunk)))

            # Circuits
            for chunk in filter_columns(dataframes["circuits"], Circuits).to_dict(
                orient="records"
            ):
                session.add(Circuits(**clean_chunk(chunk)))

            # Drivers
            for chunk in filter_columns(dataframes["drivers"], Drivers).to_dict(
                orient="records"
            ):
                session.add(Drivers(**clean_chunk(chunk)))

            # Constructors
            for chunk in filter_columns(dataframes["constructors"], Constructors).to_dict(
                orient="records"
            ):
                session.add(Constructors(**clean_chunk(chunk)))

            # Races (parse dates and times)
            for chunk in filter_columns(dataframes["races"], Races).to_dict(
                orient="records"
            ):
                for date_col in [
                    "date",
                    "fp1_date",
                    "fp2_date",
                    "fp3_date",
                    "quali_date",
                    "sprint_date",
                ]:
                    if date_col in chunk:
                        chunk[date_col] = parse_date(chunk[date_col])
                session.add(Races(**clean_chunk(chunk)))

            # Constructor Standings
            for chunk in filter_columns(
                dataframes["constructor_standings"], ConstructorStandings
            ).to_dict(orient="records"):
                session.add(ConstructorStandings(**clean_chunk(chunk)))

            # Constructor Results
            for chunk in filter_columns(
                dataframes["constructor_results"], ConstructorResults
            ).to_dict(orient="records"):
                session.add(ConstructorResults(**clean_chunk(chunk)))

            # Driver Standings
            for chunk in filter_columns(
                dataframes["driver_standings"], DriverStandings
            ).to_dict(orient="records"):
                session.add(DriverStandings(**clean_chunk(chunk)))

            # Results
            for chunk in filter_columns(dataframes["results"], Results).to_dict(
                orient="records"
            ):
                session.add(Results(**clean_chunk(chunk)))

            # Sprint Results
            for chunk in filter_columns(
                dataframes["sprint_results"], SprintResults
            ).to_dict(orient="records"):
                session.add(SprintResults(**clean_chunk(chunk)))

            # Lap Times
            for chunk in filter_columns(dataframes["lap_times"], LapTimes).to_dict(
                orient="records"
            ):
                session.add(LapTimes(**clean_chunk(chunk)))

            # Qualifying
            for chunk in filter_columns(dataframes["qualifying"], Qualifyings).to_dict(
                orient="records"
            ):
                session.add(Qualifyings(**clean_chunk(chunk)))

            # Status
            for chunk in filter_columns(dataframes["status"], Status).to_dict(
                orient="records"
            ):
                session.add(Status(**clean_chunk(chunk)))

            session.commit()
        except:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
