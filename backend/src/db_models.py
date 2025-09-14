from datetime import datetime
from typing import Union
from sqlalchemy import DateTime, Integer, String, Float, ForeignKey, Date, Time, func
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column

from utils.utils import get_datetime


class Base(DeclarativeBase):
    pass


class Circuits(Base):
    __tablename__ = "circuits"

    circuit_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    circuit_ref: Mapped[str] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=True)
    location: Mapped[str] = mapped_column(String, nullable=True)
    country: Mapped[str] = mapped_column(String, nullable=True)
    lat: Mapped[float] = mapped_column(Float, nullable=True)
    lng: Mapped[float] = mapped_column(Float, nullable=True)

    races = relationship("Races", back_populates="circuit")


class Constructors(Base):
    __tablename__ = "constructors"

    constructor_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    constructor_ref: Mapped[str] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=True)
    nationality: Mapped[str] = mapped_column(String, nullable=True)
    url: Mapped[str] = mapped_column(String, nullable=True)

    results = relationship("Results", back_populates="constructor")
    sprint_results = relationship("SprintResults", back_populates="constructor")
    standings = relationship("ConstructorStandings", back_populates="constructor")
    results_history = relationship("ConstructorResults", back_populates="constructor")
    qualifying = relationship("Qualifyings", back_populates="constructor")
    predictions: Mapped[list["Predictions"]] = relationship(
        back_populates="constructor"
    )


class ConstructorResults(Base):
    __tablename__ = "constructors_results"

    constructor_results_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=True)
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=True
    )
    points: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=True)

    race = relationship("Races", back_populates="constructor_results")
    constructor = relationship("Constructors", back_populates="results_history")


class ConstructorStandings(Base):
    __tablename__ = "constructor_standings"

    constructor_standings_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=True)
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=True
    )
    points: Mapped[float] = mapped_column(Float, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=True)
    position_text: Mapped[str] = mapped_column(String, nullable=True)
    wins: Mapped[int] = mapped_column(Integer, nullable=True)

    race = relationship("Races", back_populates="constructor_standings")
    constructor = relationship("Constructors", back_populates="standings")


class Drivers(Base):
    __tablename__ = "drivers"

    driver_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    driver_ref: Mapped[str] = mapped_column(String, nullable=True)
    code: Mapped[str] = mapped_column(String, nullable=True)
    forename: Mapped[str] = mapped_column(String, nullable=True)
    surname: Mapped[str] = mapped_column(String, nullable=True)
    dob: Mapped[Date] = mapped_column(Date, nullable=True)
    nationality: Mapped[str] = mapped_column(String, nullable=True)
    number: Mapped[Union[str, None]] = mapped_column(String, nullable=True)
    url: Mapped[str] = mapped_column(String, nullable=True)

    results = relationship("Results", back_populates="driver")
    sprint_results = relationship("SprintResults", back_populates="driver")
    standings = relationship("DriverStandings", back_populates="driver")
    lap_times = relationship("LapTimes", back_populates="driver")
    qualifying = relationship("Qualifyings", back_populates="driver")
    predictions: Mapped[list["Predictions"]] = relationship(back_populates="driver")


class DriverStandings(Base):
    __tablename__ = "driver_standings"

    driver_standings_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=True)
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=True
    )
    points: Mapped[float] = mapped_column(Float, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=True)
    position_text: Mapped[str] = mapped_column(String, nullable=True)
    wins: Mapped[int] = mapped_column(Integer, nullable=True)

    race = relationship("Races", back_populates="driver_standings")
    driver = relationship("Drivers", back_populates="standings")


class LapTimes(Base):
    __tablename__ = "lap_times"

    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), primary_key=True)
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), primary_key=True
    )
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, nullable=True)
    time: Mapped[str] = mapped_column(String, nullable=True)
    milliseconds: Mapped[int] = mapped_column(Integer, nullable=True)

    race = relationship("Races", back_populates="lap_times")
    driver = relationship("Drivers", back_populates="lap_times")


# class PitStops(Base):
#     __tablename__ = "pit_stops"

#     race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), primary_key=True)
#     driver_id: Mapped[int] = mapped_column(
#         ForeignKey("drivers.driver_id"), primary_key=True
#     )
#     stop: Mapped[int] = mapped_column(Integer, primary_key=True)
#     lap: Mapped[int] = mapped_column(Integer)
#     time: Mapped[str] = mapped_column(String)
#     duration: Mapped[str] = mapped_column(String)
#     milliseconds: Mapped[int] = mapped_column(Integer)

#     race = relationship("Races", back_populates="pit_stops")
#     driver = relationship("Drivers", back_populates="pit_stops")


class Qualifyings(Base):
    __tablename__ = "qualifying"

    qualify_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=True)
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=True
    )
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=True
    )
    number: Mapped[int] = mapped_column(String, nullable=True)
    position: Mapped[int] = mapped_column(String, nullable=True)
    q1: Mapped[str] = mapped_column(String, nullable=True)
    q2: Mapped[str] = mapped_column(String, nullable=True)
    q3: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_datetime, server_default=func.now()
    )

    race = relationship("Races", back_populates="qualifying")
    driver = relationship("Drivers", back_populates="qualifying")
    constructor = relationship("Constructors", back_populates="qualifying")


class Races(Base):
    __tablename__ = "races"

    race_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(ForeignKey("seasons.year"), nullable=True)
    round: Mapped[int] = mapped_column(Integer, nullable=True)
    circuit_id: Mapped[int] = mapped_column(
        ForeignKey("circuits.circuit_id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String, nullable=True)
    date: Mapped[Date] = mapped_column(Date, nullable=True)
    time: Mapped[Time] = mapped_column(Time, nullable=True)
    url: Mapped[str] = mapped_column(String, nullable=True)
    fp1_date: Mapped[Date] = mapped_column(Date, nullable=True)
    fp1_time: Mapped[Time] = mapped_column(Time, nullable=True)
    fp2_date: Mapped[Date] = mapped_column(Date, nullable=True)
    fp2_time: Mapped[Time] = mapped_column(Time, nullable=True)
    fp3_date: Mapped[Date] = mapped_column(Date, nullable=True)
    fp3_time: Mapped[Time] = mapped_column(Time, nullable=True)
    quali_date: Mapped[Date] = mapped_column(Date, nullable=True)
    sprint_date: Mapped[Date] = mapped_column(Date, nullable=True)
    quali_time: Mapped[str] = mapped_column(String, nullable=True)
    sprint_time: Mapped[str] = mapped_column(String, nullable=True)

    circuit = relationship("Circuits", back_populates="races")
    season = relationship("Seasons", back_populates="races")
    results = relationship("Results", back_populates="race")
    sprint_results = relationship("SprintResults", back_populates="race")
    constructor_results = relationship("ConstructorResults", back_populates="race")
    constructor_standings = relationship("ConstructorStandings", back_populates="race")
    driver_standings = relationship("DriverStandings", back_populates="race")
    lap_times = relationship("LapTimes", back_populates="race")
    qualifying = relationship("Qualifyings", back_populates="race")
    predictions: Mapped[list["Predictions"]] = relationship(back_populates="race")


class Results(Base):
    __tablename__ = "results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=True)
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=True
    )
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=True
    )
    number: Mapped[int] = mapped_column(Integer, nullable=True)
    grid: Mapped[int] = mapped_column(Integer, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=True)
    position_text: Mapped[str] = mapped_column(String, nullable=True)
    position_order: Mapped[int] = mapped_column(Integer, nullable=True)
    points: Mapped[float] = mapped_column(Float, nullable=True)
    laps: Mapped[int] = mapped_column(Integer, nullable=True)
    time: Mapped[str] = mapped_column(String, nullable=True)
    milliseconds: Mapped[int] = mapped_column(Integer, nullable=True)
    fastest_lap: Mapped[int] = mapped_column(Integer, nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=True)
    fastest_lap_time: Mapped[str] = mapped_column(String, nullable=True)
    fastest_lap_speed: Mapped[str] = mapped_column(String, nullable=True)
    status_id: Mapped[int] = mapped_column(
        ForeignKey("status.status_id"), nullable=True
    )

    race = relationship("Races", back_populates="results")
    driver = relationship("Drivers", back_populates="results")
    constructor = relationship("Constructors", back_populates="results")
    status = relationship("Status", back_populates="results")


class Seasons(Base):
    __tablename__ = "seasons"

    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String, nullable=True)

    races = relationship("Races", back_populates="season")


class SprintResults(Base):
    __tablename__ = "sprint_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=True)
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=True
    )
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=True
    )
    number: Mapped[int] = mapped_column(Integer, nullable=True)
    grid: Mapped[int] = mapped_column(Integer, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=True)
    position_text: Mapped[str] = mapped_column(String, nullable=True)
    position_order: Mapped[int] = mapped_column(Integer, nullable=True)
    points: Mapped[float] = mapped_column(Float, nullable=True)
    laps: Mapped[int] = mapped_column(Integer, nullable=True)
    time: Mapped[str] = mapped_column(String, nullable=True)
    milliseconds: Mapped[int] = mapped_column(Integer, nullable=True)
    fastest_lap: Mapped[int] = mapped_column(Integer, nullable=True)
    fastest_lap_time: Mapped[str] = mapped_column(String, nullable=True)
    status_id: Mapped[int] = mapped_column(
        ForeignKey("status.status_id"), nullable=True
    )

    race = relationship("Races", back_populates="sprint_results")
    driver = relationship("Drivers", back_populates="sprint_results")
    constructor = relationship("Constructors", back_populates="sprint_results")
    status = relationship("Status", back_populates="sprint_results")


class Status(Base):
    __tablename__ = "status"

    status_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=True)

    results = relationship("Results", back_populates="status")
    sprint_results = relationship("SprintResults", back_populates="status")


class Predictions(Base):
    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), index=True)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.driver_id"), index=True)
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), index=True, nullable=True
    )

    predicted_position: Mapped[str] = mapped_column(String, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    outcome_class: Mapped[str] = mapped_column(String, nullable=False)  # ALL, TOP3, WINNER
    status: Mapped[str] = mapped_column(String, nullable=False)  # OPEN or CLOSED
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_datetime)

    race: Mapped["Races"] = relationship(back_populates="predictions")
    driver: Mapped["Drivers"] = relationship(back_populates="predictions")
    constructor: Mapped[Union["Constructors", None]] = relationship(
        back_populates="predictions"
    )
