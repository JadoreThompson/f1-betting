from datetime import datetime
from sqlalchemy.sql import text
from sqlalchemy import (
    UUID,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from uuid import uuid4

from betting_engine import BetStatus
from enums import MarketStatus


class Base(DeclarativeBase):
    pass


class Drivers(Base):
    __tablename__ = "drivers"

    driver_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    driver_ref: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    constructor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("constructors.constructor_id"), nullable=False
    )
    permanent_number: Mapped[str] = mapped_column(String, nullable=False)
    code: Mapped[str] = mapped_column(String, nullable=False)
    dob: Mapped[Date] = mapped_column(Date, nullable=False)
    nationality: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    constructor: Mapped["Constructors"] = relationship(
        "Constructors", back_populates="drivers"
    )
    quali_results: Mapped[list["QualiResults"]] = relationship(
        "QualiResults", back_populates="driver"
    )
    sprint_results: Mapped[list["SprintResults"]] = relationship(
        "SprintResults", back_populates="driver"
    )
    grand_prix_results: Mapped[list["GrandPrixResults"]] = relationship(
        "GrandPrixResults", back_populates="driver"
    )

    def __repr__(self):
        return f"Driver(id={self.driver_id}, name={self.given_name} {self.family_name})"


class Constructors(Base):
    __tablename__ = "constructors"

    constructor_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    constructor_ref: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    nationality: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    drivers: Mapped[list["Drivers"]] = relationship(
        "Drivers", back_populates="constructor"
    )
    quali_results: Mapped[list["QualiResults"]] = relationship(
        "QualiResults", back_populates="constructor"
    )
    sprint_results: Mapped[list["SprintResults"]] = relationship(
        "SprintResults", back_populates="constructor"
    )
    grand_prix_results: Mapped[list["GrandPrixResults"]] = relationship(
        "GrandPrixResults", back_populates="constructor"
    )

    def __repr__(self):
        return f"Constructor(id={self.constructor_id}, name={self.name})"


class Circuits(Base):
    __tablename__ = "circuits"

    circuit_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    circuit_ref: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)
    country: Mapped[str] = mapped_column(String, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    quali_results: Mapped[list["QualiResults"]] = relationship(
        "QualiResults", back_populates="circuit"
    )
    sprint_results: Mapped[list["SprintResults"]] = relationship(
        "SprintResults", back_populates="circuit"
    )
    grand_prix_results: Mapped[list["GrandPrixResults"]] = relationship(
        "GrandPrixResults", back_populates="circuit"
    )

    def __repr__(self):
        return (
            f"Circuit(id={self.circuit_id}, name={self.name}, location={self.location})"
        )


class QualiResults(Base):
    __tablename__ = "quali_results"

    quali_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    circuit_id: Mapped[int] = mapped_column(
        ForeignKey("circuits.circuit_id"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=False
    )
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=False
    )
    q1: Mapped[str | None] = mapped_column(String, nullable=True)
    q2: Mapped[str | None] = mapped_column(String, nullable=True)
    q3: Mapped[str | None] = mapped_column(String, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    position_text: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    driver: Mapped["Drivers"] = relationship("Drivers", back_populates="quali_results")
    constructor: Mapped["Constructors"] = relationship(
        "Constructors", back_populates="quali_results"
    )
    circuit: Mapped["Circuits"] = relationship(
        "Circuits", back_populates="quali_results"
    )

    __table_args__ = (
        UniqueConstraint("year", "round", "driver_id", name="uq_quali_result_driver"),
    )

    def __repr__(self):
        return f"QualiResult(year={self.year}, round={self.round}, driver_id={self.driver_id}, position={self.position})"


class SprintResults(Base):
    __tablename__ = "sprint_results"

    sprint_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    circuit_id: Mapped[int] = mapped_column(
        ForeignKey("circuits.circuit_id"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=False
    )
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=False
    )
    grid: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    position_text: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    driver: Mapped["Drivers"] = relationship("Drivers", back_populates="sprint_results")
    constructor: Mapped["Constructors"] = relationship(
        "Constructors", back_populates="sprint_results"
    )
    circuit: Mapped["Circuits"] = relationship(
        "Circuits", back_populates="sprint_results"
    )

    __table_args__ = (
        UniqueConstraint("year", "round", "driver_id", name="uq_sprint_result_driver"),
    )

    def __repr__(self):
        return f"SprintResult(year={self.year}, round={self.round}, driver_id={self.driver_id}, position={self.position})"


class GrandPrixResults(Base):
    __tablename__ = "grand_prix_results"

    grand_prix_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    circuit_id: Mapped[int] = mapped_column(
        ForeignKey("circuits.circuit_id"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.driver_id"), nullable=False
    )
    constructor_id: Mapped[int] = mapped_column(
        ForeignKey("constructors.constructor_id"), nullable=False
    )

    number: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    position_text: Mapped[str] = mapped_column(String, nullable=False)
    points: Mapped[float] = mapped_column(Float, nullable=False)
    grid: Mapped[int] = mapped_column(Integer, nullable=False)
    laps: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    time_millis: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_str: Mapped[str | None] = mapped_column(String, nullable=True)

    fastest_lap_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fastest_lap_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fastest_lap_time: Mapped[str | None] = mapped_column(String, nullable=True)
    fastest_lap_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    fastest_lap_speed_unit: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationships
    driver: Mapped["Drivers"] = relationship(
        "Drivers", back_populates="grand_prix_results"
    )
    constructor: Mapped["Constructors"] = relationship(
        "Constructors", back_populates="grand_prix_results"
    )
    circuit: Mapped["Circuits"] = relationship(
        "Circuits", back_populates="grand_prix_results"
    )

    __table_args__ = (
        UniqueConstraint("year", "round", "driver_id", name="uq_gp_result_driver"),
    )

    def __repr__(self):
        return (
            f"GrandPrixResult(year={self.year}, round={self.round}, driver_id={self.driver_id}, "
            f"position={self.position}, points={self.points})"
        )


# Web Application
class Users(Base):
    __tablename__ = "users"

    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password: Mapped[str] = mapped_column(String, nullable=False)

    user_bets = relationship(
        "Bets", back_populates="user", cascade="all, delete-orphan"
    )


class Markets(Base):
    """Represents a market for betting."""

    __tablename__ = "markets"

    market_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    title: Mapped[str] = mapped_column(
        String, nullable=False
    )  # TODO: change this to driver
    category: Mapped[str] = mapped_column(String, nullable=False)
    market_status: Mapped[str] = mapped_column(
        Integer, nullable=False, default=MarketStatus.OPEN.value
    )
    numerator: Mapped[int] = mapped_column(Integer, nullable=False)
    denominator: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=True, server_default=None
    )


class Bets(Base):
    """Represents a bet placed by a user on a market."""

    __tablename__ = "bets"

    bet_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False
    )
    market_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("markets.market_id"), nullable=False
    )
    side: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    wallet_address: Mapped[str] = mapped_column(String, nullable=False)
    bet_status: Mapped[str] = mapped_column(
        String, nullable=False, default=BetStatus.PENDING.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=True, server_default=None
    )
    settlement_txn: Mapped[str] = mapped_column(String, nullable=True)

    user = relationship("Users", back_populates="user_bets")
