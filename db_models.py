from datetime import datetime
from sqlalchemy.sql import text
from sqlalchemy import UUID, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from uuid import uuid4

from betting_engine.enums import BetStatus
from enums import MarketStatus


class Base(DeclarativeBase):
    pass


class Drivers(Base):
    __tablename__ = "drivers"

    driver_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    number: Mapped[str] = mapped_column(String, nullable=True)


class F1Data(Base):
    __tablename__ = "f1_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drivers.driver_id"), nullable=False, unique=True
    )
    last_1: Mapped[str] = mapped_column(String, nullable=True)
    last_2: Mapped[str] = mapped_column(String, nullable=True)
    last_3: Mapped[str] = mapped_column(String, nullable=True)
    last_4: Mapped[str] = mapped_column(String, nullable=True)
    last_5: Mapped[str] = mapped_column(String, nullable=True)


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
    title: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    market_status: Mapped[str] = mapped_column(
        Integer, nullable=False, default=MarketStatus.OPEN.value
    )
    numerator: Mapped[int] = mapped_column(Integer, nullable=False)
    denominator: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    closed_at: Mapped[DateTime] = mapped_column(
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
    bet_status: Mapped[BetStatus] = mapped_column(
        String(20), nullable=False, default=BetStatus.PENDING.value
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    closed_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=True, server_default=None
    )

    user = relationship("Users", back_populates="user_bets")
