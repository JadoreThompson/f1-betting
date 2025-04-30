from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
