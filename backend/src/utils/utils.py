from datetime import UTC, datetime
from sqlalchemy.orm import DeclarativeBase
from typing import Any
from uuid import UUID


def dump_sqlalchemy_object(obj: DeclarativeBase) -> dict[str, Any]:
    """Convert a SQLAlchemy object to a dictionary."""
    return {k: v for k, v in vars(obj).items() if k != "_sa_instance_state"}


def dump_obj(obj: dict) -> dict[str, Any]:
    """Converts datetime and UUID fields to str"""
    return {
        k: (str(v) if isinstance(v, (datetime, UUID)) else v) for k, v in obj.items()
    }


def get_datetime() -> datetime:
    return datetime.now(UTC)
