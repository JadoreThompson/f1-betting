from typing import Any
from sqlalchemy.orm import DeclarativeBase


def dump_sqlalchemy_object(obj: DeclarativeBase) -> dict[str, Any]:
    """Convert a SQLAlchemy object to a dictionary."""
    return {k: v for k, v in vars(obj).items() if k != "_sa_instance_state"}
