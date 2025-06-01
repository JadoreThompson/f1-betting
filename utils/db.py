import configparser
import os

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

from config import DB_ENGINE, DB_URL

smaker = sessionmaker(bind=DB_ENGINE, class_=AsyncSession, expire_on_commit=False)


def write_sqlalchemy_url() -> None:
    """Writes db url into the alamebic.ini file."""
    sqlalc_uri = DB_URL.replace("+asyncpg", "").replace("%", "%%")
    config = configparser.ConfigParser(interpolation=None)
    config.read("alembic.ini")

    config["alembic"].update({"sqlalchemy.url": sqlalc_uri})

    with open("alembic.ini", "w") as f:
        config.write(f)


def remove_sqlalchemy_url() -> None:
    """Removes the db url into the alamebic.ini file."""
    config = configparser.ConfigParser(interpolation=None)
    config.read("alembic.ini")
    config["alembic"].update({"sqlalchemy.url": ""})

    with open("alembic.ini", "w") as f:
        config.write(f)


def alembic_revision(message: str) -> None:
    write_sqlalchemy_url()
    os.system(f'alembic revision --autogenerate -m "{message}"')
    remove_sqlalchemy_url()


def alembic_ugrade_head() -> None:
    write_sqlalchemy_url()
    os.system("alembic upgrade head")
    remove_sqlalchemy_url()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with smaker.begin() as session:
        yield session
