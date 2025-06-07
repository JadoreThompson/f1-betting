import asyncio
import os
from httpx import AsyncClient
import pytest

from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock

os.environ["PYTEST_RUNNING"] = "true"
from db_models import Base
from server.app import app


# Use a different database for testing
TEST_DB_URL = "sqlite+aiosqlite:///tests/test.db"

engine = create_async_engine(TEST_DB_URL)
smaker = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Creates an instance of the default event loop for the session."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_database():
    """
    Creates the test database tables before any tests run, and drops them after.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a database session for a single test function.
    Rolls back any changes after the test is complete to ensure isolation.
    """
    async with smaker() as session:
        try:
            await session.begin_nested()
            yield session
        except:
            await session.rollback()


@pytest.fixture(scope="function")
def mock_matching_engine_queue(mocker: MockerFixture) -> MagicMock:
    """Mocks the multiprocessing queue to the matching engine."""
    mock_queue = MagicMock()
    mocker.patch("server.config.MATCHING_ENGINE_QUEUE", new=mock_queue)
    return mock_queue


@pytest.fixture(scope="function")
def mock_web3(mocker: MockerFixture) -> AsyncMock:
    """Mocks the web3 provider and contract calls."""
    mock_provider = AsyncMock()
    mock_provider.eth.get_transaction_count.return_value = 1
    mock_provider.eth.gas_price = 1000
    mock_provider.eth.account.sign_transaction.return_value = MagicMock()
    mock_provider.eth.send_raw_transaction.return_value = AsyncMock(
        return_value=b"tx_hash"
    )

    mocker.patch("betting_engine.orderbook.PROVIDER", new=mock_provider)
    return mock_provider


async def override_get_db_session():
    """Override function that provides test database sessions."""
    async with smaker() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


@pytest.fixture(scope="function", autouse=True)
def override_db_dependency(monkeypatch):
    """Automatically override the database dependency for all tests."""
    monkeypatch.setattr("utils.db.get_db_session", override_get_db_session)
    # Also patch it in the app module in case it's imported there
    monkeypatch.setattr("server.app.get_db_session", override_get_db_session)


@pytest.fixture(scope="function")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Provides an async test client for the FastAPI application.
    """
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c
