import pytest

from unittest.mock import MagicMock
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from db_models import Users, Markets, Bets, Transactions
from enums import MarketCategory, Side
from betting_engine.enums import Topic


@pytest.fixture
async def test_user(db_session: AsyncSession) -> Users:
    """Creates a user in the test DB and returns the model instance."""
    user = Users(
        user_id=uuid4(),
        username="testuser",
        email="test@example.com",
        password="password123",
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.fixture
async def test_market(db_session: AsyncSession) -> Markets:
    """Creates a market in the test DB."""
    market = Markets(
        title="Max Verstappen",
        category=MarketCategory.WINNER,
        numerator=2,
        denominator=1,
    )
    db_session.add(market)
    await db_session.commit()
    return market


@pytest.fixture
def authenticated_client(client: AsyncClient, test_user: Users, mocker) -> AsyncClient:
    """Provides a client that is 'logged in' as test_user."""
    # Mock the verify_jwt dependency to bypass actual JWT logic for this unit test
    mock_jwt_payload = {"sub": str(test_user.user_id)}
    mocker.patch("server.routes.bet.route.verify_jwt", return_value=mock_jwt_payload)
    return client


# --- The Actual Test ---
@pytest.mark.asyncio
async def test_create_bet_success(
    authenticated_client: AsyncClient,
    db_session: AsyncSession,
    mock_matching_engine_queue: MagicMock,
    test_user: Users,
    test_market: Markets,
):
    """
    GIVEN a logged-in user and an existing market
    WHEN a POST request is made to /bet/create
    THEN a 201 response is returned
    AND a Bet and Transaction are created in the database
    AND a message is sent to the matching engine queue
    """
    # GIVEN
    bet_payload = {
        "market_id": test_market.market_id,
        "side": Side.BACK.value,
        "amount": 100.0,
        "wallet_address": "0x123...",
        "txn_address": "0xabc...",
    }

    # WHEN
    response = await authenticated_client.post("/bet/create", json=bet_payload)

    # THEN - Assert HTTP Response
    assert response.status_code == 201
    json_response = response.json()
    assert json_response["message"] == "Bet placed successfully"
    assert "bet_id" in json_response

    # THEN - Assert Database State
    bet_in_db = await db_session.get(Bets, json_response["bet_id"])
    assert bet_in_db is not None
    assert bet_in_db.user_id == test_user.user_id
    assert bet_in_db.amount == 100.0

    # THEN - Assert Queue Interaction
    mock_matching_engine_queue.put.assert_called_once()
    queue_payload = mock_matching_engine_queue.put.call_args[0][0]
    assert queue_payload["topic"] == Topic.CREATE
    assert queue_payload["bet"]["bet_id"] == bet_in_db.bet_id
    assert queue_payload["market"]["market_id"] == test_market.market_id
