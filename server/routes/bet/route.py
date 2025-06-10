from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import insert, select

from betting_engine import Topic
from db_models import Bets, Markets, Transactions
from enums import TransactionType
from server.middleware import verify_jwt
from server.typing import JWTPayload
from utils.db import get_db_session
from utils.utils import dump_sqlalchemy_object
from .controller import push_to_engine
from .models import Bet
from .utils import lock

bet_route = APIRouter(prefix="/bet", tags=["bet"])


@bet_route.post("/create")
async def create_bet(body: Bet, jwt_payload: JWTPayload = Depends(verify_jwt)):
    async with lock:
        async with get_db_session() as sess:
            res = await sess.execute(
                select(Markets).where(Markets.market_id == body.market_id)
            )
            market = res.scalar_one()
            if not market:
                raise HTTPException(status_code=404, detail="Market not found")

            res = await sess.execute(
                insert(Bets)
                .values(
                    user_id=jwt_payload.sub,
                    market_id=market.market_id,
                    side=body.side,
                    amount=body.amount,
                    wallet_address=body.wallet_address,
                )
                .returning(Bets)
            )
            placed_bet: Bets = res.scalar_one()

            await sess.execute(
                insert(Transactions).values(
                    user_id=jwt_payload.sub,
                    bet_id=placed_bet.bet_id,
                    market_id=body.market_id,
                    transaction_type=TransactionType.DEPOSIT.value,
                    amount=body.amount,
                    address=body.txn_address,
                )
            )

            await sess.commit()

    push_to_engine(
        Topic.CREATE,
        market={
            "market_id": market.market_id,
            "numerator": market.numerator,
            "denominator": market.denominator,
        },
        bet=dump_sqlalchemy_object(placed_bet),
    )

    return JSONResponse(
        status_code=201,
        content={"message": "Bet placed successfully", "bet_id": placed_bet.bet_id},
    )
