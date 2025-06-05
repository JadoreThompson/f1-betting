from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import insert, select

from betting_engine import Topic
from db_models import Bets, Markets
from server.middleware import verify_jwt
from server.typing import JWTPayload
from utils.db import get_db_session
from utils.utils import dump_sqlalchemy_object
from .controller import push_to_engine
from .models import CreateBet

bet_route = APIRouter(prefix="/bet", tags=["bet"])


@bet_route.post("/create")
async def create_bet(body: CreateBet, jwt_payload: JWTPayload = Depends(verify_jwt)):
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


@bet_route.delete("/cancel")
async def cancel_bet(bet_id: str, jwt_payload: JWTPayload = Depends(verify_jwt)):
    async with get_db_session() as sess:
        res = await sess.execute(
            select(Bets).where(
                Bets.bet_id == bet_id, Bets.user_id == jwt_payload.sub
            )
        )

        bet: Bets | None = res.scalars().first()

        if not bet:
            raise HTTPException(status_code=404, detail="Bet not found")

    push_to_engine(
        Topic.CLOSE,
        market_id=bet.market_id,
        bet=dump_sqlalchemy_object(bet),
    )
