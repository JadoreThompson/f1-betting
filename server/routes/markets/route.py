from fastapi import APIRouter
from sqlalchemy import select, func, case
from sqlalchemy.sql.functions import sum as sql_sum, coalesce

from db_models import Bets, Markets
from enums import BetStatus, MarketStatus, MarketCategory
from server.routes.markets.models import MarketResponse
from utils.db import get_db_session
from .models import MarketSummary

markets_route = APIRouter(prefix="/markets", tags=["markets"])


@markets_route.get("/")
async def get_markets() -> MarketResponse:
    limit = 10
    cols = (
        Markets.market_id,
        Markets.title,
        Markets.category,
        Markets.numerator,
        Markets.denominator,
    )

    async with get_db_session() as sess:
        res = await sess.execute(
            select(*cols)
            .where(
                Markets.market_status != MarketStatus.SETTLED.value,
                Markets.category == MarketCategory.TOP3.value,
            )
            .limit(limit)
        )
        top3_markets = res.all()

        res = await sess.execute(
            select(*cols)
            .where(
                Markets.market_status != MarketStatus.SETTLED.value,
                Markets.category == MarketCategory.WINNER.value,
            )
            .limit(limit)
        )
        winner_markets = res.all()

    return MarketResponse(
        titles=[col.name for col in cols],
        top3=[list(m) for m in top3_markets],
        winners=[list(m) for m in winner_markets],
    )


@markets_route.get("/summary")
async def summary() -> MarketSummary:
    async with get_db_session() as sess:
        r = await sess.execute(
            select(
                coalesce(sql_sum(Bets.amount), 0),
                coalesce(
                    sql_sum(
                        case(
                            (
                                Bets.bet_status.in_(
                                    [BetStatus.PENDING, BetStatus.OPEN]
                                ),
                                1,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
            )
        )
        d = r.first()
    return MarketSummary(total_volume=d[0], active_bets=d[1])
