from fastapi import APIRouter
from sqlalchemy import select

from db_models import Markets
from enums import MarketStatus, MarketCategory
from server.routes.markets.models import MarketResponse
from utils.db import get_db_session

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
