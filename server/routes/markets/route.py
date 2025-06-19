import os

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from json import load
from sqlalchemy import desc, select, case
from sqlalchemy.sql.functions import sum as sql_sum, coalesce

from config import SERVER_DATA_FOLDER
from db_models import Bets, Markets, Transactions
from enums import BetStatus, MarketStatus, MarketCategory, TransactionType
from utils.db import requires_db_session
from utils.utils import get_datetime
from .models import MarketSummary, NextRace, Overview, MarketResponse

markets_route = APIRouter(prefix="/markets", tags=["markets"])
next_race: NextRace | None = None


@markets_route.get("/")
async def get_markets(db_sess=Depends(requires_db_session)) -> MarketResponse:
    limit = 10
    cols = (
        Markets.market_id,
        Markets.title,
        Markets.category,
        Markets.numerator,
        Markets.denominator,
    )

    res = await db_sess.execute(
        select(*cols)
        .where(
            Markets.market_status != MarketStatus.SETTLED.value,
            Markets.category == MarketCategory.TOP3.value,
        )
        .limit(limit)
    )
    top3_markets = res.all() or []

    res = await db_sess.execute(
        select(*cols)
        .where(
            Markets.market_status != MarketStatus.SETTLED.value,
            Markets.category == MarketCategory.WINNER.value,
        )
        .limit(limit)
    )
    winner_markets = res.all() or []

    return MarketResponse(
        titles=[col.name for col in cols],
        top3=[list(m) for m in top3_markets],
        winners=[list(m) for m in winner_markets],
    )


@markets_route.get("/summary")
async def summary(
    db_sess=Depends(requires_db_session),
) -> MarketSummary:
    r = await db_sess.execute(
        select(
            coalesce(sql_sum(Bets.amount), 0),
            coalesce(
                sql_sum(
                    case(
                        (
                            Bets.bet_status.in_([BetStatus.PENDING, BetStatus.OPEN]),
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


@markets_route.get("/upcoming")
async def upcoming() -> NextRace | None:
    """Returns the upcoming race"""
    global next_race
    cur_datetime = get_datetime()

    if next_race:
        if cur_datetime >= next_race.datetime + timedelta(days=1):
            return next_race

    for r in load(open(os.path.join(SERVER_DATA_FOLDER, "schedule.json"), "r")):
        round_datetime = datetime.fromisoformat(f"{r["date"]}T{r["time"]}")
        if round_datetime > cur_datetime:
            return NextRace(
                datetime=round_datetime, name=r["raceName"], round=int(r["round"])
            )


@markets_route.get("/overview")
async def overview(
    db_sess=Depends(requires_db_session),
):
    # Get latest bet with market details
    latest_bet_query = (
        select(Transactions.amount, Markets.category, Markets.title)
        .join(Markets, Markets.market_id == Transactions.market_id)
        .where(Transactions.transaction_type == TransactionType.DEPOSIT.value)
        .order_by(desc(Transactions.created_at))
        .limit(1)
    )
    data = (await db_sess.execute(latest_bet_query)).first()
    if not data:
        raise HTTPException(204)

    lb_amount, lb_category, lb_title = data

    # Get most backed market
    most_backed_query = (
        select(
            sql_sum(Transactions.amount).label("total_amount"),
            Markets.category,
            Markets.title,
        )
        .join(Markets, Markets.market_id == Transactions.market_id)
        .where(Markets.market_status == MarketStatus.OPEN.value)
        .group_by(Transactions.market_id, Markets.category, Markets.title)
        .order_by(desc("total_amount"))
        .limit(1)
    )
    mb_amount, mb_category, mb_title = (
        await db_sess.execute(most_backed_query)
    ).first()

    return Overview(
        latest_bet_title=lb_title,
        latest_bet_category=lb_category,
        latest_bet_amount=lb_amount,
        most_backed_amount=mb_amount,
        most_backed_category=mb_category,
        most_backed_title=mb_title,
    )


@markets_route.get("/health")
async def health(
    market_id: int,
    db_sess=Depends(requires_db_session),
):
    r = await db_sess.execute(
        select(Markets.market_id).where(
            Markets.market_id == market_id,
            Markets.market_status == MarketStatus.OPEN.value,
        )
    )
    d = r.first()

    return {"health": bool(d)}
