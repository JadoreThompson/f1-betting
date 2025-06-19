import math

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.sql.functions import sum as sql_sum, count, coalesce

from db_models import Bets, Markets, Transactions
from enums import BetStatus
from server.middleware import requires_wallet_address
from utils.db import requires_db_session
from .models import (
    Position,
    UserSummary,
    Transaction,
    Pagination,
    PositionResponse,
    TransactionResponse,
)

user_route = APIRouter(prefix="/user", tags=["user"])
PAGE_SIZE = 10


@user_route.get("/summary")
async def summary(
    wallet_address: str = Depends(requires_wallet_address),
    db_sess=Depends(requires_db_session),
) -> UserSummary:
    pnl_vol_markets_traded_q = select(
        coalesce(sql_sum(Bets.settle_amount), 0),
        coalesce(sql_sum(Bets.amount), 0),
        coalesce(count(Bets.wallet_address), 0),
    ).where(Bets.wallet_address == wallet_address)

    current_pos_value_q = select(coalesce(sql_sum(Bets.amount), 0)).where(
        (Bets.wallet_address == wallet_address)
        & (Bets.bet_status == BetStatus.OPEN.value)
    )

    # async with get_db_session() as db_sess:
    r = await db_sess.execute(pnl_vol_markets_traded_q)
    pnl, volume, markets_traded = r.first()

    r = await db_sess.execute(current_pos_value_q)
    current_pos_value = r.scalar_one()

    return UserSummary(
        total_pos_value=current_pos_value,
        pnl=pnl,
        volume=volume,
        markets_traded=markets_traded,
    )


@user_route.get("/positions")
async def positions(
    wallet_address: str = Depends(requires_wallet_address),
    page: int = 1,
    db_sess=Depends(requires_db_session),
) -> PositionResponse:
    if page < 1:
        page = 1

    market_info_subq = select(
        Markets.market_id,
        Markets.numerator,
        Markets.denominator,
        Markets.title,
        Markets.category,
    ).subquery()

    # First, get the total count for pagination
    count_query = select(count()).select_from(
        select(Bets.bet_id)
        .where(
            # (Bets.user_id == wallet_address.sub)
            (Bets.wallet_address == wallet_address)
            & (Bets.bet_status == BetStatus.OPEN.value)
        )
        .subquery()
    )

    # Get total count
    total_count_result = await db_sess.execute(count_query)
    total_items = total_count_result.scalar()

    # Calculate pagination info
    total_pages = math.ceil(total_items / PAGE_SIZE) if total_items > 0 else 1
    has_next = page < total_pages
    has_prev = page > 1

    # Get paginated data
    r = await db_sess.execute(
        select(
            Bets.amount,
            Bets.side,
            Bets.created_at,
            market_info_subq.c.numerator,
            market_info_subq.c.denominator,
            market_info_subq.c.title,
            market_info_subq.c.category,
        )
        .where(
            (Bets.wallet_address == wallet_address)
            & (Bets.bet_status == BetStatus.OPEN.value)
        )
        .offset(PAGE_SIZE * (page - 1))
        .limit(PAGE_SIZE)
        .outerjoin(
            market_info_subq,
            market_info_subq.c.market_id == Bets.market_id,
        )
    )

    d = r.all()

    positions = [
        Position(
            title=title,
            category=category,
            odds=f"{numerator}/{denominator}",
            side=side,
            amount=amount,
            created_at=created_at,
        )
        for amount, side, created_at, numerator, denominator, title, category in d
    ]

    pagination = Pagination(
        has_next=has_next,
        has_prev=has_prev,
        current_page=page,
        total_pages=total_pages,
        total_quantity=total_items,
    )

    return PositionResponse(data=positions, pagination=pagination)


@user_route.get("/activity")
async def activity(
    wallet_address: str = Depends(requires_wallet_address),
    page: int = 1,
    db_sess=Depends(requires_db_session),
) -> TransactionResponse:
    if page < 1:
        page = 1

    # First, get the total count for pagination
    count_query = select(count()).select_from(
        select(Transactions.transaction_id)
        .where(Transactions.wallet_address == wallet_address)
        .subquery()
    )

    # Get total count
    total_count_result = await db_sess.execute(count_query)
    total_items = total_count_result.scalar()

    # Calculate pagination info
    total_pages = math.ceil(total_items / PAGE_SIZE) if total_items > 0 else 1
    has_next = page < total_pages
    has_prev = page > 1

    # Get paginated data
    r = await db_sess.execute(
        select(
            Transactions.transaction_type,
            Transactions.amount,
            Transactions.address,
            Markets.title,
            Markets.category,
        )
        .where(Transactions.wallet_address == wallet_address)
        .offset(PAGE_SIZE * (page - 1))
        .limit(PAGE_SIZE)
        .join(Markets, (Markets.market_id == Transactions.market_id))
    )

    d = r.all()

    transactions_data = [
        Transaction(type=ttype, amount=amount, address=addr, title=title, category=cat)
        for ttype, amount, addr, title, cat in d
    ]

    pagination = Pagination(
        has_next=has_next,
        has_prev=has_prev,
        current_page=page,
        total_pages=total_pages,
        total_quantity=total_items,
    )

    return TransactionResponse(data=transactions_data, pagination=pagination)
