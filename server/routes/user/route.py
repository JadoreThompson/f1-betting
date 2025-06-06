from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.sql.functions import sum as sql_sum, count, coalesce
import math

from db_models import Bets, Markets, Transactions, Users
from enums import BetStatus
from server.middleware import verify_jwt
from server.typing import JWTPayload
from utils.db import get_db_session
from .models import Position, UserSummary, Transaction, Pagination, PositionResponse, TransactionResponse

user_route = APIRouter(prefix="/user", tags=["user"])
PAGE_SIZE = 10


@user_route.get("/summary")
async def summary(jwt_payload: JWTPayload = Depends(verify_jwt)) -> UserSummary:
    pnl_vol_mt_q = select(
        coalesce(sql_sum(Bets.settle_amount), 0),
        coalesce(sql_sum(Bets.amount), 0),
        coalesce(count(Bets.user_id), 0),
    ).where(Bets.user_id == jwt_payload.sub)

    current_pos_value_q = select(coalesce(sql_sum(Bets.amount), 0)).where(
        (Bets.user_id == jwt_payload.sub) & (Bets.bet_status == BetStatus.OPEN.value)
    )

    async with get_db_session() as sess:
        r = await sess.execute(
            select(Users.username, Users.created_at).where(
                Users.user_id == jwt_payload.sub
            )
        )
        username, created_at = r.first()

        r = await sess.execute(pnl_vol_mt_q)
        pnl, volume, markets_traded = r.first()

        r = await sess.execute(current_pos_value_q)
        current_pos_value = r.first()[0]

    return UserSummary(
        username=username,
        joined_at=created_at.date(),
        total_pos_value=current_pos_value,
        pnl=pnl,
        volume=volume,
        markets_traded=markets_traded,
    )


@user_route.get("/positions")
async def positions(
    jwt_payload: JWTPayload = Depends(verify_jwt), page: int = 1
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
            (Bets.user_id == jwt_payload.sub)
            & (Bets.bet_status == BetStatus.OPEN.value)
        )
        .subquery()
    )

    async with get_db_session() as sess:
        # Get total count
        total_count_result = await sess.execute(count_query)
        total_items = total_count_result.scalar()
        
        # Calculate pagination info
        total_pages = math.ceil(total_items / PAGE_SIZE) if total_items > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1

        # Get paginated data
        r = await sess.execute(
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
                (Bets.user_id == jwt_payload.sub)
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

    positions_data = [
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
        total_quantity=total_items
    )

    return PositionResponse(
        data=positions_data,
        pagination=pagination
    )


@user_route.get("/activity")
async def activity(
    jwt_payload: JWTPayload = Depends(verify_jwt), page: int = 1
) -> TransactionResponse:
    if page < 1:
        page = 1

    # First, get the total count for pagination
    count_query = select(count()).select_from(
        select(Transactions.transaction_id)
        .where(Transactions.user_id == jwt_payload.sub)
        .subquery()
    )

    async with get_db_session() as sess:
        # Get total count
        total_count_result = await sess.execute(count_query)
        total_items = total_count_result.scalar()
        
        # Calculate pagination info
        total_pages = math.ceil(total_items / PAGE_SIZE) if total_items > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1

        # Get paginated data
        r = await sess.execute(
            select(
                Transactions.transaction_type,
                Transactions.amount,
                Transactions.address,
                Markets.title,
                Markets.category,
            )
            .where(Transactions.user_id == jwt_payload.sub)
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
        total_quantity=total_items
    )

    return TransactionResponse(
        data=transactions_data,
        pagination=pagination
    )