from multiprocessing import Queue
from r_mutex import LockClient
from sqlalchemy import insert, select
from typing import Any, Optional

from db_models import Bets, Markets, Transactions
from enums import MarketStatus, Side, TransactionType
from utils.db import get_db_session
from utils.utils import dump_sqlalchemy_object
from ..config import BETTING_ESCROW_CONTRACT
from ..enums import Topic


class BetService:
    def __init__(self, queue: Queue, lock: LockClient) -> None:
        self._lock = lock
        self._queue = queue

    async def run(self) -> None:
        """
        Starts the service and begins listening for BetPlaced blockchain events.

        This method initializes the lock and subscribes to new BetPlaced logs from the
        latest block.
        """
        await self._lock.run()
        bp_filter = await BETTING_ESCROW_CONTRACT.events.BetPlaced.create_filter(
            from_block="latest"
        )
        while True:
            events = await bp_filter.get_new_entries()
            for ev in events:
                await self._handle_bet_placed(ev)

    async def _handle_bet_placed(self, event: dict) -> None:
        """
        Processes a single BetPlaced event from the blockchain.

        Performs validation against market state, inserts the bet and corresponding
        transaction into the database, and pushes a structured payload to the engine queue.

        Args:
            event (dict): Raw blockchain event data emitted by the BetPlaced event.

        Side Effects:
            - Writes to Bets and Transactions tables.
            - Commits changes to the DB.
            - Sends a payload to the queue.
        """
        payload = event["args"]
        wallet_address = payload["participant"].lower()
        amount = payload["amount"] / 1e6

        async with self._lock:
            async with get_db_session() as sess:
                res = await sess.execute(
                    select(
                        Markets.market_id,
                        Markets.numerator,
                        Markets.denominator,
                        Markets.market_status,
                    ).where(Markets.market_id == payload["marketId"])
                )
                market = res.first()

                if not market:
                    return

                market_id, numerator, denominator, market_status = market

                if market_status != MarketStatus.OPEN.value:
                    return

                res = await sess.execute(
                    insert(Bets)
                    .values(
                        market_id=market_id,
                        side=(
                            Side.BACK.value if payload["side"] == 0 else Side.LAY.value
                        ),
                        amount=amount,
                        wallet_address=wallet_address,
                    )
                    .returning(Bets)
                )
                placed_bet: Bets = res.scalar_one()

                await sess.execute(
                    insert(Transactions).values(
                        bet_id=placed_bet.bet_id,
                        market_id=payload["marketId"],
                        transaction_type=TransactionType.DEPOSIT.value,
                        amount=amount,
                        address=f"0x{event["transactionHash"].hex()}",
                        wallet_address=wallet_address,
                    )
                )

                await sess.commit()

        self._push_to_engine(
            Topic.CREATE,
            market={
                "market_id": market_id,
                "numerator": numerator,
                "denominator": denominator,
            },
            bet=dump_sqlalchemy_object(placed_bet),
        )

    def _push_to_engine(
        self,
        topic: Topic,
        *,
        market: Optional[dict[str, Any]] = None,
        bet: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "topic": topic,
        }

        if market:
            payload["market"] = market
        if bet:
            payload["bet"] = bet

        self._queue.put(payload)
