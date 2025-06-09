from sqlalchemy import insert, update
from typing import Iterable

from config import PRIVATE_KEY
from db_models import Bets, Markets, Transactions
from enums import MarketStatus, Side, BetStatus, TransactionType
from utils.db import get_db_session
from utils.utils import get_datetime

from .config import PROVIDER, USDT_CONTRACT, BE_CONTRACT
from .pusher import Pusher
from .order import Order


class OrderBook:
    def __init__(self, market_id: int, numerator: int, denominator: int, pusher: Pusher) -> None:
        self._market_id = market_id
        self._numerator = numerator
        self._denominator = denominator
        self._bids: dict[str, Order] = {}
        self._bid_orders = self._bids.values()
        self._asks: dict[str, Order] = {}
        self._ask_orders = self._asks.values()
        
        if not pusher.is_running:
            raise RuntimeError("Pusher must be running prior to initialisation.")
        self._pusher = pusher

    def append(self, order: Order) -> None:
        bet_id = order.payload["bet_id"]

        if order.side == Side.BACK:
            self._bids[bet_id] = order
        else:
            self._asks[bet_id] = order

    def remove(self, order: Order | dict) -> None:
        if not isinstance(order, (Order, dict)):
            raise TypeError("order must be an instance of Order or a dictionary.")

        bet_id = (
            order.payload["bet_id"] if isinstance(order, Order) else order["bet_id"]
        )

        if (order.side if isinstance(order, Order) else order["side"]) == Side.BACK:
            if bet_id in self._bids:
                self._bids.pop(bet_id)
        else:
            if bet_id in self._asks:
                self._asks.pop(bet_id)

    async def settle(self, side: Side) -> None:
        if side == Side.BACK:
            winners = self._bids.values()
        else:
            winners = self._asks.values()

        k: int = self._numerator if side == Side.BACK else self._denominator
        k += 1
        usdt_decimals = await USDT_CONTRACT.functions.decimals().call()

        close_time = get_datetime()
        for w in winners:
            wallet_addr = w.payload["wallet_address"]
            payout = w.payload["amount"] * k
            w.payload["settlement_amount"] = payout
            payout_usdt = payout * 10**usdt_decimals

            txn = await BE_CONTRACT.functions.withdraw(
                self._market_id,
                wallet_addr,
                int(payout_usdt),
            ).build_transaction(
                {
                    "nonce": await PROVIDER.eth.get_transaction_count(wallet_addr),
                    "gas": 300000,  # optional: specify to avoid estimateGas
                    "gasPrice": await PROVIDER.eth.gas_price,
                }
            )

            signed_txn = PROVIDER.eth.account.sign_transaction(txn, PRIVATE_KEY)
            tx_hash = await PROVIDER.eth.send_raw_transaction(
                signed_txn.raw_transaction
            )

            w.payload["settlement_txn"] = tx_hash.to_0x_hex()
            w.payload["bet_status"] = BetStatus.SETTLED.value
            w.payload["closed_at"] = close_time
            
        self._pusher.append(list(winners))

        async with get_db_session() as s:
            await s.execute(update(Bets), [w.payload for w in winners])
            await s.execute(
                insert(Transactions),
                [
                    {
                        "user_id": w.payload["user_id"],
                        "bet_id": w.payload["bet_id"],
                        "market_id": w.payload["market_id"],
                        "transaction_type": TransactionType.SETTLE.value,
                        "amount": w.payload["amount"],
                        "address": w.payload["settlement_txn"]
                    }
                    for w in winners
                ],
            )

            await s.execute(
                update(Markets)
                .values(market_status=MarketStatus.SETTLED.value)
                .where(Markets.market_id == self._market_id)
            )

            await s.commit()

    @property
    def bids(self) -> Iterable[Order]:
        return self._bid_orders

    @property
    def asks(self) -> Iterable[Order]:
        return self._ask_orders

    @property
    def numerator(self) -> int:
        return self._numerator

    @property
    def denominator(self) -> int:
        return self._denominator
