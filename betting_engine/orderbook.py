import warnings

from sqlalchemy import insert, update
from typing import Iterable
from web3.exceptions import ContractLogicError

from config import PRIVATE_KEY
from db_models import Bets, Markets, Transactions
from enums import MarketStatus, Side, BetStatus, TransactionType
from utils.db import get_db_session
from utils.utils import get_datetime

from .config import PROVIDER, USDT_CONTRACT, BE_CONTRACT
from .exc import OrderBookError
from .pusher import Pusher
from .order import Order


class OrderBook:
    """
    Manages the in-memory order book for a specific betting market.

    Responsible for maintaining bid (BACK) and ask (LAY) orders, handling settlements
    via on-chain transactions, pushing updates, and persisting changes to the database.

    Attributes:
        _market_id (int): Unique identifier for the market.
        _numerator (int): Multiplier used for BACK-side payouts.
        _denominator (int): Multiplier used for LAY-side payouts.
        _bids (dict[str, Order]): BACK-side orders, keyed by bet ID.
        _asks (dict[str, Order]): LAY-side orders, keyed by bet ID.
        _pusher (Pusher): Used to push real-time updates to external consumers.
    """

    def __init__(
        self, market_id: int, numerator: int, denominator: int, pusher: Pusher
    ) -> None:
        """
        Initializes the order book with the provided market configuration.

        Args:
            market_id (int): Unique identifier of the market.
            numerator (int): BACK-side payout multiplier.
            denominator (int): LAY-side payout multiplier.
            pusher (Pusher): An active Pusher instance.

        Raises:
            RuntimeError: If the provided Pusher is not running.
        """
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
        """
        Adds an order to the order book.

        Args:
            order (Order): The order to add.

        Raises:
            OrderBookError: If an order with the same bet ID already exists.
        """
        bet_id = order.payload["bet_id"]
        book = self._bids if order.side == Side.BACK else self._asks

        if bet_id in book:
            raise OrderBookError("Order already exists.")

        book[bet_id] = order

    def remove(self, order: Order | dict) -> None:
        """
        Removes an order from the order book.

        Args:
            order (Order | dict): The order or payload to remove.

        Raises:
            TypeError: If the input is not an Order or dict.
        """
        if not isinstance(order, (Order, dict)):
            raise TypeError("order must be an instance of Order or a dictionary.")

        bet_id = (
            order.payload["bet_id"] if isinstance(order, Order) else order["bet_id"]
        )
        side = order.side if isinstance(order, Order) else order["side"]

        if side == Side.BACK:
            self._bids.pop(bet_id, None)
        else:
            self._asks.pop(bet_id, None)

    async def _handle_payout(
        self, k: int, usdt_decimals: int, orders: list[Order]
    ) -> None:
        """
        Processes and sends payouts to winners via blockchain.

        Args:
            k (int): Payout multiplier for the order (e.g., 2 for BACK win).
            usdt_decimals (int): USDT decimal precision.
            orders (list[Order]): Orders to pay out.

        Modifies:
            - Updates order payload with payout amount, transaction hash, and status.
        """
        warning_template = "Settlement failed for bet {bet_id}"
        close_time = get_datetime()

        for o in orders:
            wallet_addr = o.payload["wallet_address"]
            payout = o.payload["amount"] * k

            txn = await BE_CONTRACT.functions.withdraw_(
                self._market_id, wallet_addr, k
            ).build_transaction(
                {
                    "nonce": await PROVIDER.eth.get_transaction_count(wallet_addr),
                    "gas": 300000,
                    "gasPrice": await PROVIDER.eth.gas_price,
                }
            )

            signed_txn = PROVIDER.eth.account.sign_transaction(txn, PRIVATE_KEY)

            try:
                tx_hash = await PROVIDER.eth.send_raw_transaction(
                    signed_txn.raw_transaction
                )
            except ContractLogicError:
                warnings.warn(warning_template.format(bet_id=o.payload["bet_id"]))
                continue

            o.payload.update(
                {
                    "settlement_txn": tx_hash.to_0x_hex(),
                    "settlement_amount": payout,
                    "bet_status": BetStatus.SETTLED.value,
                    "closed_at": close_time,
                }
            )

    # TODO: Payout all unfilled orders.
    async def settle(self, side: Side) -> None:
        """
        Settles all orders on the specified side (BACK or LAY).

        Performs:
            - On-chain payouts.
            - Local status updates.
            - Database persistence.
            - Market status closure.

        Args:
            side (Side): Side of the market to settle.
        """
        winners = self._bids.values() if side == Side.BACK else self._asks.values()
        filled_winners = []
        unfilled_winners = []

        for w in winners:
            if w.payload["bet_status"] == BetStatus.OPEN.value:
                filled_winners.append(w)
            else:
                unfilled_winners.append(w)

        k: int = 1 + (self._numerator if side == Side.BACK else self._denominator)
        usdt_decimals = await USDT_CONTRACT.functions.decimals().call()

        await self._handle_payout(k, usdt_decimals, filled_winners)
        await self._handle_payout(1, usdt_decimals, unfilled_winners)

        self._pusher.append(winners)

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
                        "address": w.payload["settlement_txn"],
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
        """
        Returns:
            Iterable[Order]: Collection of current BACK orders.
        """
        return self._bid_orders

    @property
    def asks(self) -> Iterable[Order]:
        """
        Returns:
            Iterable[Order]: Collection of current LAY orders.
        """
        return self._ask_orders

    @property
    def numerator(self) -> int:
        """
        Returns:
            int: Multiplier for BACK-side payouts.
        """
        return self._numerator

    @property
    def denominator(self) -> int:
        """
        Returns:
            int: Multiplier for LAY-side payouts.
        """
        return self._denominator
