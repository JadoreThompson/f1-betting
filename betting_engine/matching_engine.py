from multiprocessing import Queue
from typing import Iterable
from r_mutex import LockClient

from betting_engine.pusher import Pusher
from config import LOCK_CHANNEL, REDIS_CLIENT
from enums import Side, BetStatus
from .enums import OrderStatus, Topic
from .order import Order
from .orderbook import OrderBook
from .typing import EnginePayload, Payload, SettlePayload


class MatchingEngine:
    """
    The core matching engine responsible for processing betting actions including:
    - Creating new orders
    - Closing unmatched orders
    - Settling markets

    The engine matches orders using a simple price-time priority mechanism and
    uses a Pusher instance to coordinate batched updates and notifications.

    Attributes:
        _pusher (Pusher): Manages updates and notifications to external systems.
        _orderbooks (dict[int, OrderBook]): A mapping of market IDs to their respective orderbooks.
        _queue (Queue): A multiprocessing queue used to receive order events.
    """

    def __init__(self, queue: Queue) -> None:
        """
        Initializes the MatchingEngine with a communication queue.

        Args:
            queue (Queue): Inter-process queue used to receive order-related events.
        """
        self._pusher = Pusher(LockClient(REDIS_CLIENT, LOCK_CHANNEL, False))
        self._orderbooks: dict[int, OrderBook] = {}
        self._queue = queue

    async def run(self) -> None:
        """
        Continuously consumes payloads from the queue and routes them to the appropriate
        handler based on their topic (CREATE, CLOSE, SETTLE).
        """
        await self._pusher.start()

        while True:
            payload: EnginePayload = self._queue.get()

            if payload["topic"] == Topic.CREATE:
                self._place_order(payload)
            elif payload["topic"] == Topic.CLOSE:
                self._close_order(payload)
            elif payload["topic"] == Topic.SETTLE:
                await self._settle_orderbook(payload)
            else:
                raise ValueError(f"Unknown topic: {payload['topic']}")

    def _place_order(self, payload: EnginePayload) -> None:
        """
        Handles a new order creation by attempting to match it immediately.
        If not matched, the order is added to the appropriate orderbook.

        Args:
            payload (EnginePayload): Contains bet and market data.
        """
        orderbook = self._orderbooks.setdefault(
            payload["bet"]["market_id"],
            OrderBook(
                payload["bet"]["market_id"],
                payload["market"]["numerator"],
                payload["market"]["denominator"],
                self._pusher
            ),
        )

        payload = payload["bet"]

        if payload["side"] == Side.BACK:
            expected_payout_value: float = (1 + orderbook.numerator) * payload["amount"]
        else:
            expected_payout_value: float = (
                1 + round(100 / (100 - (100 / orderbook.numerator)))
            ) * payload["amount"]

        expected_payout_value = round(expected_payout_value, 2)
        order = Order(payload, expected_payout_value)

        prev_bs = payload["bet_status"]
        is_matched = self._match_order(order, orderbook)

        if not is_matched:
            orderbook.append(order)
        
        if prev_bs != order.payload["bet_status"]:
            self._pusher.append(order.payload)

    def _close_order(self, payload: Payload) -> None:
        """
        Removes a specific order from the orderbook and updates its status.

        Args:
            payload (Payload): Contains order ID, market ID, and current bet status.
        """
        orderbook = self._orderbooks[payload["market_id"]]
        orderbook.remove(payload)

        if payload["bet_status"] == BetStatus.OPEN:
            payload["bet_status"] = BetStatus.CLOSED.value
        else:
            payload["bet_status"] = BetStatus.CANCELLED.value
        payload["closed_at"] = datetime.now()
            
        self._pusher.append(payload)

    def _match_order(
        self,
        order: Order,
        orderbook: OrderBook,
    ) -> bool:
        """
        Attempts to match an incoming order against existing resting orders
        in the orderbook. Orders must match fully or not at all—partial
        fills that fragment EV are rejected.

        Args:
            order (Order): The new incoming order.
            orderbook (OrderBook): The orderbook for the relevant market.

        Returns:
            bool: True if the order is fully matched; False otherwise.
        """
        filled_orders: list[Order] = []
        touched_payloads: list[dict] = []

        book: Iterable[Order] = (
            orderbook.asks if order.side == Side.BACK else orderbook.bids
        )

        for resting_order in book:
            min_bet_amount: float = min(
                order.standing_bet_amount, resting_order.standing_bet_amount
            )

            # Reject orders that result in fragmented bets
            if (
                order.standing_bet_amount - min_bet_amount == 0
                and order.standing_ev - min_bet_amount > 0
            ) or (
                resting_order.standing_bet_amount - min_bet_amount == 0
                and resting_order.standing_ev - min_bet_amount > 0
            ):
                continue

            order.reduce_unfilled_amount(min_bet_amount)
            resting_order.reduce_unfilled_amount(min_bet_amount)

            if resting_order.status == OrderStatus.FILLED:
                filled_orders.append(resting_order)
            
            touched_payloads.append(resting_order.payload)

            if order.status == OrderStatus.FILLED:
                break

        # Clean up filled orders
        for _order in filled_orders:
            orderbook.remove(_order)
            
        self._pusher.append(touched_payloads)

        return order.status == OrderStatus.FILLED

    async def _settle_orderbook(self, data: EnginePayload) -> None:
        """
        Settles a given orderbook using a list of winning user IDs and removes it from memory.

        Args:
            data (EnginePayload): Contains the `settle_data` field with market ID and winners.
        """
        data: SettlePayload = data["settle_data"]
        ob = self._orderbooks.setdefault(
            data["market_id"], OrderBook(data["market_id"], 2, 2)
        )

        if ob is None:
            return

        await ob.settle(data["winners"])
        self._orderbooks.pop(data["market_id"])
