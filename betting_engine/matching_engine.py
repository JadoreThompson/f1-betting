from datetime import datetime
from multiprocessing import Queue
from typing import Iterable

from enums import Side, BetStatus
from .enums import OrderStatus, Topic
from .order import Order
from .orderbook import OrderBook
from .typing import EnginePayload, Payload, SettlePayload


class MatchingEngine:
    def __init__(self, queue: Queue) -> None:
        self._orderbooks: dict[int, OrderBook] = {}
        self._queue = queue

    async def run(self) -> None:
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
        orderbook = self._orderbooks.setdefault(
            payload["bet"]["market_id"],
            OrderBook(
                payload["bet"]["market_id"],
                payload["market"]["numerator"],
                payload["market"]["denominator"],
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

        is_matched = self._match_order(order, orderbook)

        if not is_matched:
            orderbook.append(order)

    def _close_order(self, payload: Payload) -> None:
        orderbook = self._orderbooks[payload["market_id"]]
        orderbook.remove(payload)

        if payload["bet_status"] == BetStatus.OPEN:
            payload["bet_status"] = BetStatus.CLOSED.value
        else:
            payload["bet_status"] = BetStatus.CANCELLED.value

    def _match_order(
        self,
        order: Order,
        orderbook: OrderBook,
    ) -> bool:
        filled_orders: list[Order] = []

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

            if order.status == OrderStatus.FILLED:
                break

        # Clean up filled orders
        close_time = datetime.now()
        for _order in filled_orders:
            _order.payload["closed_at"] = close_time
            orderbook.remove(_order)

        return order.status == OrderStatus.FILLED

    async def _settle_orderbook(self, data: EnginePayload) -> None:
        data: SettlePayload = data["settle_data"]
        ob = self._orderbooks.setdefault(
            data["market_id"], OrderBook(data["market_id"], 2, 2)
        )

        if ob is None:
            return

        await ob.settle(data["winners"])
        self._orderbooks.pop(data["market_id"])
