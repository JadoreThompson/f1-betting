from datetime import datetime
from typing import Iterable
from .enums import OrderStatus, Side
from .order import Order
from .orderbook import OrderBook
from .typing import Payload


class MatchingEngine:
    def __init__(self) -> None:
        self._orderbooks: dict[str, OrderBook] = {}

    def place_order(self, payload: dict[str, Payload]) -> None:
        orderbook = self._orderbooks.setdefault(
            payload["order"]["bet_id"],
            OrderBook(payload["bet"]["numerator"], payload["bet"]["denominator"]),
        )

        payload = payload["order"]

        if payload["side"] == Side.BID:
            expected_payout_value: float = orderbook.numerator * payload["bet_amount"]
        else:
            expected_payout_value: float = (
                round(100 / (100 - (100 / orderbook.numerator)))
            ) * payload["bet_amount"]

        expected_payout_value = round(expected_payout_value, 2)
        order = Order(payload, expected_payout_value)

        is_matched = self._match_order(order, orderbook)

        if not is_matched:
            orderbook.append(order)

    def close_order(self, payload: Payload) -> None:
        orderbook = self._orderbooks[payload["bet_id"]]
        orderbook.remove(payload)

        if payload["status"] == OrderStatus.FILLED:
            payload["status"] = OrderStatus.CLOSED
        else:
            payload["status"] = OrderStatus.CANCELLED

    def _match_order(
        self,
        order: Order,
        orderbook: OrderBook,
    ) -> bool:
        filled_orders: list[Order] = []

        book: Iterable[Order] = (
            orderbook.asks if order.side == Side.BID else orderbook.bids
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

            if resting_order.payload["status"] == OrderStatus.FILLED:
                filled_orders.append(resting_order)

            if order.payload["status"] == OrderStatus.FILLED:
                break

        # Clean up filled orders
        close_time = datetime.now()
        for _order in filled_orders:
            _order.payload["closed_at"] = close_time
            orderbook.remove(_order)

        return order.payload["status"] == OrderStatus.FILLED
