from datetime import datetime
from .order import Order
from .enums import OrderStatus, Side
from .typing import Payload
from .orderbook import OrderBook


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

        matched, standing_amount = self._match_order(
            payload, orderbook, round(expected_payout_value, 2)
        )

        if not matched:
            orderbook.place_order(Order(payload, standing_amount))
        else:
            payload["status"] = OrderStatus.FILLED
        

    def close_order(self, payload: Payload) -> None:
        orderbook = self._orderbooks[payload["bet_id"]]
        orderbook.remove_order(payload)

        if payload["status"] == OrderStatus.FILLED:
            payload["status"] = OrderStatus.CLOSED
        else:
            payload["status"] = OrderStatus.CANCELLED

    def _match_order(
        self, payload: Payload, orderbook: OrderBook, expected_payout_value: float
    ) -> tuple[bool, float]:
        filled_orders: list[Order] = []

        book = orderbook.asks if payload["side"] == Side.BID else orderbook.bids
        standing_bet_amount = expected_payout_value
        result = False

        for resting_order in book:
            minimum = min(standing_bet_amount, resting_order._standing_bet_amount)
            resting_order.reduce_standing_bet_amount(minimum)
            standing_bet_amount -= minimum

            if resting_order.standing_bet_amount == 0:
                filled_orders.append(resting_order)

            if standing_bet_amount == 0:
                result = True
                break

        close_time = datetime.now()
        for order in filled_orders:
            order.payload["closed_at"] = close_time
            orderbook.remove_order(order)

        return result, standing_bet_amount
