from typing import Iterable
from .enums import Side
from .order import Order


class OrderBook:
    def __init__(self, numerator: int, denominator: int) -> None:
        self._numerator = numerator
        self._denominator = denominator
        self._bids: dict[str, Order] = {}
        self._bid_orders = self._bids.values()
        self._asks: dict[str, Order] = {}
        self._ask_orders = self._asks.values()

    def place_order(self, order: Order) -> None:
        order_id = order.payload["order_id"]

        if order.side == Side.BID:
            self._bids[order_id] = order
        else:
            self._asks[order_id] = order

    def remove_order(self, order: Order | dict) -> None:
        order_id = (
            order.payload["order_id"] if isinstance(order, Order) else order["order_id"]
        )

        if (
            order.payload["side"] if isinstance(order, Order) else order["side"]
        ) == Side.BID:
            if order_id in self._bids:
                self._bids.pop(order_id)
        else:
            if order_id in self._asks:
                self._asks.pop(order_id)

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
