from typing import Iterable
from enums import Side
from .order import Order


class OrderBook:
    def __init__(self, numerator: int, denominator: int) -> None:
        self._numerator = numerator
        self._denominator = denominator
        self._bids: dict[str, Order] = {}
        self._bid_orders = self._bids.values()
        self._asks: dict[str, Order] = {}
        self._ask_orders = self._asks.values()

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
