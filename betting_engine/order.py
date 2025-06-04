from __future__ import annotations
from enums import BetStatus
from .enums import OrderStatus
from .typing import Payload


class Order:
    def __init__(self, payload: Payload, expected_value: float) -> None:
        self.status = OrderStatus.PENDING
        self._payload = payload
        self._side = payload["side"]
        self._standing_bet_amount = payload["amount"]
        self._standing_ev = expected_value

    def reduce_unfilled_amount(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount to reduce must be greater than 0.")

        self._standing_bet_amount -= amount
        self._standing_ev -= amount

        if self._standing_ev == 0:
            self.status = OrderStatus.FILLED
            self._payload["bet_status"] = BetStatus.OPEN.value
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return self._payload["bet_id"] == value.payload["bet_id"]

        if isinstance(value, dict):
            return self._payload["bet_id"] == value["bet_id"]

        raise NotImplemented

    @property
    def payload(self) -> Payload:
        return self._payload

    @property
    def side(self) -> int:
        return self._side

    @property
    def standing_bet_amount(self) -> int:
        return self._standing_bet_amount

    @property
    def standing_ev(self) -> int:
        return self._standing_ev
