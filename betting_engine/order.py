from .enums import OrderStatus
from .typing import Payload


class Order:
    def __init__(self, payload: Payload, expected_value: float) -> None:
        self._payload = payload
        self._side = payload["side"]
        self._standing_bet_amount = expected_value

    def reduce_standing_bet_amount(self, amount: int) -> None:
        self._standing_bet_amount -= amount

        if self._standing_bet_amount == 0:
            self._payload["status"] = OrderStatus.FILLED
        else:
            self._payload["status"] = OrderStatus.PARTIALLY_FILLED

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return self._payload["order_id"] == value.payload["order_id"]

        if isinstance(value, dict):
            return self._payload["order_id"] == value["order_id"]

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
