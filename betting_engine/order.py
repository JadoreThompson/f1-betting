from enums import BetStatus
from .enums import OrderStatus


class Order:
    def __init__(self, payload: dict[str, str | int], expected_value: float) -> None:
        self.status = OrderStatus.PENDING
        self._payload = payload
        self._side = payload["side"]
        self._standing_bet_amount = payload["amount"]
        self._standing_ev = expected_value

    def reduce_unfilled_amount(self, amount: float) -> None:
        """Reduces the unfilled amount and expected value of the order.

        Updates the order status depending on whether the full amount has been filled.

        Args:
            amount (float): Amount to reduce from the standing bet and expected value.

        Raises:
            ValueError: If the amount is less than or equal to zero.
        """
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
        """Compares this order with another object or dictionary.

        Args:
            value (object): Another Order instance or dictionary to compare with.

        Returns:
            bool: True if the bet IDs match, False otherwise.

        Raises:
            NotImplemented: If comparison is not supported for the given type.
        """
        if isinstance(value, self.__class__):
            return self._payload["bet_id"] == value.payload["bet_id"]

        if isinstance(value, dict):
            return self._payload["bet_id"] == value["bet_id"]

        return id(self) == id(value)

    @property
    def payload(self) -> dict[str, str | int]:
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
