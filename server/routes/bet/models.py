from enums import Side
from server.models import CustomBaseModel


class Bet(CustomBaseModel):
    market_id: int
    side: Side
    amount: float
    wallet_address: str
    txn_address: str
