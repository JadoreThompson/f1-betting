from enums import Side
from server.models import CustomBaseModel


class CreateBet(CustomBaseModel):
    market_id: int
    side: Side
    amount: float
    wallet_address: str
