import jwt

from datetime import datetime
from sqlalchemy import select
from typing import Any

from config import JWT_SECRET_KEY, JWT_ALGO, JWT_EXPIRY
from db_models import Users
from utils.db import get_db_session
from ..typing import JWTPayload
from ..exc import JWTError


def generate_jwt(payload: dict[str, Any]) -> str:
    """Generates a JWT token

    Args:
        payload (dict[str, Any]): kwargs for JWT object

    Returns:
        str: JWT token
    """
    payload["exp"] = datetime.now() + JWT_EXPIRY
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGO)


def decode_jwt(token: str) -> JWTPayload:
    try:
        return JWTPayload(**jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGO]))
    except jwt.ExpiredSignatureError:
        raise JWTError("Token has expired")
    except jwt.InvalidTokenError:
        raise JWTError("Invalid token")


async def validate_jwt_payload(payload: JWTPayload) -> JWTPayload:
    """Validate a JWT token and return the decoded payload.

    Args:
        payload (JWTPayload): The decoded JWT payload.

    Raises:
        JWTError: If the user referenced in the payload does not exist.

    Returns:
        JWTPayload: The validated JWT payload.
    """
    async with get_db_session() as sess:
        res = await sess.execute(
            select(Users).where(Users.user_id == payload.sub)
        )
        user = res.first()

    if not user:
        raise JWTError("Invalid user")

    return payload
