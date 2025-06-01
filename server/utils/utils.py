import jwt

from datetime import datetime
from fastapi import HTTPException
from typing import Any
from sqlalchemy import select

from config import JWT_SECRET_KEY, JWT_ALGO, JWT_EXPIRY
from db_models import Users
from .db import get_db_session


class JWTError(Exception):
    """Custom exception for JWT errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def generate_jwt(payload: dict[str, Any]) -> str:
    payload["exp"] = datetime.now() + JWT_EXPIRY
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGO)


def decode_jwt(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise JWTError("Token has expired")
    except jwt.InvalidTokenError:
        raise JWTError("Invalid token")


async def validate_jwt_payload(decoded_payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a JWT token and return the decoded payload.

    Args:
        payload (dict[str, Any]): The payload containing the JWT token.

    Raises:
        HTTPException: If the user is not found or the token is invalid.

    Returns:
        dict[str, Any]: The decoded JWT payload if valid.
    """
    async with get_db_session() as sess:
        res = await sess.execute(
            select(Users).where(Users.user_id == decoded_payload["sub"])
        )
        user = res.first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid user")

    return decoded_payload
