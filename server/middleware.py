from fastapi import Request
from config import COOKIE_ALIAS
from .exc import JWTError
from .utils.utils import decode_jwt, validate_jwt_payload
from .typing import JWTPayload


async def verify_jwt(req: Request) -> JWTPayload:
    """Verify the JWT token from the request cookies and validate it.

    Args:
        req (Request)

    Raises:
        JWTError: If the JWT token is missing, expired, or invalid.

    Returns:
        JWTPayload: The decoded JWT payload if valid.
    """
    token = req.cookies.get(COOKIE_ALIAS)

    if not token:
        raise JWTError("Authentication token is missing")

    payload = decode_jwt(token)
    return await validate_jwt_payload(payload)
