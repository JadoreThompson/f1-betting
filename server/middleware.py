from fastapi import Request
from config import COOKIE_ALIAS, WALLET_HEADER_KEY
from .exc import JWTError, MissingHeaderError
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


def requires_wallet_address(req: Request) -> str:
    """
    Extracts the wallet address from the request headers.

    Raises:
        MissingHeaderError: If the wallet address header is not present.

    Returns:
        str: Wallet address from headers.
    """
    headers = req.headers

    if WALLET_HEADER_KEY in headers:
        return headers[WALLET_HEADER_KEY]
    raise MissingHeaderError(f"{WALLET_HEADER_KEY} is missing from request header.")
