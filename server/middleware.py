from typing import Any
from fastapi import HTTPException, Request
from config import COOKIE_ALIAS
from server.utils.utils import decode_jwt, validate_jwt_payload


async def verify_jwt(req: Request) -> dict[str, Any]:
    """Verify the JWT token from the request cookies and validate it.

    Args:
        req (Request)

    Raises:
        HTTPException: If the JWT token is missing or invalid.

    Returns:
        dict[str, Any]: The decoded JWT payload if valid.
    """
    print("Verifying JWT...")
    print(f"Cookies: {req.cookies}")
    token = req.cookies.get(COOKIE_ALIAS)

    if not token:
        raise HTTPException(status_code=401, detail="Authentication token is missing")

    payload = decode_jwt(token)
    return await validate_jwt_payload(payload)
