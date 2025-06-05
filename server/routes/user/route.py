from fastapi import APIRouter

from server.typing import JWTPayload

user_router = APIRouter(prefix="/user", tags=["user"])

@user_router.get("/summary")
async def summary(jwt_payload: JWTPayload): ...