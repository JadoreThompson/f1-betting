from uuid import UUID
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from db_models import Users
from config import COOKIE_ALIAS
from server.routes.auth.models import LoginBody, RegisterBody
from utils.db import get_db_session
from server.utils.utils import generate_jwt

auth_route = APIRouter(prefix="/auth", tags=["auth"])


def set_cookie(rsp: Response, user_id: UUID) -> Response:
    rsp.set_cookie(
        key=COOKIE_ALIAS, value=generate_jwt({"sub": str(user_id)}), httponly=True
    )
    return rsp


@auth_route.post("/register")
async def register(body: RegisterBody):
    try:
        async with get_db_session() as sess:
            res = await sess.execute(
                insert(Users)
                .values(
                    username=body.username, email=body.email, password=body.password
                )
                .returning(Users.user_id)
            )
            user_id = res.scalar_one()
            await sess.commit()

        rsp = JSONResponse(
            status_code=200, content={"message": "User created successfully"}
        )
        return set_cookie(rsp, user_id)
    except IntegrityError:
        return JSONResponse(status_code=409, content={"error": "User already exists"})


@auth_route.post("/login")
async def login(body: LoginBody):
    async with get_db_session() as sess:
        res = await sess.execute(
            select(Users).where(
                (Users.username == body.login) | (Users.email == body.login)
            )
        )
        user = res.scalar_one_or_none()

        if not user or user.password != body.password:
            return JSONResponse(
                status_code=401, content={"error": "Invalid credentials"}
            )

    rsp = JSONResponse(status_code=200, content={"message": "Login successful"})
    return set_cookie(rsp, user.user_id)
