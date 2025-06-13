from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .exc import JWTError
from .routes import auth_route, bet_route, markets_route, user_route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:5173",
        "http://192.168.1.145:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_route)
app.include_router(bet_route)
app.include_router(markets_route)
app.include_router(user_route)


@app.exception_handler(JWTError)
async def jwt_error_handler(req: Request, exc: JWTError):
    return JSONResponse(status_code=401, content={"error": exc.message})
