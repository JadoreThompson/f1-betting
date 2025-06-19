from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .exc import JWTError, MissingHeaderError
from .routes import markets_route, user_route

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

app.include_router(markets_route)
app.include_router(user_route)


@app.exception_handler(JWTError)
async def jwt_error_handler(req: Request, exc: JWTError):
    return JSONResponse(status_code=401, content={"error": str(exc)})


@app.exception_handler(MissingHeaderError)
async def missing_header_error_handler(req: Request, exc: MissingHeaderError):
    return JSONResponse(status_code=400, content={"error": str(exc)})
