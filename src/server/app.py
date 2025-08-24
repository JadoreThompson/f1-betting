from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import base_route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(base_route)