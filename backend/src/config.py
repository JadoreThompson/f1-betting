import os

from datetime import timedelta
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine


load_dotenv()


BASE_PATH = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_PATH, "data")


# DB
DB_USER_CREDS = f"{os.getenv("DB_USER")}:{quote(os.getenv("DB_PASSWORD"))}"
DB_HOST_CREDS = f"{os.getenv("DB_HOST")}:{os.getenv('DB_PORT')}"
DB_NAME = os.getenv("DB_NAME")
DB_URL = f"postgresql+asyncpg://{DB_USER_CREDS}@{DB_HOST_CREDS}/{DB_NAME}"
ASYNC_DB_ENGINE = create_async_engine(DB_URL)
SYNC_DB_ENGINE = create_engine(DB_URL.replace("+asyncpg", ""))


# Cookie and JWT
COOKIE_ALIAS = "f1-betting-cookie"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "secret-key")
JWT_EXPIRY = timedelta(minutes=1e6)
JWT_ALGO = "HS256"


F1_API_BASE_URL = os.getenv("POLLING_BASE_URL")
