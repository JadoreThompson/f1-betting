from datetime import timedelta
import redis
import os

from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine


load_dotenv()

BPATH = os.path.dirname(__file__)
SERVER_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(SERVER_DATA_FOLDER):
    os.mkdir(SERVER_DATA_FOLDER)

# DB
DB_URL = f"postgresql+asyncpg://{os.getenv("DB_USER")}:{quote(os.getenv("DB_PASSWORD"))}@{os.getenv("DB_HOST")}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
ASYNC_DB_ENGINE = create_async_engine(
    DB_URL,
    future=True,
    echo_pool=True,
    # pool_size=10,
    # max_overflow=20,
    # pool_timeout=30,
    # pool_recycle=6000,
)
SYNC_DB_ENGINE = create_engine(DB_URL.replace("+asyncpg", ""))


# Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
REDIS_CLIENT = redis.asyncio.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    connection_pool=redis.asyncio.ConnectionPool(
        connection_class=redis.asyncio.Connection,
        max_connections=100,
        host=REDIS_HOST,
        port=REDIS_PORT,
        retry_on_timeout=True,
    ),
)
ORDER_UPDATE_CHANNEL = os.getenv("ORDER_UPDATE_CHANNEL")
LOCK_CHANNEL = os.getenv("LOCK_CHANNEL")

# Cookie and JWT
COOKIE_ALIAS = "f1-betting-cookie"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "secret-key")
JWT_EXPIRY = timedelta(minutes=1e6)
JWT_ALGO = "HS256"


POLLING_BASE_URL = os.getenv("POLLING_BASE_URL")
INFURA_API_KEY = os.getenv("INFURA_API_KEY")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
BE_CONTRACT_ADDR = os.getenv("BE_CONTRACT")
USDT_CONTRACT_ADDR = os.getenv("USDT_CONTRACT")
