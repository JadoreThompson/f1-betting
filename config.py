import redis
import os
import ydf

from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy.ext.asyncio import create_async_engine


load_dotenv()
BPATH = os.path.dirname(__file__)

# DB
DB_URL = f"postgresql+asyncpg://{os.getenv("DB_USER")}:{quote(os.getenv("DB_PASSWORD"))}@{os.getenv("DB_HOST")}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
DB_ENGINE = create_async_engine(
    DB_URL,
    future=True,
    echo_pool=True,
    # pool_size=10,
    # max_overflow=20,
    # pool_timeout=30,
    # pool_recycle=6000,
)


# Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
REDIS_CLIENT = redis.asyncio.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    db=0,
    connection_pool=redis.asyncio.ConnectionPool(
        connection_class=redis.asyncio.Connection,
        max_connections=100,
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        retry_on_timeout=True,
    ),
)

# Extras
# MODEL = ydf.load_model(os.path.join(BPATH, "engine", "models", "model_1"))
