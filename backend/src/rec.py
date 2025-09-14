import asyncio
from db_models import Drivers
from services import MarketGenerator

asyncio.run(MarketGenerator.generate(2023, 5))
