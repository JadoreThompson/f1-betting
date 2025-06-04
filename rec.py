import asyncio
import json

import aiohttp
from sqlalchemy import insert
from betting_engine.config import BE_CONTRACT, PROVIDER
from betting_engine.pollers.settlement_poller import SettlementPoller
from config import PRIVATE_KEY
from db_models import Markets
from enums import MarketCategory
from utils.db import (
    alembic_revision,
    alembic_ugrade_head,
    get_db_session,
    remove_sqlalchemy_url,
    write_sqlalchemy_url,
)


def alembic_revision_wrapper(msg) -> None:
    if not msg:
        return

    alembic_revision(msg)
    alembic_ugrade_head()


async def gen_markets():
    def helper(category: MarketCategory) -> list[dict]:
        return [
            {
                "title": f"{driver}",
                "numerator": 1,
                "denominator": 1,
                "category": category.value,
            }
            for _ in range(max_markets // len(drivers))
            for driver in drivers
        ]

    max_markets = 1000
    drivers = [
        "max_verstappen",
        "hamilton",
        "leclerc",
        "norris",
        "perez",
        "russell",
        "sainz",
        "alonso",
        "piastri",
        "stroll",
    ]

    winner_markets = helper(MarketCategory.WINNER)
    top3_markets = helper(MarketCategory.TOP3)

    try:
        async with get_db_session() as sess:
            await sess.execute(insert(Markets).values(winner_markets + top3_markets))
            await sess.commit()
    except Exception as e:
        print(str(e))


class A:
    def __init__(self):
        self.payload = None


async def f():
    data = await SettlementPoller.poll()
    json.dump(data, open("jolpica_race_data.json", "w"))


# write_sqlalchemy_url()
# remove_sqlalchemy_url()
# alembic_revision_wrapper("Added wins to driver standings table")
# asyncio.run(gen_markets())
# asyncio.run(f())
# asyncio.run(Poller.poll())
