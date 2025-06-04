import asyncio
import pandas as pd
from sqlalchemy import select
from db_models import Circuits, ConstructorStandings, Constructors, DriverStandings, Drivers, GrandPrixResults, QualiResults
from utils.db import get_db_session


class Pipeline:
    async def start(self):
        d = await self._get_datasets()
        

    async def _get_datasets(self) -> dict[str, pd.DataFrame]:
        async with get_db_session() as sess:
            # Load data from database
            circuits = (await sess.execute(select(Circuits))).scalars().all()
            constructors = (await sess.execute(select(Constructors))).scalars().all()
            drivers = (await sess.execute(select(Drivers))).scalars().all()
            driver_standings = (await sess.execute(select(DriverStandings))).scalars().all()
            constructor_standings = (await sess.execute(select(ConstructorStandings))).scalars().all()
            results = (await sess.execute(select(GrandPrixResults))).scalars().all()
            qualifying = (await sess.execute(select(QualiResults))).scalars().all()
            
        


p = Pipeline()
asyncio.run(p.start())
