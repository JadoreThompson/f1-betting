from sqlalchemy import select
from db_models import Drivers, F1Data
from engine import interact, Prediction
from server.utils.db import get_db_session



async def get_predictions() -> tuple[int, str, str, float]:

    async with get_db_session() as sess:
        res = await sess.execute(
            select(
                F1Data.driver_id,
                Drivers.name,
                F1Data.last_1,
                F1Data.last_2,
                F1Data.last_3,
                F1Data.last_4,
                F1Data.last_5,
            ).join(Drivers, F1Data.driver_id == Drivers.driver_id)
        )

        last_races = res.all()

    predictions: tuple[Prediction, ...] = interact([l[2:] for l in last_races])

    return tuple(
        (
            last_races[i][0],
            last_races[i][1],
            predictions[i].prediction,
            predictions[i].percentage,
        )
        for i in range(len(last_races))
    )
