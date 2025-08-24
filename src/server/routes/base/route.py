import os
from datetime import datetime
from json import load

from fastapi import APIRouter, Depends
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import DATA_FOLDER
from db_models import Constructors, Drivers, Predictions, Qualifyings
from enums import PredictionStatus
from server.dependencies import depends_db_session
from utils.utils import get_datetime
from .models import Driver, PredictionsResponse


route = APIRouter(prefix="")


@route.get("/predictions")
async def get_predictions_simple(db_sess: AsyncSession = Depends(depends_db_session)):
    query = (
        select(Predictions, Drivers, Constructors)
        .select_from(Predictions)
        .join(Drivers, Predictions.driver_id == Drivers.driver_id)
        .join(Qualifyings, Qualifyings.driver_id == Predictions.driver_id)
        .join(Constructors, Qualifyings.constructor_id == Constructors.constructor_id)
        .where(Predictions.status == PredictionStatus.OPEN.value)
        .order_by(Predictions.driver_id, desc(Qualifyings.created_at))
        .distinct(Predictions.driver_id)
    )

    res = await db_sess.execute(query)
    preds = res.all()
    print(preds[0][0].predicted_probability)

    return [
        PredictionsResponse(
            driver=Driver(
                driver_name=driver.driver_ref,
                nationality=driver.nationality,
                constructor=constructor.constructor_ref,
            ),
            predicted_position=str(pred.predicted_position),
            predicted_probability=pred.predicted_probability,
        )
        for pred, driver, constructor in preds
    ]
