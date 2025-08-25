from io import BytesIO
import json
import os
from datetime import datetime
from json import load

import aiofiles
from fastapi import APIRouter, Depends
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import BASE_PATH
from core.typing import F1ModelConfig
from db_models import Constructors, Drivers, Predictions, Qualifyings
from enums import PredictionOutcome, PredictionStatus
from server.dependencies import depends_db_session
from .models import Driver, PredictionsResponse


route = APIRouter(prefix="")


@route.get("/predictions")
async def get_predictions_simple(
    outcome_class: PredictionOutcome = PredictionOutcome.ALL,
    db_sess: AsyncSession = Depends(depends_db_session),
):
    query = (
        select(Predictions, Drivers, Constructors)
        .select_from(Predictions)
        .join(Drivers, Predictions.driver_id == Drivers.driver_id)
        .join(Qualifyings, Qualifyings.driver_id == Predictions.driver_id)
        .join(Constructors, Qualifyings.constructor_id == Constructors.constructor_id)
        .where(
            Predictions.status == PredictionStatus.OPEN.value,
            Predictions.outcome_class == outcome_class.value,
        )
        .order_by(Predictions.driver_id, desc(Qualifyings.created_at))
        .distinct(Predictions.driver_id)
    )

    res = await db_sess.execute(query)
    preds = res.all()

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


@route.get("/model-config")
async def get_model_config(outcome_class: PredictionOutcome = PredictionOutcome.ALL):
    res = None
    folder = os.path.join(BASE_PATH, "model_development", "models")

    if outcome_class == PredictionOutcome.ALL:
        folder = os.path.join(folder, "model-all-2")
    elif outcome_class == PredictionOutcome.TOP3:
        folder = os.path.join(folder, "model-top3-2")
    elif outcome_class == PredictionOutcome.WINNER:
        folder = os.path.join(folder, "model-winner-2")

    fp = os.path.join(folder, "config.json")

    if os.path.exists(fp):
        async with aiofiles.open(fp, "rb") as f:
            data = await f.read()
            data = json.load(BytesIO(data))  # TODO: make awaitable
            res = F1ModelConfig(**data)
    return res
