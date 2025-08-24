import os
from datetime import datetime
from json import load

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config import DATA_FOLDER
from server.dependencies import depends_db_session
from utils.utils import get_datetime


route = APIRouter(prefix="")

fpath = os.path.join(DATA_FOLDER, "schedule.json")

if os.path.exists(fpath):
    SEASON_SCHEDULE: list[dict] = load(open(fpath, "rb"))
else:
    SEASON_SCHEDULE = []    

upcoming_round: int | None = None


@route.get("/predictions")
async def get_predictions(db_sess: AsyncSession = Depends(depends_db_session)):
    dt = get_datetime()
    year = dt.year

    if upcoming_round is None:
        for race in SEASON_SCHEDULE:
            if datetime.fromisoformat(race["date"]) > dt:
                upcoming_round = int(race["round"])

    await db_sess.scalars()
