import asyncio
import os
import uvicorn
import requests

from datetime import datetime
from json import dump
from multiprocessing import Process, Queue
from sqlalchemy import select


from config import (
    LOCK_CHANNEL,
    API_BASE_URL,
    REDIS_CLIENT,
    SERVER_DATA_FOLDER,
    SYNC_DB_ENGINE,
)
from db_models import Base, Markets
from enums import PredictionCategory, MarketStatus, Side
from services import SessionPoller
from utils.db import get_db_session


def run_data_pipeline() -> None:
    asyncio.run(SessionPoller.poll())


def run_server() -> None:
    r = requests.get(API_BASE_URL + f"/{datetime.now().year}")
    d = r.json()
    dump(
        d["MRData"]["RaceTable"]["Races"],
        open(os.path.join(SERVER_DATA_FOLDER, "schedule.json"), "w"),
    )

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


async def main() -> None:
    args = (
        (run_data_pipeline, (), {}, "data_pipeline"),
        (run_server, (), {}, "server"),
    )

    ps: list[Process] = [
        Process(target=f, args=args, kwargs=kwargs, name=name)
        for f, args, kwargs, name in args
    ]

    for p in ps:
        p.start()

    try:
        while True:
            for ind, p in enumerate(ps[:]):
                if not p.is_alive():
                    print(f"Process {p.name} has stopped unexpectedly. Restarting...")
                    p.terminate()
                    p.join()

                    f, args, kw, name = args[ind]
                    p = Process(target=f, args=args, kwargs=kw, name=name)
                    p.start()
                    ps[ind] = p

            await asyncio.sleep(0.1)
    except BaseException:
        import traceback

        traceback.print_exc()

        print("Shutting down...")
        for p in ps:
            p.terminate()
            p.join()
        print("Shutdown complete.")


def wrapped_main() -> None:
    try:
        Base.metadata.create_all(bind=SYNC_DB_ENGINE)
        asyncio.run(main())
    finally:
        Base.metadata.drop_all(bind=SYNC_DB_ENGINE)


if __name__ == "__main__":
    asyncio.run(main())
