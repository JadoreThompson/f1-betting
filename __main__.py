import asyncio
import time
import uvicorn

from typing import Callable
from sqlalchemy import select
from multiprocessing import Process, Queue

from betting_engine import Poller, Topic, MatchingEngine
from db_models import Markets
from enums import MarketCategory, MarketStatus, Side
from server import config
from utils.db import get_db_session


async def pipeline(queue: Queue) -> None:
    race_data = await Poller.poll()

    async with get_db_session() as s:
        res = await s.execute(
            select(Markets.market_id, Markets.category, Markets.title).where(
                Markets.market_status == MarketStatus.CLOSED.value
            )
        )
        markets = res.all()

    top3_drivers: tuple[str, ...] = tuple(
        r["Driver"]["driverId"] for r in race_data if int(r["position"]) < 4
    )
    winner: tuple[str] = tuple(
        r["Driver"]["driverId"] for r in race_data if int(r["position"]) == 1
    )

    print("Top 3 Drivers: ", *top3_drivers)
    print("Winner", *winner)
    print(markets)

    for m in markets:
        if m[1] == MarketCategory.TOP3.value:
            if m[2] in top3_drivers:
                winners = Side.BACK
            else:
                winners = Side.LAY
        else:
            if m[2] in winner:
                winners = Side.BACK
            else:
                winners = Side.LAY

        queue.put(
            {
                "topic": Topic.SETTLE,
                "settle_data": {"winners": winners, "market_id": m[0]},
            }
        )


def run_settlement_pipeline(queue: Queue) -> None:
    asyncio.run(pipeline(queue))


def run_engine(queue: Queue) -> None:
    engine = MatchingEngine(queue)
    asyncio.run(engine.run())


def run_server(queue: Queue) -> None:
    async def helper() -> None:
        config.MATCHING_ENGINE_QUEUE = queue
        server_config = uvicorn.Config(
            "server.app:app", host="localhost", port=8000, reload=True
        )
        server = uvicorn.Server(server_config)
        await server.serve()

    asyncio.run(helper())


def main() -> None:
    matching_engine_queue = Queue()

    args: tuple[Callable[[Queue], None], ...] = (
        (run_server, "server"),
        (run_engine, "matching_engine"),
        (run_settlement_pipeline, "settlement_pipeline"),
    )

    ps: tuple[Process, ...] = [
        Process(target=f, args=(matching_engine_queue,), name=name) for f, name in args
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

                    p = Process(
                        target=args[ind][0],
                        args=(matching_engine_queue,),
                        name=args[ind][1],
                    )
                    p.start()
                    ps[ind] = p

            time.sleep(2)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("Shutting down...")
        for p in ps:
            p.terminate()
            p.join()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
