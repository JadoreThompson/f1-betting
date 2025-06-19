import asyncio
import os
import uvicorn
import requests

from datetime import datetime
from json import dump
from multiprocessing import Process, Queue
from r_mutex import LockClient, LockManager
from sqlalchemy import select

from betting_engine import (
    Topic,
    MatchingEngine,
    DataPoller,
    SettlementPoller,
    BetService,
)
from config import (
    LOCK_CHANNEL,
    POLLING_BASE_URL,
    REDIS_CLIENT,
    SERVER_DATA_FOLDER,
    SYNC_DB_ENGINE,
)
from db_models import Base, Markets
from enums import MarketCategory, MarketStatus, Side
from server import config
from utils.db import get_db_session


def run_bet_service(queue: Queue) -> None:
    asyncio.run(BetService(queue, LockClient(REDIS_CLIENT, LOCK_CHANNEL, False)).run())


def run_data_pipeline() -> None:
    """Runs the F1 data polling pipeline in a separate process.

    Initializes a DataPoller with a 10-second sleep duration and starts
    the continuous polling loop to fetch and persist F1 race data
    (qualifying, sprint, and grand prix results).

    This function is designed to be executed in a multiprocessing context
    and will run indefinitely until the process is terminated.
    """
    asyncio.run(DataPoller(15).poll())


async def settlement_pipeline(queue: Queue) -> None:
    """Processes race results and determines betting market settlements.

    Fetches the latest race results, retrieves all closed betting markets,
    and determines winners/losers for each market based on the race outcome.
    Settlement decisions are queued for processing by the matching engine.

    The function handles two market categories:
    - TOP3: Determines if a driver finished in the top 3 positions
    - WINNER: Determines if a driver won the race

    Args:
        queue (Queue): Multiprocessing queue for sending settlement data
                      to the matching engine.
    """
    async for race_data in SettlementPoller().poll():
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
    """Wrapper function to run the settlement pipeline in a separate process.

    Executes the async settlement_pipeline function using asyncio.run().
    This function serves as an entry point for multiprocessing execution.

    Args:
        queue (Queue): Multiprocessing queue for communication with the
                      matching engine process.
    """
    asyncio.run(settlement_pipeline(queue))


def run_engine(queue: Queue) -> None:
    """Runs the betting matching engine in a separate process.

    Initializes and starts the MatchingEngine which processes betting
    orders, matches back and lay bets, and handles settlement operations.
    The engine communicates via the provided queue.

    Args:
        queue (Queue): Multiprocessing queue for receiving messages from
                      other processes (settlements, orders, etc.).
    """
    engine = MatchingEngine(queue)
    asyncio.run(engine.run())


def run_server(queue: Queue) -> None:
    """Runs the FastAPI web server in a separate process.

    Configures and starts a uvicorn server to handle HTTP requests for
    the betting API. The server is configured to run on localhost:8000
    and uses the provided queue for communication with the matching engine.

    Args:
        queue (Queue): Multiprocessing queue for server-to-engine communication.
    """

    def fetch_shedule() -> None:
        r = requests.get(POLLING_BASE_URL + f"/{datetime.now().year}")
        d = r.json()
        dump(
            d["MRData"]["RaceTable"]["Races"],
            open(os.path.join(SERVER_DATA_FOLDER, "schedule.json"), "w"),
        )

    async def helper() -> None:
        """Inner async function to configure and start the uvicorn server."""
        config.MATCHING_ENGINE_QUEUE = queue

        from server.routes.bet import utils as bet_utils

        await bet_utils.lock.run()

        fetch_shedule()

        server_config = uvicorn.Config(
            "server.app:app",
            host="0.0.0.0",
            port=8000,
        )
        server = uvicorn.Server(server_config)
        await server.serve()

    # asyncio.run(helper())
    fetch_shedule()
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
    )


async def main() -> None:
    """Main orchestrator function that manages all application processes.

    Sets up and manages multiple processes for the F1 betting system:
    - Data pipeline: Polls F1 API for race data
    - Settlement pipeline: Processes race results for bet settlement
    - Matching engine: Handles bet matching and settlement execution
    - Web server: Provides HTTP API for betting operations

    The function implements automatic process restart logic - if any process
    dies unexpectedly, it will be automatically restarted. The main loop
    continues until interrupted by a keyboard interrupt or system signal.

    Process management features:
    - Automatic restart of failed processes
    - Graceful shutdown on interruption
    - Process health monitoring with 0.5s check interval

    Note: Currently only the data pipeline is enabled in the args tuple.
    Uncomment other processes as needed for full system operation.
    """
    matching_engine_queue = Queue()  # TODO: Change to async queue.
    lock_manager = LockManager(REDIS_CLIENT, LOCK_CHANNEL)

    args = (
        (run_settlement_pipeline, "settlement_pipeline", True),
        (run_data_pipeline, "data_pipeline", False),
        (run_engine, "matching_engine", True),
        (run_server, "server", True),
        (run_bet_service, "bet_service", True),
    )

    # TODO: Reduce quantity of processes.
    ps: list[Process] = [
        Process(target=f, args=(matching_engine_queue,) if use_queue else (), name=name)
        for f, name, use_queue in args
    ]

    # Initialising
    lm_task = asyncio.create_task(lock_manager.run())

    while not lock_manager.is_running:
        await asyncio.sleep(1)
        print("LockManager not running...")

    for p in ps:
        p.start()

    try:
        while True:
            for ind, p in enumerate(ps[:]):
                if not p.is_alive():
                    print(f"Process {p.name} has stopped unexpectedly. Restarting...")
                    p.terminate()
                    p.join()

                    f, name, use_queue = args[ind]
                    p = Process(
                        target=f,
                        args=(matching_engine_queue,) if use_queue else (),
                        name=name,
                    )
                    p.start()
                    ps[ind] = p

            await asyncio.sleep(0.1)
    except BaseException:
        import traceback

        traceback.print_exc()
        lm_task.cancel()

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
