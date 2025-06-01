import asyncio
import time
import uvicorn

from multiprocessing import Process, Queue
from betting_engine.matching_engine import MatchingEngine
from server import config


def run_engine(queue: Queue) -> None:
    engine = MatchingEngine(queue)
    engine.run()


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

    args = ((run_server, "server"), (run_engine, "matching_engine"))

    ps: tuple[Process, ...] = tuple(
        Process(target=f, args=(matching_engine_queue,), name=name) for f, name in args
    )

    for p in ps:
        p.start()

    try:
        while True:
            for ind, p in enumerate(ps[::]):
                if not p.is_alive():
                    print(f"Process {p.name} has stopped unexpectedly. Restarting...")
                    p.terminate()
                    p.join()

                    p = Process(
                        target=args[ind], args=matching_engine_queue, name=args[ind][1]
                    )
                    p.start()
                    ps[ind] = p

            time.sleep(2)
    except BaseException:
        print("Shutting down...")
        for p in ps:
            p.terminate()
            p.join()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
