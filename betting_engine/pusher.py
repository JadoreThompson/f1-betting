import asyncio

from collections import deque
from datetime import datetime
from r_mutex import LockClient
from sqlalchemy import update
from typing import Literal
from uuid import UUID

from config import ORDER_UPDATE_CHANNEL, REDIS_CLIENT
from db_models import Bets
from utils.db import get_db_session
from utils.utils import dump_obj


class Pusher:
    """
    Consolidates batched updates to order records and publishes them to a pub-sub
    channel. Supports both high-priority ("fast") and low-priority ("slow") updates,
    optimizing for throughput and latency based on urgency.

    Attributes:
        lock (LockClient): Mutex lock for safe concurrent access to shared resources.
        batch_size (int): Number of records processed per update cycle.
        _slow_queue (deque): Queue for non-urgent updates.
        _fast_queue (deque): Queue for urgent updates.
        _slow_delay (float): Delay interval (in seconds) for processing slow updates.
        _fast_delay (float): Delay interval (in seconds) for processing fast updates.
        _is_running (bool): Whether the Pusher has been started.
        _slow_running (bool): Status of the slow update loop.
        _fast_running (bool): Status of the fast update loop.
    """

    def __init__(
        self,
        lock: LockClient,
        slow_delay: float = 2.0,
        fast_delay: float = 0.8,
        batch_size: int = 100,
    ) -> None:
        """
        Initializes the Pusher instance.

        Args:
            lock (LockClient): A mutex client to control database access.
            slow_delay (float): Delay in seconds between slow updates.
            fast_delay (float): Delay in seconds between fast updates.
            batch_size (int): Number of records processed per batch.
        """
        self._lock = lock
        self.batch_size = batch_size
        self._slow_queue = deque()
        self._fast_queue = deque()
        self._slow_delay = slow_delay
        self._fast_delay = fast_delay
        self._slow_running: bool = False
        self._fast_running: bool = False

    async def start(self) -> None:
        """
        Starts the asynchronous loops for processing fast and slow queues.
        """
        asyncio.create_task(self._push_fast())
        asyncio.create_task(self._push_slow())
        await asyncio.sleep(2)    

    def append(
        self,
        obj: dict | list[dict],
        speed: Literal["slow", "fast"] = "fast",
    ) -> None:
        """
        Appends one or multiple update records to the appropriate queue.

        Args:
            obj (dict | list[dict]): Single record or list of records to update.
            speed (Literal["slow", "fast"]): Determines which queue to use.
        """
        queue = self._slow_queue if speed == "slow" else self._fast_queue

        if isinstance(obj, list):
            queue.extend(obj)
        else:
            queue.append(obj)

    def _get_batch(self, queue: deque) -> list[dict]:
        """
        Retrieves a batch of items from a given queue and parses fields to appropriate types.

        Args:
            queue (deque): Source queue.

        Returns:
            list[dict]: Batch of parsed update records.
        """
        ret_value: list[dict] = []
        obj: dict

        for _ in range(self.batch_size):
            try:
                obj = queue.popleft()
                obj_copy = obj.copy()

                obj_copy["user_id"] = UUID(obj["user_id"])
                obj_copy["order_id"] = UUID(obj["order_id"])
                obj_copy["created_at"] = datetime.fromisoformat(obj_copy["created_at"])

                ret_value.append(obj_copy)
            except IndexError:
                break

        return ret_value

    async def _push_slow(self) -> None:
        """
        Handles updates from the slow queue at a defined delay interval.

        Updates database records and publishes them to the Redis pub-sub channel.
        """
        self._slow_running = True

        while True:
            if self._slow_queue:
                collection = self._get_batch(self._slow_queue)

                async with self._lock:
                    async with get_db_session() as sess:
                        await sess.execute(update(Bets), collection)
                        await sess.commit()

                async with REDIS_CLIENT.pipeline() as pipe:
                    for item in collection:
                        await pipe.publish(ORDER_UPDATE_CHANNEL, dump_obj(item))
                    await pipe.execute()

            await asyncio.sleep(self._slow_delay)

    async def _push_fast(self) -> None:
        """
        Handles updates from the fast queue at a defined delay interval.

        Updates database records and publishes them to the Redis pub-sub channel.
        """
        self._fast_running = True

        while True:
            if self._fast_queue:
                collection = self._get_batch(self._fast_queue)

                async with self._lock:
                    async with get_db_session() as sess:
                        await sess.execute(update(Bets), collection)
                        await sess.commit()

                async with REDIS_CLIENT.pipeline() as pipe:
                    for item in collection:
                        await pipe.publish(ORDER_UPDATE_CHANNEL, dump_obj(item))
                    await pipe.execute()

            await asyncio.sleep(self._fast_delay)

    @property
    def is_running(self) -> bool:
        """
        Returns:
            bool: True if all queues are actively processing.
        """
        return self._fast_running and self._slow_running
