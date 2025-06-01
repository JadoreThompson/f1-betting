from aiohttp import ClientSession
from asyncio import sleep
from config import POLLING_BASE_URL
from datetime import datetime, UTC
from typing import Any, Optional


class Poller:
    """Poller class to manage the issuing of funds
    to the winners and losers of a market.
    """

    @classmethod
    async def poll(cls, sleep_duration: int = 5) -> bool:
        async with ClientSession() as sess:
            schedule = await cls._fetch_schedule(sess)

            if schedule is None:
                raise Exception("No target round was found. Season possibly over.")

            round_number, round_time = cls._get_target_round(schedule)
            time_left = round_time.timestamp() - datetime.now(UTC).timestamp()
            print(
                f"Target round: {round_number}, time left until start: {time_left:.2f} seconds."
            )

            await sleep(time_left)
            print(f"Finished Sleeping")

            endpoint = (
                POLLING_BASE_URL
                + f"/{datetime.now().date().year}/{round_number}/results"
            )
            print("Polling endpoint: {endpoint}")

            while True:
                async with sess.get(endpoint) as rsp:
                    if rsp.status != 200:
                        raise Exception(f"{endpoint} threw status code: {rsp.status}")

                    data = await rsp.json()

                    if data["MRData"]["RaceTable"]["Races"]:
                        print("Race results found!")
                        return True

                    print(f"No results yet. Sleeping for {sleep_duration} seconds...")
                    await sleep(sleep_duration)

    @classmethod
    async def _fetch_schedule(cls, session: ClientSession) -> dict[str, Any]:
        async with session.get(POLLING_BASE_URL + f"/current") as rsp:
            if rsp.status != 200:
                raise Exception("Error fetching season schedule.")
            return await rsp.json()

    @classmethod
    def _get_target_round(cls, data: dict[str, Any]) -> Optional[tuple[int, datetime]]:
        """Returns the next round coming up and the time it starts

        Args:
            data (dict[str, Any]): Jolpica race schedule data.

        Returns:
            Optional[tuple[int, datetime]]:
                - int: Next round.
                - datetime: Datetime of next round's grand prix race.
        """
        cur_datetime = datetime.now(UTC)

        for d in data["MRData"]["RaceTable"]["Races"]:
            round_datetime = datetime.fromisoformat(f"{d["date"]}T{d["time"]}")
            if round_datetime > cur_datetime:
                return (int(d["round"]), round_datetime)
