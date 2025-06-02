import json
from typing import Any, Optional
from aiohttp import ClientSession
from asyncio import sleep
from config import POLLING_BASE_URL
from datetime import datetime, UTC
from .base_poller import BasePoller, TargetNotFound


class SettlementPoller(BasePoller):
    """Poller class to manage the issuing of funds
    to the winners and losers of a market.
    """

    def __init__(self, sleep_duration: int = 5) -> None:
        super().__init__(sleep_duration)

    async def poll(self) -> list[dict[str, Any]]:
        """Polls jolpica to check if the grand prix has ended.

        Args:
            sleep_duration (int, optional): Duration to sleep between calls to
                the api. Defaults to 5.

        Raises:
            TargetNotFound: The next round in the grand prix couldn't be found.
            Exception: The response to the api endpoint resulted in a status code
                other than 200.

        Returns:
            list[dict[str, Any]]: The race data for each participant driver
                withib the grand prix.
        """
        async with ClientSession() as sess:
            schedule = await super()._fetch_schedule(sess)
            json.dump(schedule, open("jolpica_schedule.json", "w"))

            target_data = self._get_target(schedule)

            if target_data is None:
                raise TargetNotFound

            round_number, round_time = target_data
            time_left = round_time.timestamp() - datetime.now(UTC).timestamp()
            print(
                f"Target round: {round_number}, time left until start: {time_left:.2f} seconds."
            )

            # await sleep(time_left) # Commented out for testing purposes.
            print(f"Finished Sleeping")

            endpoint = POLLING_BASE_URL + f"/{datetime.now().date().year}/{2}/sprint"
            print("Polling endpoint: {endpoint}")

            while True:
                async with sess.get(endpoint) as rsp:
                    if rsp.status != 200:
                        raise Exception(f"{endpoint} threw status code: {rsp.status}")

                    data = await rsp.json()
                    # json.dump(data, open("race_data.json", "w"))

                    if race_data := data["MRData"]["RaceTable"]["Races"][0]["Results"]:
                        print("Race results found!")
                        return race_data

                    print(
                        f"No results yet. Sleeping for {self._sleep_duration} seconds..."
                    )
                    await sleep(self._sleep_duration)

    def _get_target(self, data: dict[str, Any]) -> Optional[tuple[int, datetime]]:
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
