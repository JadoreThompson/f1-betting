from sqlalchemy import select
from db_models import Drivers, F1Data
from engine import interact, Prediction
from server.utils.db import get_db_session


# BASE_PATH = "https://ergast.com/api/f1"
# NS = {"mrd": "http://ergast.com/mrd/1.5"}


# def parse_standings(xml_string: str) -> Tuple[list[Driver], str]:
#     root = ET.fromstring(xml_string)
#     results: list[Driver] = []

#     for result in root.findall(".//mrd:Driver", NS):
#         results.append(
#             Driver(
#                 name=result.attrib["driverId"],
#                 number=result.findtext("mrd:PermanentNumber", namespaces=NS),
#             )
#         )

#     return results, root.find(".//mrd:StandingsList", NS).attrib["round"]


# async def fetch_standings(
#     year: int | None = None, round_: int | None = None
# ) -> Tuple[list[Driver], str]:
#     if year is None:
#         year = datetime.now().year

#     async with AsyncClient() as client:
#         rsp = await client.get(
#             BASE_PATH
#             + f"/{year}{f"/{round_}" if round_ is not None else ""}/driverStandings"
#         )
#         if rsp.status_code != 200:
#             raise ValueError(f"Request failed with status code: {rsp.status_code}")
#         return parse_standings(rsp.text)


async def get_predictions() -> tuple[int, str, str, float]:

    async with get_db_session() as sess:
        res = await sess.execute(
            select(
                F1Data.driver_id,
                Drivers.name,
                F1Data.last_1,
                F1Data.last_2,
                F1Data.last_3,
                F1Data.last_4,
                F1Data.last_5,
            ).join(Drivers, F1Data.driver_id == Drivers.driver_id)
        )

        last_races = res.all()

    predictions: tuple[Prediction, ...] = interact([l[2:] for l in last_races])

    return tuple(
        (
            last_races[i][0],
            last_races[i][1],
            predictions[i].prediction,
            predictions[i].percentage,
        )
        for i in range(len(last_races))
    )
