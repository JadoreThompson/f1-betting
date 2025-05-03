import asyncio
import json

from engine import interact
from engine.utils import get_position_category
from .typing import (
    ConstructedRaceData,
    Dataset,
    ParsedQualiData,
)
from .utils import (
    build_last_races_dataset,
    build_qualifying_dataset,
    get_avg_position,
    get_avg_position_move,
    get_rolling_pos_move,
    get_sma,
    get_std,
    get_features
)

API_VERSION = "v2"
POS_CAT = "loose"
INTERACT_TYPE = "multi"
YEARS = [2024]
LAST_N_RACES = 2
ROUNDS = 24


async def test_driver_predictions():
    """
    Evaluates models predictions on a per driver basis.
    """
    last_n_data, quali_data = await asyncio.gather(
        build_last_races_dataset(ROUNDS, LAST_N_RACES, YEARS, version=API_VERSION),
        build_qualifying_dataset(ROUNDS, YEARS, version=API_VERSION),
    )

    # last_n_data: Dataset[ConstructedRaceData] = await build_last_races_dataset(
    #     ROUNDS, LAST_N_RACES, YEARS, version=API_VERSION
    # )

    # quali_data: Dataset[ParsedQualiData] = await build_qualifying_dataset(
    #     ROUNDS, YEARS, version=API_VERSION
    # )

    drivers = []
    for r in last_n_data:
        drivers.extend(list(last_n_data[r].keys()))

    drivers = set(drivers)
    results: list[float] = []

    if False:
        for r in list(last_n_data.keys()):
            for n in list(last_n_data[r].keys()):
                if last_n_data[r][n].real != "1":
                    last_n_data[r].pop(n)

    for d in drivers:
        success = 0
        total = 0

        for r in last_n_data:
            if d not in last_n_data[r]:
                continue

            # total += 1

            dataset = get_features(
                race_data=last_n_data,
                quali_data=quali_data,
                last_n_races=LAST_N_RACES,
                round_=r,
                name=d,
            )

            # driver_data = last_n_data[r].get(
            #     d, ConstructedRaceData.construct(LAST_N_RACES, d)
            # )
            driver_data = last_n_data[r][d]

            # Handling prediction
            pred = interact([dataset])[0]

            if False:
                print(
                    "Pred:",
                    pred.prediction,
                    " Actual:",
                    get_position_category(driver_data.real, POS_CAT),
                )

            # if pred.prediction == get_position_category(driver_data.real, POS_CAT):
            #     success += 1

            if (
                "1" <= pred.prediction < "3"
                or "1" <= get_position_category(driver_data.real, POS_CAT) < "3"
            ):
                total += 1
                if pred.prediction == get_position_category(driver_data.real, POS_CAT):
                    success += 1

        # print("Total:", total, " Success:", success)
        if success and total:
            success /= total

        if success or total:
            results.append(success)

            # print(f"Driver: {d}, Success Rate: {success:.2%}")
            print(
                f"Driver: {r}, Success Rate: {success:.2%}, Success count: {success}, Total count: {total}"
            )

    print(f"\nAverage Success: {sum(results) / len(results) if results else 0:.2%}")
    print(
        f"Median Success: {list(sorted(results))[len(results) // 2] if results else 0:.2%}"
    )


async def test_predictions(log: bool = True) -> tuple[float, float]:
    """Evaluates models predictions across a season.

    Returns:
        tuple[float, float]: (median success rate, avg succes rate)
    """
    last_n_data, quali_data = await asyncio.gather(
        build_last_races_dataset(ROUNDS, LAST_N_RACES, YEARS, version=API_VERSION),
        build_qualifying_dataset(ROUNDS, YEARS, version=API_VERSION),
    )

    # last_n_data: Dataset[ConstructedRaceData] = await build_last_races_dataset(
    #     ROUNDS, LAST_N_RACES, YEARS, version=API_VERSION
    # )

    # quali_data: Dataset[ParsedQualiData] = await build_qualifying_dataset(
    #     ROUNDS, years, version=API_VERSION
    # )

    results: list[float] = []

    if False:
        for r in list(last_n_data.keys()):
            for n in list(last_n_data[r].keys()):
                if last_n_data[r][n].real != "1":
                    last_n_data[r].pop(n)

    msg = ""
    for r in last_n_data:
        success = 0
        total = 0

        for name in last_n_data[r]:
            dataset = get_features(
                race_data=last_n_data,
                quali_data=quali_data,
                last_n_races=LAST_N_RACES,
                round_=r,
                name=name,
            )

            driver_data = last_n_data[r][name]

            # Handling prediction
            pred = interact([dataset])[0]

            if False:
                print(
                    "Pred:",
                    pred.prediction,
                    " Actual:",
                    get_position_category(driver_data.real, POS_CAT),
                )

            # if pred.prediction == get_position_category(driver_data.real, POS_CAT):
            #     success += 1

            if (
                "1" <= pred.prediction < "3"
                or "1" <= get_position_category(driver_data.real, POS_CAT) < "3"
            ):
                total += 1
                if pred.prediction == get_position_category(driver_data.real, POS_CAT):
                    success += 1

        if success:
            # success /= len(last_n_data[r])
            success /= total

        # results.append(success)

        if success or total:
            results.append(success)
            if log:
                msg += f"\nRound: {r}, Success Rate: {success:.2%}, Success count: {success}, Total: {total}"
            # print(
            #     f"Round: {r}, Success Rate: {success:.2%}, Success count: {success}, Total: {total}"
            # )
        # print(f"Round: {r}, Success Rate: {success:.2%}, Success count: {success}")
    if log:
        print(msg)
        print(f"\nAverage Success: {sum(results) / len(results) if results else 0:.2%}")
        print(
            f"Median Success: {list(sorted(results))[len(results) // 2] if results else 0:.2%}"
        )

    if results:
        return sum(results) / len(results), list(sorted(results))[len(results) // 2]
    return 0, 0


if __name__ == "__main__":
    asyncio.run(test_predictions())
    # asyncio.run(test_driver_predictions())
