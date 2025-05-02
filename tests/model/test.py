import asyncio
import json

from engine import interact, get_position_category
from .typing import (
    ConstructedRaceData,
    Dataset,
    ParsedQualiData,
)
from .utils import (
    build_last_races_dataset,
    build_qualifying_dataset,
    get_avg_position_move,
    get_rolling_pos_move,
    get_sma,
)

API_VERSION = "v2"
POS_CAT = "loose"
INTERACT_TYPE = "multi"
YEARS = [2024]
LAST_N_RACES = 2
ROUNDS = 24


def get_features(
    *,
    last_n_data: Dataset[ConstructedRaceData],
    quali_data: Dataset[ParsedQualiData],
    LAST_N_RACES: int,
    round: int,
    name: str,
) -> list:
    """
    Returns list of features to be inputted into the model.

    Args:
        last_n_data (Dataset[ConstructedRaceData])
        quali_data (Dataset[ParsedQualiData])
        LAST_N_RACES (int)
        round (int)
        name (str)

    Returns:
        list: A list of features for a given driver of a given round.
    """
    dataset = []
    specific_last_n_data = last_n_data[round].get(
        name,
        ConstructedRaceData.construct(LAST_N_RACES, name),
    )
    specific_quali_data = quali_data[round].get(name, ParsedQualiData())

    dataset.append(int(specific_last_n_data.grid))
    # dataset.append(get_position_category(specific_quali_data.position, "loose"))
    # dataset.append(int(specific_quali_data.position))
    # dataset.append(get_sma(specific_last_n_data, LAST_N_RACES))
    # dataset.append(
    #     min(
    #         [
    #             specific_quali_data.q3_secs,
    #             specific_quali_data.q2_secs,
    #             specific_quali_data.q1_secs,
    #         ]
    #     )
    # )
    # dataset.append(specific_quali_data.q3_secs)
    # dataset.append(specific_quali_data.q2_secs)
    # dataset.append(specific_quali_data.q1_secs)
    # quali_t = int
    # if specific_quali_data:
    #     dataset.append(quali_t(specific_quali_data.position))
    # else:
    #     dataset.append(quali_t(0))
    # dataset.extend(specific_last_n_data.prev_positions)
    # dataset.extend(
    #     [float(p) if p.isdigit() else 0.0 for p in specific_last_n_data.prev_positions]
    # )
    dataset.extend(
        [get_position_category(p, "loose") for p in specific_last_n_data.prev_positions]
    )
    # dataset.append(get_avg_position_move(last_n_data, name))
    # dataset.append(get_rolling_pos_move(last_n_data, 3, round, name))
    # dataset.append(specific_data.constructor_name)

    return dataset


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
            total += 1
            dataset = get_features(
                last_n_data=last_n_data,
                quali_data=quali_data,
                LAST_N_RACES=LAST_N_RACES,
                round=r,
                name=d,
            )
            driver_data = last_n_data[r].get(
                d, ConstructedRaceData.construct(LAST_N_RACES, d)
            )

            # Handling prediction
            pred = interact([dataset], INTERACT_TYPE)[0]

            if False:
                print(
                    "Pred:",
                    pred.prediction,
                    " Actual:",
                    get_position_category(driver_data.real, POS_CAT),
                )

            if pred.prediction == get_position_category(driver_data.real, POS_CAT):
                success += 1

        if success and total:
            success /= total

        results.append(success)

        print(f"Driver: {d}, Success Rate: {success:.2%}")

    print(f"\nAverage Success: {sum(results) / len(results) if results else 0:.2%}")
    print(
        f"Median Success: {list(sorted(results))[len(results) // 2] if results else 0:.2%}"
    )


async def test_predictions() -> None:
    """Evaluates models predictions across a season."""
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

    for r in last_n_data:
        success = 0

        for name in last_n_data[r]:
            dataset = get_features(
                last_n_data=last_n_data,
                quali_data=quali_data,
                LAST_N_RACES=LAST_N_RACES,
                round=r,
                name=name,
            )

            driver_data = last_n_data[r][name]

            # Handling prediction
            pred = interact([dataset], INTERACT_TYPE)[0]

            if False:
                print(
                    "Pred:",
                    pred.prediction,
                    " Actual:",
                    get_position_category(driver_data.real, POS_CAT),
                )

            if pred.prediction == get_position_category(driver_data.real, POS_CAT):
                success += 1

        if success:
            success /= len(last_n_data[r])

        results.append(success)

        print(f"Round: {r}, Success Rate: {success:.2%}")

    print(f"\nAverage Success: {sum(results) / len(results) if results else 0:.2%}")
    print(
        f"Median Success: {list(sorted(results))[len(results) // 2] if results else 0:.2%}"
    )


if __name__ == "__main__":
    asyncio.run(test_predictions())
    # asyncio.run(test_driver_predictions())
