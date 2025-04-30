import asyncio

from engine import interact, get_position_category
from .typing import (
    ConstructedRaceData,
    Dataset,
    ParsedQualiData,
    default_constructed_race_data,
)
from .utils import (
    build_last_races_dataset,
    build_qualifying_dataset,
    get_avg_position_move,
)

POS_CAT = "loose"
INTERACT_TYPE = "multi"


def get_row(
    *,
    last_n_data: Dataset[ConstructedRaceData],
    quali_data: Dataset[ParsedQualiData],
    last_n_races: int,
    round: int,
    name: str,
) -> list:
    dataset = []
    specific_data = last_n_data[round].get(
        name, default_constructed_race_data(last_n_races, name)
    )

    sma_value = sum(int(p) if p.isdigit() else 0 for p in specific_data.prev_positions)

    if sma_value:
        sma_value /= last_n_races

    dataset.append(int(specific_data.grid))

    quali_t = int
    if quali := quali_data[round].get(name):
        dataset.append(quali_t(quali.position))
    else:
        dataset.append(quali_t(0))

    dataset.append(sma_value)
    # dataset.extend(specific_data.prev_positions)
    dataset.append(get_avg_position_move(last_n_data)[name])
    # dataset.append(specific_data.constructor_name)
    # dataset.extend(
    #     [get_position_category(p) for p in driver_data.prev_positions]
    # )

    return dataset


async def test_driver_predictions():
    year = 2024
    last_n_races = 5
    rounds = 20

    last_n_data: Dataset[ConstructedRaceData] = await build_last_races_dataset(
        year, rounds=rounds, last_n_races=last_n_races
    )
    quali_data: Dataset[ParsedQualiData] = await build_qualifying_dataset(
        year, last_n_races, rounds
    )

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

        for r in last_n_data:
            dataset = get_row(
                last_n_data=last_n_data,
                quali_data=quali_data,
                last_n_races=last_n_races,
                round=r,
                name=d,
            )
            driver_data = last_n_data[r].get(
                d, default_constructed_race_data(last_n_races, d)
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

        if success:
            success /= len(last_n_data)

        results.append(success)

        print(f"Driver: {d}, Success Rate: {success:.2%}")

    print(f"\nAverage Success: {sum(results) / len(results) if results else 0:.2%}")
    print(
        f"Median Success: {list(sorted(results))[len(results) // 2] if results else 0:.2%}"
    )


async def test_predictions() -> None:
    year = 2024
    last_n_races = 5
    rounds = 20

    last_n_data: Dataset[ConstructedRaceData] = await build_last_races_dataset(
        year, rounds, last_n_races
    )
    quali_data: Dataset[ParsedQualiData] = await build_qualifying_dataset(
        year, last_n_races, rounds
    )

    results: list[float] = []

    if False:
        for r in list(last_n_data.keys()):
            for n in list(last_n_data[r].keys()):
                if last_n_data[r][n].real != "1":
                    last_n_data[r].pop(n)

    for r in last_n_data:
        success = 0

        for name in last_n_data[r]:
            dataset = get_row(
                last_n_data=last_n_data,
                quali_data=quali_data,
                last_n_races=last_n_races,
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
