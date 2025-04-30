import asyncio
import json
import pandas as pd

from engine import interact, evaluate, get_position_category
from engine.config import TRAINED_MODEL_FEATURES

# from .models import ConstructedRaceData
from .typing import ConstructedRaceData, Dataset, ParsedQualiData
from .utils import (
    build_last_races_dataset,
    build_qualifying_dataset,
    fetch_quali_results,
    fetch_race_results,
    # merge_last_n_races_and_quali,
)


async def test_predictions(
    year: int = 2024, rounds: int = 20, last_n_races: int = 5
) -> None:
    last_n_data: Dataset[ConstructedRaceData] = await build_last_races_dataset(
        year, rounds=rounds, last_n_races=last_n_races
    )

    quali_data: Dataset[ParsedQualiData] = await build_qualifying_dataset(
        year, last_n_races, rounds
    )

    # print(quali_data)

    # grid
    # print(last_n_data)
    # json.dump(last_n_data, open("file.json", "w"), indent=4)

    # if True:
    #     return

    results: list[float] = []
    for r in last_n_data:
        success = 0

        for name in last_n_data[r]:
            dataset: list[float] = []
            driver_data = last_n_data[r][name]

            sma_value = sum(
                int(p) if p.isdigit() else 0 for p in driver_data.prev_positions
            )

            if sma_value:
                sma_value /= last_n_races

            # Constructing dataset
            # dataset.append(int(driver_data.grid))

            # if quali := quali_data[r].get(name):
            #     dataset.append(int(quali.position))
            # else:
            #     dataset.append(0)

            # dataset.append(driver_data.constructor_name)
            # dataset.append(sma_value)
            dataset.extend(driver_data.prev_positions)
            # dataset.extend(
            #     [get_position_category(p) for p in driver_data.prev_positions]
            # )

            # Handling prediction
            pred = interact([dataset])[0]

            if pred.prediction == get_position_category(driver_data.real):
                success += 1

        success_rate = success / len(last_n_data[r])
        results.append(success_rate)

        print(f"Round: {r}, Success Rate: {success_rate if success else 0:.2%}")

    print(f"\nAverage Success: {sum(results) / len(results) if results else 0:.2%}")
    print(
        f"Median Success: {list(sorted(results))[len(results) // 2] if results else 0:.2%}"
    )


# async def test_predictions_fixed(
#     year: int = 2024, rounds: int = 20, last_n_races: int = 5
# ) -> None:
#     """
#     Fix the external test to properly match the format expected by the model
#     """
#     last_n_data = await build_last_races_dataset(
#         year, rounds=rounds, last_n_races=last_n_races
#     )
#     print(f"Model features: {TRAINED_MODEL_FEATURES}")

#     for r in last_n_data:
#         success = 0
#         total = 0

#         for driver_id in last_n_data[r]:
#             row_data = {}

#             past_positions = [
#                 int(p) if p.isdigit() else 0
#                 for p in last_n_data[r][driver_id]["data"][:4]
#             ]
#             sma_value = (
#                 sum(past_positions) / len(past_positions) if past_positions else 0
#             )

#             row_data["sma"] = sma_value
#             test_df = pd.DataFrame([row_data], columns=TRAINED_MODEL_FEATURES)

#             pred = interact(test_df)[0]
#             actual = last_n_data[r][driver_id]["real"]

#             if pred.prediction == actual:
#                 success += 1

#             total += 1
#             # print(f"Driver: {driver_id}, Predicted: {pred.prediction}, Actual: {actual}")

#         print(f"Round: {r}, Success Rate: {success / total if total else 0:.2%}")


def test_json():
    data = json.load(open("file.json", "r"))

    for r in data:
        success = 0
        total = 0

        for driver_id in data[r]:
            row_data = {}

            past_positions = [
                int(p) if p.isdigit() else 0 for p in data[r][driver_id]["data"][:4]
            ]
            sma_value = (
                sum(past_positions) / len(past_positions) if past_positions else 0
            )

            row_data["sma"] = sma_value
            row_data["positionText"] = data[r][driver_id]["real"]
            test_df = pd.DataFrame([row_data], columns=["sma", "positionText"])
            # print(test_df)
            evaluate(df=test_df)


if __name__ == "__main__":
    # test_json()
    # asyncio.run(test_predictions_fixed(2023, last_n_races=4))
    asyncio.run(test_predictions(2024, last_n_races=3))
    # asyncio.run(fetch_quali_results())
