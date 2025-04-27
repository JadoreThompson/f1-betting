import numpy as np
import pandas as pd
import ydf

races_df = pd.read_csv("./datasets/races.csv")
results_df = pd.read_csv("./datasets/results.csv")


def prepare_dataset() -> pd.DataFrame:
    global races_df, results_df

    races_df = races_df[races_df["year"] >= 2023]
    df = races_df.merge(results_df, on=["raceId"], suffixes=("_race", "_result"))
    df = df[
        [
            "raceId",
            "year",
            "round",
            "circuitId",
            "date",
            "fp1_time",
            "fp2_time",
            "fp3_time",
            "quali_time",
            "sprint_time",
            "resultId",
            "driverId",
            "constructorId",
            "grid",
            "positionText",
            "laps",
        ]
    ]

    df["raceId"] = df["raceId"].astype("str")
    df["year"] = df["year"].astype("str")
    df["round"] = df["round"].astype("str")
    df["circuitId"] = df["circuitId"].astype("str")
    df["resultId"] = df["resultId"].astype("str")
    df["driverId"] = df["driverId"].astype("str")
    df["constructorId"] = df["constructorId"].astype("str")

    return df


def add_lookback(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Add last n race results for each driver to the dataset."""
    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driverId")["positionText"].shift(i)

    df["raceId"] = df["raceId"].astype("int")
    df = df.sort_values("raceId")
    df = df.dropna()
    df["raceId"] = df["raceId"].astype("str")
    return df


def train(df: pd.DataFrame, year: int = 2024) -> np.ndarray:
    """Train the model with the given dataset.

    Args:
        df (pd.DataFrame): dataset
        year (int, optional): Defaults to 2024. Used to split the dataset into train and test sets.
    """
    # Placeholder for training logic
    train_df = df[df["year"] < str(year)]
    test_df = df[df["year"] == str(year)]

    model = ydf.GradientBoostedTreesLearner(label="positionText").train(train_df)
    return model.predict(test_df)


if __name__ == "__main__":
    df = prepare_dataset()
    df = add_lookback(df, 5)
    results = train(df)
    print("Results: ", results)

    # df.to_csv("./datasets/processed_dataset.csv", index=False)
