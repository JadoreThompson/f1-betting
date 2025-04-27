import os
import numpy as np
import pandas as pd
import ydf
from typing import Tuple

BPATH = os.path.dirname(__file__)
DPATH = os.path.join(BPATH, "datasets")
LEARNER = ydf.GradientBoostedTreesLearner(label="positionText")


def apply_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply types to the dataset."""
    df["raceId"] = df["raceId"].astype("str")
    df["position"] = (
        pd.to_numeric(df["position"], errors="coerce").fillna(0).astype("int")
    )
    df["year"] = df["year"].astype("str")
    df["round"] = df["round"].astype("str")
    df["circuitId"] = df["circuitId"].astype("str")
    df["resultId"] = df["resultId"].astype("str")
    df["driverId"] = df["driverId"].astype("str")
    df["constructorId"] = df["constructorId"].astype("str")
    return df


def prepare_train_dataset(min_year: int = 2023) -> pd.DataFrame:
    races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))
    results_df = pd.read_csv(os.path.join(DPATH, "results.csv"))

    races_df = races_df[races_df["year"] >= min_year]
    df = races_df.merge(results_df, on=["raceId"], suffixes=("_race", "_result"))
    df = df[
        [
            "raceId",
            "resultId",
            "driverId",
            "year",
            "round",
            "circuitId",
            "date",
            "fp1_time",
            "fp2_time",
            "fp3_time",
            "quali_time",
            "sprint_time",
            "constructorId",
            "grid",
            "laps",
            "position",
            "positionText",
        ]
    ]

    return apply_types(df)


def add_lookback(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Add last n race results for each driver to the dataset."""
    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driverId")["positionText"].shift(i)

    df["raceId"] = df["raceId"].astype("int")
    df = df.sort_values("raceId")
    df = df.dropna()
    df["raceId"] = df["raceId"].astype("str")
    return df


def get_df(min_year: int = 2023, lookback: int = 5) -> pd.DataFrame:
    df = prepare_train_dataset(min_year)
    return add_lookback(df, lookback)


def train(
    df: pd.DataFrame, year: int = 2024
) -> Tuple[np.ndarray, ydf.GradientBoostedTreesLearner]:
    """Train the model with the given dataset.

    Args:
        df (pd.DataFrame): dataset
        year (int, optional): Defaults to 2024. Used to split the dataset into train and test sets.
    """
    global LEARNER

    df["year"] = df["year"].astype("int")
    train_df = df[df["year"] <= year]
    test_df = df[df["year"] > year]
    df["year"] = df["year"].astype("str")
    model = LEARNER.train(train_df)

    return model.predict(test_df), model


def save_model(model) -> None:
    fpath = os.path.join(BPATH, "models")

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    model.save(os.path.join(fpath, "model_1"))


def load_model() -> None:
    """Load and make predictions with the model"""
    model = ydf.load_model(os.path.join(BPATH, "models", "model_1"))
    df = pd.read_csv(os.path.join(DPATH, "model_1_dataset.csv"))
    df = apply_types(df)

    res = model.predict(
        pd.DataFrame(df[(df["raceId"] == "1143") & (df["driverId"] == "815")])
    )

    ind = res[0].tolist().index(max(res[0]))
    print(model.label_classes())
    print(model.label_classes()[ind])


def run_train() -> None:
    df = get_df(2020)
    _, model = train(df, 2023)
    save_model(model)
    df.to_csv(os.path.join(DPATH, "model_1_dataset.csv"), index=False)


def main() -> None:
    # run_train()
    load_model()


if __name__ == "__main__":
    main()
