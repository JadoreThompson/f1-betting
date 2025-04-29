import os
import numpy as np
import pandas as pd

from typing import Tuple, TypedDict
from .config import BPATH, DPATH, LEARNER_TYPE, MPATH
from .utils import prepare_train_dataset, sma

TARGET_LABEL = "position_text"
LEARNER: LEARNER_TYPE = LEARNER_TYPE(label=TARGET_LABEL)


class ResultData(TypedDict):
    last_n_races: int
    sma: int
    size: int  # num races in dataset


Results = dict[float, ResultData]  # {success_rate: {metadata}}


def get_df(
    min_year: int = 2023, last_n_races: int = None, sma_lookback: int = None
) -> pd.DataFrame:
    df = prepare_train_dataset(min_year)

    if last_n_races:
        df = add_last_races(df, last_n_races)

    if sma_lookback:
        df["sma"] = df.apply(lambda x: sma(x, df, "position", sma_lookback), axis=1)

    return df.drop(["position", "race_id"], axis=1)


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        ["position", "race_id", "driver_id", "year"],
        axis=1,
    )


def add_last_races(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Add last n race results for each driver to the dataset."""
    for i in range(1, lookback + 1):
        df[f"last_{i}"] = df.groupby("driver_id")["position_text"].shift(i)

    df["race_id"] = df["race_id"].astype("int")
    df = df.sort_values("race_id")
    df["race_id"] = df["race_id"].astype("str")
    return df


def save_model(model, name: str) -> None:
    fpath = os.path.join(MPATH, name)

    if not os.path.exists(MPATH):
        os.makedirs(MPATH)

    if os.path.exists(fpath):
        os.remove(fpath)

    model.save(fpath)


def train(
    df: pd.DataFrame,
    year: int = 2024,
) -> Tuple[np.ndarray, LEARNER_TYPE, pd.DataFrame]:
    """Train the model with the given dataset.

    Args:
        df (pd.DataFrame): dataset
        year (int, optional): Defaults to 2024. Used to split the dataset into train and test sets.
    Returns:
        Tuple[np.ndarray, ydf.GradientBoostedTreesLearner, pd.DataFrame]: predictions, model, test_df
    """
    df["year"] = df["year"].astype("int")
    train_df = df[df["year"] <= year]
    test_df = df[df["year"] > year]
    df["year"] = df["year"].astype("str")
    train_df, test_df = drop_columns(train_df), drop_columns(test_df)

    model = LEARNER.train(train_df)
    return model.predict(test_df), model, test_df


def print_train_results(results: Results) -> None:
    print("\n\nOutcomes", end=f"\n{"*" * 20}\n")
    for k in results:
        print(
            f"Last n Races: {results[k]['last_n_races']}, SMA: {results[k]['sma']}, Num Races in Dataset: {results[k]['size']}, Success Rate: {k:.2%}"
        )


def run_train(
    df: pd.DataFrame | None = None,
    *,
    save: bool = False,
    model_name: str,
    last_n_races: int = 5,
    sma_lookback: int = 5,
) -> LEARNER_TYPE:
    drop = df is not None

    if df is None:
        df = get_df(
            2017,
            last_n_races=last_n_races,
            sma_lookback=sma_lookback,
        )

    _, model, _ = train(df, 2020)

    if drop:
        df = drop_columns(df)

    if save:
        save_model(model, model_name)
        df.to_csv(os.path.join(DPATH, f"{model_name}_dataset.csv"), index=False)

    return model


def process_results(results: Results, df: pd.DataFrame) -> LEARNER_TYPE | None:
    """Process the results of the model training and save the best model."""
    results_df = pd.DataFrame(
        [[*results[k].values(), k] for k in results],
        columns=[*ResultData.__annotations__.keys(), "success_rate"],
    )
    results_df["last_n_races"] = results_df["last_n_races"].apply(lambda x: int(x))
    results_df["sma"] = results_df["sma"].apply(lambda x: int(x))

    results_df.to_csv(os.path.join(BPATH, "test_results.csv"), index=False)

    if (max_result := max(results)) > 0.8:
        results_df = results_df[results_df["success_rate"] == max_result]
        row = results_df.iloc[0]

        if races := int(row["last_n_races"]):
            df = add_last_races(df, races)

        if sma_ := int(row["sma"]):
            df["sma"] = df.apply(lambda x: sma(x, df, "position", sma_), axis=1)

        print(
            f"\n\nRunning final model with the best parameters:\n\t{results[max_result]}\n"
        )
        return run_train(df, save=True, model_name="model_1")


def run_train_new() -> None:
    base_df = prepare_train_dataset(2017)
    outcomes: Results = {}
    races = 5
    saved = False

    for i in range(races):
        last_races_df = add_last_races(base_df, i + 1)

        for j in range(races):
            sma_df = last_races_df.copy()
            sma_df["sma"] = sma_df.apply(
                lambda x: sma(x, sma_df, "position", j + 1), axis=1
            )

            if len(sma_df[TARGET_LABEL].unique()) < 2:
                break

            _, model, test_df = train(sma_df, 2020)
            labels = model.label_classes()
            predictions = model.predict(test_df)
            success_rate = 0
            unqiue_races = len(sma_df["race_id"].unique())

            if not saved:
                drop_columns(sma_df).to_csv(
                    os.path.join(BPATH, "structure.csv"), index=False
                )
                saved = True

            for ind, pred in enumerate(predictions):
                result = pred.tolist().index(max(pred))
                if test_df.iloc[ind]["position_text"] == labels[result]:
                    success_rate += 1

            outcomes[
                (
                    success_rate / len(predictions)
                    if success_rate and predictions.any()
                    else 0.0
                )
            ] = {"last_n_races": i + 1, "sma": j, "size": unqiue_races}

    print_train_results(outcomes)
    process_results(outcomes, base_df)


if __name__ == "__main__":
    run_train_new()
