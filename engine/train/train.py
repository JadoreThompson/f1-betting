import os
import ydf
import pandas as pd

from .utils import compute_success_rate, save_train_configs, get_train_test
from ..config import (
    LEARNER_TYPE,
    MDPATH,
    MODEL_TYPE,
    MPATH,
    TARGET_LABEL,
)
from ..features.build_features import get_dataset, drop_features
from ..features.utils import PosCat
from ..hyperparam_tester import HyperParamTester

HYERPARAMS = {
    # "max_depth": 5,
    # "num_trees": 45,
    # "focal_loss_alpha": 0.68,
    "max_depth": 5,
    "num_trees": 100,
    "focal_loss_alpha": 0.8,
}
LEARNER: LEARNER_TYPE = LEARNER_TYPE(
    label=TARGET_LABEL, task=ydf.Task.CLASSIFICATION, **HYERPARAMS
)
TOP_RANGE = False


def train_model(
    pos_cat: PosCat,
    *,
    min_year: int = 2017,
    max_year: int = 2022,
    split_year: int = 2021,
    save_model: bool = False,
    model_name: str = "model_x",
) -> tuple[MODEL_TYPE, float]:
    """
    Train model for given position category and evaluate on test data.

    Args:
        pos_cat: Position category for training
        min_year: Minimum year for dataset
        max_year: Maximum year for dataset
        split_year: Year threshold for train/test split
        save_model: Whether to save model and datasets to disk
        model_name: Name to use when saving model

    Returns:
        (trained_model, success_rate)
    """
    train_df, test_df = get_train_test(
        pos_cat, min_year=min_year, max_year=max_year, split_year=split_year
    )

    if test_df.empty:
        print("Empty test dataset")
        return

    train_df = pd.concat(
        [train_df, *([train_df[train_df["target"] == "1"]] * 1)], ignore_index=True
    )

    model: MODEL_TYPE = LEARNER.train(train_df)
    print("Features:", model.input_feature_names())

    success_rate = compute_success_rate(test_df, model, pos_cat, top_range=TOP_RANGE)
    print(f"Training success rate: {success_rate:.2%}")

    if save_model:
        model.save(os.path.join(MPATH, model_name))
        train_df.to_csv(
            os.path.join(MDPATH, f"{model_name}_train_dataset.csv"), index=False
        )
        test_df.to_csv(
            os.path.join(MDPATH, f"{model_name}_test_dataset.csv"), index=False
        )

    return model, success_rate


def evaluate_2024(
    pos_cat: PosCat, model=None
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate model performance on 2024 data.

    Args:
        pos_cat: Position category for evaluation
        model: Model to evaluate (uses TRAINED_MODEL if None)

    Returns:
        tuple[float, DataFrame, DataFrame]:
            - Success rate
            - DataFrame used to compute success rate
            - DataFrame retrieved from get_dataset call.
    """
    global TOP_RANGE
    raw_df = get_dataset(pos_cat)
    raw_df = raw_df[raw_df["year"] == 2024]
    df = drop_features(raw_df)
    success = compute_success_rate(df, model, pos_cat, top_range=TOP_RANGE)
    print(f"2024 success rate: {success:.2%}")
    return success, df, raw_df


def train() -> MODEL_TYPE:
    global TOP_RANGE

    pos_cat = "loose"
    kwargs = {
        "pos_cat": pos_cat,
        "min_year": 2017,
        "max_year": 2023,
        "split_year": 2022,
    }

    TOP_RANGE = False
    model, whole_test_success = train_model(**kwargs)
    whole_2024_success, _, _ = evaluate_2024(pos_cat, model)

    TOP_RANGE = True
    model, top_range_test_success = train_model(**kwargs)
    top_range_2024_success, dfa, dfb = evaluate_2024(pos_cat, model)

    save_train_configs(
        model,
        pos_cat,
        HYERPARAMS,
        top_range_test_success,
        top_range_2024_success,
        whole_test_success,
        whole_2024_success,
    )

    dfb.to_csv("raw.csv", index=False)
    dfa.to_csv("eval.csv", index=False)
    print(dfa.dtypes)
    return model


def test_hyperparams() -> None:
    ht = HyperParamTester(TARGET_LABEL, LEARNER_TYPE)
    ht.run(
        "top3",
        {
            "max_depth": {"min": 3, "max": 100, "step": 1},
            "num_trees": {"min": 5, "max": 1000, "step": 5},
            "growing_strategy": {"value": "BEST_FIRST_GLOBAL"},
            "focal_loss_alpha": {"min": 0.01, "max": 0.99, "step": 0.01},
        },
        True,
        10_000,
        1,
    )


if __name__ == "__main__":
    train()
    # test_hyperparams()
    # df = get_dataset("top3")
    # df = df[df["year"] == 2024]
    # df.to_csv("file.csv", index=False)
