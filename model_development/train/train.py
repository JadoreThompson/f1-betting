import json
import os
import pandas as pd
import pickle
import ydf

from typing import Literal, Optional

from .utils import (
    balance_classes,
    compute_success_rate,
    get_classification_report,
    get_files,
    save_train_configs_forest,
    get_train_test,
)
from ..config import (
    LEARNER_TYPE,
    MODEL_TYPE,
    MPATH,
    TARGET_LABEL,
)
from ..features.build_features import build_features, drop_features
from ..features.utils import PosCat
from ..preprocessing import merge_datasets

# Model hyperparameters and learner definition
HYERPARAMS = {
    "max_depth": 5,
    "num_trees": 100,
    "min_examples": 1,
    "growing_strategy": "BEST_FIRST_GLOBAL",
    "task": ydf.Task.CLASSIFICATION,
}
LEARNER: LEARNER_TYPE = LEARNER_TYPE(label=TARGET_LABEL, **HYERPARAMS)


class EmptyDataFrame(Exception):
    """Raised when a DataFrame is unexpectedly empty during training or evaluation."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def evalute_performance_forest(
    eval_pos_cat: PosCat,
    top_range_test_success: Optional[float] = None,
    top_range_eval_success: Optional[float] = None,
    whole_test_success: Optional[float] = None,
    whole_eval_success: Optional[float] = None,
) -> bool:
    """
    Compares current model performance to previous saved metrics and determines
    whether the current model has improved.

    Args:
        eval_pos_cat (PosCat): Evaluation label category.
        top_range_test_success (Optional[float]): Top-range test success rate.
        top_range_eval_success (Optional[float]): Top-range eval success rate.
        whole_test_success (Optional[float]): Full test set success rate.
        whole_eval_success (Optional[float]): Full eval set success rate.

    Returns:
        bool: True if the model outperforms previous configurations in any tracked metric.
    """
    folder, old_fname, _ = get_files(eval_pos_cat, "forest")

    try:
        prev_content = json.load(open(os.path.join(folder, old_fname), "r"))
    except FileNotFoundError:
        prev_content = {}

    if (
        top_range_eval_success is not None
        and prev_content.get("top_range", {}).get("eval") is not None
    ):
        if top_range_eval_success > prev_content["top_range"]["eval"]:
            return True

    if (
        top_range_test_success is not None
        and prev_content.get("top_range", {}).get("test") is not None
    ):
        if top_range_test_success > prev_content["top_range"]["test"]:
            return True

    if (
        whole_eval_success is not None
        and prev_content.get("whole", {}).get("eval") is not None
    ):
        if whole_eval_success > prev_content["whole"]["eval"]:
            return True

    if (
        whole_test_success is not None
        and prev_content.get("whole", {}).get("test") is not None
    ):
        if whole_test_success > prev_content["whole"]["test"]:
            return True

    return False


def save_train_configs(
    mtype: Literal["forest"],
    eval_pos_cat: PosCat,
    features: list[str],
    top_range_test_success: Optional[float] = None,
    top_range_eval_success: Optional[float] = None,
    whole_test_success: Optional[float] = None,
    whole_eval_success: Optional[float] = None,
) -> None:
    """
    Saves current model configuration and evaluation performance to disk for future comparison.

    Args:
        mtype (Literal["forest"]): Model type.
        eval_pos_cat (PosCat): Evaluation label category.
        features (list[str]): List of feature names used in training.
        top_range_test_success (Optional[float]): Success rate on top-range test.
        top_range_eval_success (Optional[float]): Success rate on top-range eval.
        whole_test_success (Optional[float]): Success rate on entire test set.
        whole_eval_success (Optional[float]): Success rate on entire eval set.
    """
    config = {
        "features": features,
        "hparams": {
            k: (str(v) if isinstance(v, ydf.Task) else v) for k, v in HYERPARAMS.items()
        },
        "top_range": {
            "test": round(top_range_test_success, 2),
            "eval": round(top_range_eval_success, 2),
        },
        "whole": {
            "test": round(whole_test_success, 2),
            "eval": round(whole_eval_success, 2),
        },
    }

    folder, _, fname = get_files(eval_pos_cat, mtype)
    os.makedirs(folder, exist_ok=True)
    json.dump(config, open(os.path.join(folder, fname), "w"))

    print(f"Training configuration saved to {os.path.join(folder, fname)}")


def _train_forest_model(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_range: bool,
    eval_pos_cat: PosCat,
    log: bool = True,
) -> tuple[MODEL_TYPE, float]:
    """
    Trains a forest model and evaluates it on the test set.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        top_range (bool): Whether to compute success based on top-N predictions.
        eval_pos_cat (PosCat): Category for positive class evaluation.
        log (bool): Whether to print training info.

    Returns:
        tuple[MODEL_TYPE, float]: Trained model and test success rate.
    """
    model = LEARNER.train(train_df)

    if log:
        print("Features:")
        for feature in model.input_feature_names():
            print(f"  - {feature}")

        print("\nClasses:")
        for label in model.label_classes():
            print(f"  - {label}")

    success_rate, _ = compute_success_rate(
        test_df, model, eval_pos_cat, top_range=top_range
    )

    if log:
        print(f"Testing success rate: {success_rate:.2%}")

    return model, success_rate


def train_forest(
    dataset_pos_cat: PosCat,
    eval_pos_cat: PosCat,
    *,
    min_year: int = 2017,
    max_year: int = 2023,
    split_year: int = 2022,
    eval_year: int = 2024,
) -> MODEL_TYPE:
    """
    Orchestrates the full training pipeline for a forest model:
    data preparation, feature engineering, model training, and performance evaluation.

    Args:
        dataset_pos_cat (PosCat): Positive class for training dataset.
        eval_pos_cat (PosCat): Positive class for evaluation dataset.
        min_year (int): Minimum year to include in training.
        max_year (int): Year used for test set.
        split_year (int): Max year for training data.
        eval_year (int): Year used for evaluation set.

    Returns:
        MODEL_TYPE: The trained forest model.
    """
    raw_df: pd.DataFrame = merge_datasets()
    features_df: pd.DataFrame = build_features(raw_df, dataset_pos_cat)

    train_df = features_df[
        (features_df["year"] >= min_year) & (features_df["year"] <= split_year)
    ]
    test_df = features_df[features_df["year"] == max_year]
    eval_df = features_df[features_df["year"] == eval_year]

    train_df = drop_features(train_df)
    test_df = drop_features(test_df)
    eval_df = drop_features(eval_df)

    # Whole population training
    top_range = False
    model, whole_test_success = _train_forest_model(
        train_df=train_df,
        test_df=test_df,
        top_range=top_range,
        eval_pos_cat=eval_pos_cat,
    )
    whole_eval_success, _ = compute_success_rate(
        eval_df, model, eval_pos_cat, top_range=top_range
    )

    # Top-range evaluation
    top_range = True
    model, top_range_test_success = _train_forest_model(
        train_df=train_df,
        test_df=test_df,
        top_range=top_range,
        eval_pos_cat=eval_pos_cat,
    )
    top_range_eval_success, _ = compute_success_rate(
        eval_df, model, eval_pos_cat, top_range=top_range
    )

    improved = evalute_performance_forest(
        eval_pos_cat,
        top_range_test_success,
        top_range_eval_success,
        whole_test_success,
        whole_eval_success,
    )

    if improved:
        save_train_configs(
            "forest",
            eval_pos_cat,
            [f.name for f in model.input_features()],
            top_range_test_success,
            top_range_eval_success,
            whole_test_success,
            whole_eval_success,
        )

    return model


def main() -> None:
    ydf.verbose(0)
    model = train_forest("loose", "top3")


if __name__ == "__main__":
    main()
