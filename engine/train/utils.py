import json
import os
import ydf

from enum import Enum
from imblearn.over_sampling import SMOTE, SMOTENC
from pandas import DataFrame, Series
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from ..config import (
    BPATH,
    TRAINED_MODEL,
    TARGET_LABEL,
)
from ..features.build_features import get_dataset, drop_features
from ..features.utils import PosCat, get_position_category


@runtime_checkable
class SupportsPredict(Protocol):
    def predict(self, *args, **kwargs) -> Any: ...


def get_files(category: str) -> tuple[str, str, str]:
    """
    Get folder path and filenames for parameter tracking.

    Args:
        category: Parameter category name

    Returns:
        tuple[str, str, str]: (folder_path, old_filename, new_filename)
    """
    folder = os.path.join(BPATH, "params", category)
    if not os.path.exists(folder):
        os.mkdir(folder)

    old_fname = f"param_tracker_{category}_{len(os.listdir(folder)) - 1}.json"
    new_fname = f"param_tracker_{category}_{len(os.listdir(folder))}.json"

    return folder, old_fname, new_fname


def save_train_configs(
    model,
    pos_cat: PosCat,
    hparams: dict[str, Any],
    top_range_test_success: float = None,
    top_range_2024_success: float = None,
    whole_test_success: float = None,
    whole_2024_success: float = None,
) -> None:
    """
    Save model configuration and performance metrics if improvement is detected.

    Args:
        model: Trained model
        pos_cat: Position category used for training
        hparams: Hyperparameters applied to the model.
        top_range_test_success: Success rate for top range on test set
        top_range_2024_success: Success rate for top range on 2024 data
        whole_test_success: Success rate for all positions on test set
        whole_2024_success: Success rate for all positions on 2024 data
    """
    print(f"\n\n{"*" * 20}")

    if top_range_test_success is not None:
        top_range_test_success = round(top_range_test_success, 2)
    if top_range_2024_success is not None:
        top_range_2024_success = round(top_range_2024_success, 2)
    if whole_test_success is not None:
        whole_test_success = round(whole_test_success, 2)
    if whole_2024_success is not None:
        whole_2024_success = round(whole_2024_success, 2)

    folder, old_fname, new_fname = get_files(pos_cat)

    try:
        content = json.load(open(os.path.join(folder, old_fname), "r"))
    except FileNotFoundError:
        content = {}

    new_content = {
        "features": model.input_feature_names(),
        "learner_params": {
            k: (v.value if isinstance(v, Enum) else v) for k, v in hparams.items()
        },
    }

    gain = False
    old_top_range = old_whole = 0.0

    if (
        top_range_2024_success is not None
        and top_range_test_success is not None
        and whole_2024_success is not None
        and whole_test_success is not None
    ):
        gain = top_range_2024_success > (
            old_top_range := content.get("top_range", {}).get("2024", 0.0)
        ) and whole_2024_success > (
            old_whole := content.get("whole", {}).get("2024", 0.0)
        )

        if gain:
            new_content["top_range"] = {
                "test": top_range_test_success,
                "2024": top_range_2024_success,
            }
            new_content["whole"] = {
                "test": whole_test_success,
                "2024": whole_2024_success,
            }

    elif top_range_2024_success is not None and top_range_test_success is not None:
        gain = top_range_2024_success > (
            old_top_range := content.get("top_range", {}).get("2024", 0.0)
        )

        if gain:
            new_content["top_range"] = {
                "test": top_range_test_success,
                "2024": top_range_2024_success,
            }

    else:
        gain = whole_2024_success > (
            old_whole := content.get("whole", {}).get("2024", 0.0)
        )

        if gain:
            new_content["whole"] = {
                "test": whole_test_success,
                "2024": whole_2024_success,
            }

    if gain:
        json.dump(new_content, open(os.path.join(folder, new_fname), "w"), indent=4)
        print(
            f"Overall improvement - Top Range: + {top_range_2024_success - old_top_range:.2%}, Whole + {whole_2024_success - old_whole:.2%}."
        )
    else:
        print(f"No gain. Configs are the same. Achievied:")

        print(
            json.dumps(
                {
                    k: v
                    for k, v in locals().items()
                    if k
                    in (
                        "params",
                        "top_range_test_success",
                        "top_range_2024_success",
                        "whole_test_success",
                        "whole_2024_success",
                    )
                },
                indent=4,
            )
        )


def get_top_range_funcs(pos_cat: PosCat) -> Callable[[str, int, Series], bool]:
    if pos_cat == "loose":

        def func(pred: str, pred_ind: int, s: Series) -> bool:
            return pred in ("1", "2") or s.iloc[pred_ind] in ("1", "2")

    else:

        def func(pred: str, pred_ind: int, s: Series) -> bool:
            return pred > "0" or s.iloc[pred_ind] > "0"

    return func


def handle_classification(
    df: DataFrame, model: SupportsPredict, pos_cat: PosCat, top_range: bool
) -> tuple[float, list[str]]:
    target_s = df[TARGET_LABEL]
    predictions = model.predict(df)
    pred_values: list[str] = []
    success = 0.0
    count = 0

    if top_range:
        tr_func = get_top_range_funcs(pos_cat)

    for ind, preds in enumerate(predictions):
        if len(model.label_classes()) == 2:
            pred_index = 0 if preds < 0.5 else 1
        else:
            pred_index = preds.tolist().index(max(preds))

        pred = model.label_classes()[pred_index]
        pred_values.append(pred)

        if top_range:
            if tr_func(pred, ind, target_s):
                count += 1
                if pred == target_s.iloc[ind]:
                    success += 1
        else:
            if pred == target_s.iloc[ind]:
                success += 1

    if success:
        if top_range:
            success /= count
        else:
            success /= len(predictions)

    return success, pred_values


def handle_regression(
    df: DataFrame,
    model: SupportsPredict,
    top_range: bool,
    pos_cat: Optional[PosCat] = None,
) -> tuple[float, list[str]]:
    predictions: list = model.predict(df)
    pred_values: list[str] = []
    target_s: Series = df[TARGET_LABEL].apply(
        lambda x: get_position_category(str(x), pos_cat)
    )

    success = 0.0
    count = 0

    if top_range:
        tr_func = get_top_range_funcs(pos_cat)

    if pos_cat is None:
        for ind, pred in enumerate(predictions):
            if pred == df.iloc[ind][TARGET_LABEL]:
                success += 1
            pred_values.append(pred)

        success /= len(predictions)

    else:
        for ind, pred in enumerate(predictions):
            pred = get_position_category(str(round(pred)), pos_cat)

            if top_range:
                if tr_func(pred, ind, target_s):
                    count += 1
                    if pred == target_s.iloc[ind]:
                        success += 1
            else:

                if pred == target_s.iloc[ind]:
                    success += 1

            pred_values.append(pred)

        if top_range:
            if count:
                success /= count
        else:
            try:
                success /= len(predictions)
            except ZeroDivisionError:
                pass

    return success, pred_values


def compute_success_rate(
    dataset: DataFrame,
    model: Optional[SupportsPredict] = None,
    pos_cat: PosCat = None,
    *,
    top_range: bool = False,
) -> float:
    """
    Calculate prediction success rate, optionally focusing on specific position ranges.

    Args:
        dataset: DataFrame with features and actual positions
        model: Model to evaluate (uses TRAINED_MODEL if None)
        pos_cat: Position category, required if top_range is True

    Returns:
        Success rate as float between 0.0-1.0

    Raises:
        ValueError if top_range is True but pos_cat is None
    """
    if top_range and pos_cat is None:
        raise ValueError("pos_cat must be passed if top_range is True.")

    if model is None:
        model = TRAINED_MODEL
    dataset = dataset.copy()
    if (mtask := model.task()) == ydf.Task.CLASSIFICATION:
        success, pred_values = handle_classification(dataset, model, pos_cat, top_range)
    elif mtask == ydf.Task.REGRESSION:
        success, pred_values = handle_regression(dataset, model, top_range, pos_cat)

    dataset["prediction"] = pred_values
    return success


def get_train_test(
    pos_cat: PosCat,
    min_year: int = 2010,
    max_year: int = 2024,
    split_year: int = 2022,
) -> tuple[DataFrame, DataFrame]:
    """
    Get training and testing datasets filtered by year range.

    Args:
        pos_cat: Position category for dataset
        min_year: Minimum year to include
        max_year: Maximum year to include
        split_year: Year threshold for train/test split

    Returns:
        (train_df, test_df)
    """
    df = get_dataset(pos_cat)
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    train_df, test_df = (
        df[df["year"] <= split_year],
        df[df["year"] > split_year],
    )
    return drop_features(train_df), drop_features(test_df)


def balance_classes(df: DataFrame) -> DataFrame:
    """Balances the quantity of target classes
        using SMOTE.

    Args:
        df (DataFrame): DataFrame to be balanced.

    Returns:
        DataFrame: DataFrame with balanced target quantity
            of classes.
    """
    cfs = tuple(col for col in df.columns if df[col].dtype == "O" and col != "target")

    if cfs:
        smote = SMOTENC(
            categorical_features=cfs,
            random_state=42,
        )
    else:
        smote = SMOTE()

    x = df.drop("target", axis=1)
    y = df["target"]

    x_rs, y_rs = smote.fit_resample(x, y)
    x_rs["target"] = y_rs
    return x_rs
