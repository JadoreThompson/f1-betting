import json
import os
from pandas import DataFrame

from .config import (
    BPATH,
    LEARNER_TYPE,
    MDPATH,
    MODEL_TYPE,
    MPATH,
    TRAINED_MODEL,
    TARGET_LABEL,
)
from .features.build_features import get_dataset, drop_features
from .features.utils import PosCat

LEARNER_PARAMS = {
    "label": TARGET_LABEL,
    "max_depth": 5,
    "num_trees": 25,
    "growing_strategy": "BEST_FIRST_GLOBAL",
    # "compute_permutation_variable_importance": True,
    "focal_loss_alpha": 0.01,
}
LEARNER: LEARNER_TYPE = LEARNER_TYPE(
    **LEARNER_PARAMS,
)
TOP_RANGE = False


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
        top_range_test_success: Success rate for top range on test set
        top_range_2024_success: Success rate for top range on 2024 data
        whole_test_success: Success rate for all positions on test set
        whole_2024_success: Success rate for all positions on 2024 data
    """
    print(f"\n\n{"*" * 20}")
    params = {k: v for k, v in locals().items() if k != "model" and k != "category"}

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
        "learner_params": LEARNER_PARAMS,
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
        print(json.dumps(params, indent=4))


def compute_success_rate(
    dataset: DataFrame,
    model=None,
    pos_cat: PosCat = None,
) -> float:
    """
    Calculate prediction success rate, optionally focusing on specific position ranges.

    Args:
        dataset: DataFrame with features and actual positions
        model: Model to evaluate (uses TRAINED_MODEL if None)
        pos_cat: Position category, required if TOP_RANGE is True

    Returns:
        Success rate as float between 0.0-1.0

    Raises:
        ValueError if TOP_RANGE is True but pos_cat is None
    """
    global TOP_RANGE

    if TOP_RANGE and pos_cat is None:
        raise ValueError("pos_cat must be passed if top_range is True.")

    success = 0.0
    count = 0

    if model is None:
        model = TRAINED_MODEL

    predictions = model.predict(dataset)
    pred_values = []  # To be added as a series

    for i, preds in enumerate(predictions):
        if len(model.label_classes()) == 2:
            pred_index = 0 if preds < 0.5 else 1
        else:
            pred_index = preds.tolist().index(max(preds))

        pred = model.label_classes()[pred_index]
        pred_values.append(pred)

        if TOP_RANGE:
            elibible = False

            if pos_cat == "tight":
                if pred > "0" or dataset.iloc[i][TARGET_LABEL] > "0":
                    elibible = True
            else:
                if "1" <= pred < "3" or "1" <= dataset.iloc[i][TARGET_LABEL] < "3":
                    elibible = True

            if elibible:
                count += 1
                if pred == dataset.iloc[i][TARGET_LABEL]:
                    success += 1
        else:
            if pred == dataset.iloc[i][TARGET_LABEL]:
                success += 1

    dataset["predictions"] = pred_values

    if success:
        if TOP_RANGE:
            success /= count
        else:
            success /= len(predictions)

    return success


def get_train_test(
    pos_cat: PosCat = "tight",
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
    df = df.dropna()
    train_df, test_df = (
        df[df["year"] <= split_year],
        df[df["year"] > split_year],
    )
    return drop_features(train_df), drop_features(test_df)


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

    model: MODEL_TYPE = LEARNER.train(train_df)
    print("Features:", model.input_feature_names())

    success_rate = compute_success_rate(test_df, model, pos_cat)
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


def evaluate_2024(pos_cat: PosCat, model=None) -> float:
    """
    Evaluate model performance on 2024 data.

    Args:
        pos_cat: Position category for evaluation
        model: Model to evaluate (uses TRAINED_MODEL if None)

    Returns:
        Success rate on 2024 data
    """
    df = get_dataset(pos_cat)
    df = df[df["year"] == 2024]
    df = drop_features(df)
    success = compute_success_rate(df, model, pos_cat)
    print(f"2024 success rate: {success:.2%}")
    return success


def train() -> None:
    global TOP_RANGE
    pos_cat = "winner"
    kwargs = {
        "pos_cat": "winner",
        "min_year": 2020,
        "max_year": 2023,
        "split_year": 2022,
    }

    TOP_RANGE = False
    model, whole_test_success = train_model(**kwargs)
    whole_2024_success = evaluate_2024(pos_cat, model)

    TOP_RANGE = True
    model, top_range_test_success = train_model(**kwargs)
    top_range_2024_success = evaluate_2024(pos_cat, model)

    save_train_configs(
        model,
        pos_cat,
        top_range_test_success,
        top_range_2024_success,
        whole_test_success,
        whole_2024_success,
    )


if __name__ == "__main__":
    train()
    # df = get_dataset()
    # df, _ = get_train_test()
    # df = drop_features(df)
    # print(df.dtypes)
    # df.to_csv("file.csv", index=False)
