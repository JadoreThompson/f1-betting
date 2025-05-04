import asyncio
import json
import os
import pandas as pd

from .hyperparam_tester import HyperParamTester
from .utils import compute_success_rate, drop_columns, get_df, get_train_test, split_df
from .config import DPATH, LEARNER_TYPE, MODEL_TYPE, MPATH, TRAINED_MODEL

TARGET_LABEL = "positionText"
LEARNER_PARAMS = {
    "label": TARGET_LABEL,
    "max_depth": 5,
    "num_trees": 100,
    # "max_num_nodes": 50,
    "growing_strategy": "BEST_FIRST_GLOBAL",
    # "compute_permutation_variable_importance": True,
    "focal_loss_alpha": 0.01,
}

LEARNER: LEARNER_TYPE = LEARNER_TYPE(
    **LEARNER_PARAMS,
)


SMA_LENGTH = 10
TOP_RANGE = False


# def get_sma_length():
#     return 10


# def get_top_range():
#     return False


def train_model(
    save_model: bool = False, model_name: str = "model_x"
) -> tuple[MODEL_TYPE, float]:
    global TOP_RANGE, SMA_LENGTH

    train_df, test_df, _ = get_train_test(
        min_year=2017, max_year=2022, split_year=2021, sma_length=SMA_LENGTH, save=True
    )

    if test_df.empty:
        print("Empty test dataset")
        return

    train_df = pd.concat(
        [train_df, train_df[train_df["positionText"].isin(["1", "2"])]],
        ignore_index=True,
    )

    model: MODEL_TYPE = LEARNER.train(train_df)
    print("Features:", model.input_feature_names())

    success_rate = compute_success_rate(test_df, TARGET_LABEL, model, TOP_RANGE)
    print(f"Training success rate: {success_rate:.2%}")
    print(LEARNER_PARAMS)

    if save_model:
        model.save(os.path.join(MPATH, model_name))
        train_df.to_csv(
            os.path.join(DPATH, f"{model_name}_train_dataset.csv"), index=False
        )
        test_df.to_csv(
            os.path.join(DPATH, f"{model_name}_test_dataset.csv"), index=False
        )

    return model, success_rate


def test_hyperparams():
    t = HyperParamTester(TARGET_LABEL, LEARNER_TYPE)
    t.run(
        rounds=24,
        last_n_races=2,
        params={
            "shrinkage": {"min": 0.01, "max": 1, "step": 0.01},
            "focal_loss_alpha": {"min": 0.01, "max": 1, "step": 0.01},
        },
    )


def evaluate_2024(model=None) -> float:
    global TOP_RANGE, SMA_LENGTH

    df = get_df(2024, 2024, SMA_LENGTH)
    df = drop_columns(df)
    success = compute_success_rate(df, TARGET_LABEL, model, TOP_RANGE)
    print(f"2024 success rate: {success:.2%}")
    return success


def save_train_configs(
    model,
    category: str,
    top_range_test_success: float,
    top_range_2024_success: float,
    whole_test_success: float,
    whole_2024_success: float,
) -> None:

    top_range_test_success = round(top_range_test_success, 2)
    top_range_2024_success = round(top_range_2024_success, 2)
    whole_test_success = round(whole_test_success, 2)
    whole_2024_success = round(whole_2024_success, 2)

    fname = f"param_tracker_{category}.json"

    try:
        content = json.load(open(fname, "r"))
    except FileNotFoundError:
        content = {}

    if top_range_2024_success > (
        old_top_range := content.get("top_range", {}).get("2024", 0.0)
    ) and whole_2024_success > (old_whole := content.get("whole", {}).get("2024", 0.0)):
        print(
            f"Overall improvement - Top Range: + {top_range_2024_success - old_top_range:.2%}, Whole {whole_2024_success - old_whole:.2%}."
        )
        content = {
            "features": model.input_feature_names(),
            "learner_params": LEARNER_PARAMS,
            "top_range": {
                "test": top_range_test_success,
                "2024": top_range_2024_success,
            },
            "whole": {"test": whole_test_success, "2024": whole_2024_success},
        }
        json.dump(content, open(fname, "w"), indent=4)
    else:
        print(
            f"No gain. Configs are the same. Results - Top Range: {top_range_2024_success:.2%} , Whole: {whole_2024_success:.2%}"
        )


def func() -> None:
    global TOP_RANGE

    TOP_RANGE = False
    model, whole_test_success = train_model()
    whole_2024_success = evaluate_2024(model)

    TOP_RANGE = True
    model, top_range_test_success = train_model()
    top_range_2024_success = evaluate_2024(model)

    save_train_configs(
        model,
        "top3",
        top_range_test_success,
        top_range_2024_success,
        whole_test_success,
        whole_2024_success,
    )


if __name__ == "__main__":
    # train_model(True, "loose")
    func()
    # get_df(
    #     2020,
    #     2024,
    # )
    # get_df(2010).to_csv("file.csv", index=False)

    # model, _ = train_model()
    # evaluate_2024(model)

    # save_train_configs(model)

    # test_hyperparams()
    # evaluate(df=split_df(get_df(2024, 2024, 5), 2024)[0])
