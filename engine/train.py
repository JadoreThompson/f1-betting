import json
import os
import pandas as pd

from .hyperparam_tester import HyperParamTester
from .utils import compute_success_rate, drop_features, get_df, get_train_test
from .config import (
    BPATH,
    LEARNER_TYPE,
    MDPATH,
    MODEL_TYPE,
    MPATH,
    TRAINED_MODEL,
    TARGET_LABEL,
)

LEARNER_PARAMS = {
    "label": TARGET_LABEL,
    "max_depth": 5,
    "num_trees": 100,
    "growing_strategy": "BEST_FIRST_GLOBAL",
    # "compute_permutation_variable_importance": True,
    "focal_loss_alpha": 0.01,
}

LEARNER: LEARNER_TYPE = LEARNER_TYPE(
    **LEARNER_PARAMS,
)


SMA_LENGTH = 10
TOP_RANGE = False


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

    # train_df = pd.concat(
    #     [train_df, train_df[train_df["positionText"].isin(["1", "2"])]],
    #     ignore_index=True,
    # )

    model: MODEL_TYPE = LEARNER.train(train_df)
    print("Features:", model.input_feature_names())

    success_rate = compute_success_rate(test_df, model, TOP_RANGE, "tight")
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
    df = drop_features(df)
    success = compute_success_rate(df, model, TOP_RANGE, "tight")
    print(f"2024 success rate: {success:.2%}")
    return success


def get_files(category: str) -> tuple[str, str, str]:
    folder = os.path.join(BPATH, "params", category)
    if not os.path.exists(folder):
        os.mkdir(folder)

    old_fname = f"param_tracker_{category}_{len(os.listdir(folder)) - 1}.json"
    new_fname = f"param_tracker_{category}_{len(os.listdir(folder))}.json"

    return folder, old_fname, new_fname


def save_train_configs(
    model,
    category: str,
    top_range_test_success: float = None,
    top_range_2024_success: float = None,
    whole_test_success: float = None,
    whole_2024_success: float = None,
) -> None:
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

    folder, old_fname, new_fname = get_files(category)

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


def func() -> None:
    global TOP_RANGE

    TOP_RANGE = False
    model, whole_test_success = train_model(save_model=True)
    whole_2024_success = evaluate_2024(model)

    TOP_RANGE = True
    model, top_range_test_success = train_model()
    top_range_2024_success = evaluate_2024(model)

    save_train_configs(
        model,
        "tight",
        top_range_test_success,
        top_range_2024_success,
        whole_test_success,
        whole_2024_success,
    )


if __name__ == "__main__":
    # func()
    # get_train_test(
    #     min_year=2017, max_year=2022, split_year=2021, sma_length=SMA_LENGTH, save=True
    # )
    # df = get_df(2017, 2024)
    # df.to_csv("file.csv", index=False)
    a = compute_success_rate(drop_features(get_df(2024)), TRAINED_MODEL, True, "top3")
    print(a)
