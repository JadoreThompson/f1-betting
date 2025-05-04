import asyncio
import os
import pandas as pd

from .hyperparam_tester import HyperParamTester
from .utils import compute_success_rate, drop_columns, get_df, get_train_test, split_df
from .config import DPATH, LEARNER_TYPE, MODEL_TYPE, MPATH, TRAINED_MODEL

TARGET_LABEL = "positionText"
LEARNER_PARAMS = {
    "label": TARGET_LABEL,
    "max_depth": 3,
    "num_trees": 25,
    # "max_num_nodes": 50,
    "growing_strategy": "BEST_FIRST_GLOBAL",
    # "compute_permutation_variable_importance": True,
    "focal_loss_alpha": 0.01,
}

LEARNER: LEARNER_TYPE = LEARNER_TYPE(
    **LEARNER_PARAMS,
)


def get_sma_length():
    return 10


def get_top_range():
    return False


def train_model() -> MODEL_TYPE:
    train_df, test_df, _ = get_train_test(
        min_year=2017, max_year=2023, split_year=2021, sma_length=get_sma_length()
    )

    if test_df.empty:
        print("Empty test dataset")
        return

    train_df = pd.concat(
        [train_df, *[train_df[train_df["positionText"].isin(["1", "2"])]]],
        ignore_index=True,
    )

    model: MODEL_TYPE = LEARNER.train(train_df)
    print("Features:", model.input_feature_names())
    success_rate = compute_success_rate(test_df, TARGET_LABEL, model, get_top_range())
    print(f"Training success rate: {success_rate:.2%}")
    print(LEARNER_PARAMS)
    return model

    # if input("Save? (y/n)").lower() == "y":
    #     model.save(os.path.join(MPATH, "model_x"))
    #     train_df.to_csv(os.path.join(DPATH, "model_x_train_dataset.csv"), index=False)
    #     test_df.to_csv(os.path.join(DPATH, "model_x_test_dataset.csv"), index=False)


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


def func(model=None):
    df = get_df(2024, 2024, get_sma_length())
    df = drop_columns(df)
    success = compute_success_rate(df, TARGET_LABEL, model, get_top_range())
    print(f"2024 success rate: {success:.2%}")


if __name__ == "__main__":
    model = train_model()
    func(model)
    # test_hyperparams()
    # evaluate(df=split_df(get_df(2024, 2024, 5), 2024)[0])
