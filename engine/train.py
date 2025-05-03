import asyncio
import os
import pandas as pd

from .hyperparam_tester import HyperParamTester
from .utils import compute_success_rate, drop_columns, get_df, get_train_test, split_df
from .config import DPATH, LEARNER_TYPE, MPATH, TRAINED_MODEL

TARGET_LABEL = "positionText"
LEARNER_PARAMS = {
    "label": TARGET_LABEL,
    "max_depth": 5,
    "num_trees": 25,
    # "max_num_nodes": 9,
    # "growing_strategy": "BEST_FIRST_GLOBAL",
}

LEARNER: LEARNER_TYPE = LEARNER_TYPE(
    **LEARNER_PARAMS,
)


def train_model() -> None:
    sma_length = 2
    train_df, test_df, _ = get_train_test(
        min_year=2017, max_year=2022, split_year=2021, sma_length=sma_length
    )

    if test_df.empty:
        print("Empty test dataset")
        return

    model = LEARNER.train(train_df)
    compute_success_rate(test_df, model)


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


if __name__ == "__main__":
    train_model()
    # test_hyperparams()
    # evaluate(df=split_df(get_df(2024, 2024, 5), 2024)[0])
