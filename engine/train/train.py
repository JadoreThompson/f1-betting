import asyncio
import os
import pandas as pd

from .trainer_evaluator import TrainerEvaluator
from .utils import drop_columns, get_df, get_train_test, split_df
from ..config import DPATH, LEARNER_TYPE, MPATH, TRAINED_MODEL

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


def evaluate(model=None, df=None, *, save: bool = False, name: str = "model_x") -> None:
    if model is None:
        model = TRAINED_MODEL

    if df is None:
        df = pd.read_csv(os.path.join(DPATH, "test.csv"))

    predictions = model.predict(df)
    success = 0

    for i, preds in enumerate(predictions):
        if len(model.label_classes()) == 2:
            pred_index = 0 if preds < 0.5 else 1
        else:
            pred_index = preds.tolist().index(max(preds))

        pred = model.label_classes()[pred_index]

        if pred == df.iloc[i][TARGET_LABEL]:
            success += 1

    if success:
        success /= len(predictions)

    if save:
        df.to_csv(os.path.join(DPATH, f"{name}_dataset.csv"), index=False)
        model.save(os.path.join(MPATH, name))

    print(f"{"*" * 20}\nSuccess Rate: {success:.2%}\n{"*" * 20}")
    print(f"\nDF Types:\n{df.dtypes}")


def run_train() -> None:
    global LEARNER
    train_df, test_df, size = get_train_test(
        min_year=2017, max_year=2022, split_year=2021, sma_length=2
    )

    if test_df.empty:
        print("Empty test dataset")
        print(test_df)
    else:
        model = LEARNER.train(train_df)
        evaluate(
            model,
            test_df,
            save=True,
        )


def use_trainer_evaluator():
    t = TrainerEvaluator(TARGET_LABEL, LEARNER_TYPE)
    t.test_hyperparameters(
        rounds=24,
        last_n_races=2,
        params={
            "shrinkage": {"min": 0.01, "max": 1, "step": 0.01},
            "focal_loss_alpha": {"min": 0.01, "max": 1, "step": 0.01},
        },
    )


if __name__ == "__main__":
    run_train()
    # use_trainer_evaluator()
    # evaluate(df=split_df(get_df(2024, 2024, 5), 2024)[0])
