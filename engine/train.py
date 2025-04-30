import json
import os
import numpy as np
import pandas as pd

from typing import Tuple, TypedDict
from .config import BPATH, DPATH, LEARNER_TYPE, MPATH, TRAINED_MODEL
from .utils import drop_columns, get_df, get_train_test, sma

TARGET_LABEL = "positionText"
LEARNER: LEARNER_TYPE = LEARNER_TYPE(label=TARGET_LABEL)


def evaluate(model=None, df=None, *, save: bool = False) -> None:
    if model is None:
        model = TRAINED_MODEL

    if df is None:
        df = pd.read_csv(os.path.join(DPATH, "test.csv"))

    predictions = model.predict(df)
    success = 0

    for i, preds in enumerate(predictions):
        pred_index = preds.tolist().index(max(preds))
        pred = model.label_classes()[pred_index]

        if pred == df.iloc[i][TARGET_LABEL]:
            success += 1

    if success:
        success /= len(predictions)

    if save:
        model.save(os.path.join(MPATH, "model_x"))

    print(f"Succes Rate: {success:.2%}")
    print(f"\nDF Types:\n{df.dtypes}")


def rt() -> None:
    train_df, test_df, size = get_train_test(2017, 3)
    # print(train_df.dtypes)
    model = LEARNER.train(train_df)
    evaluate(model, test_df, save=True)


if __name__ == "__main__":
    rt()
    # evaluate(df=drop_columns(get_df(2024)))

    # races_df = pd.read_csv(os.path.join(DPATH, "races.csv"))
    # races_df[races_df["year"] <= 2023].to_csv(
    #     os.path.join(DPATH, "races-2023.csv"), index=False
    # )