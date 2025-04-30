import os
import pandas as pd

from .config import DPATH, LEARNER_TYPE, MPATH, TRAINED_MODEL
from .utils import drop_columns, get_df, get_train_test, split_df

TARGET_LABEL = "positionText"
LEARNER: LEARNER_TYPE = LEARNER_TYPE(label=TARGET_LABEL)


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
        model.save(os.path.join(MPATH, name))

    print(f"{"*" * 20}\nSuccess Rate: {success:.2%}\n{"*" * 20}")
    print(f"\nDF Types:\n{df.dtypes}")


def rt() -> None:
    train_df, test_df, size = get_train_test(2017, 2023, 5)

    if test_df.empty:
        print("Empty test dataset")
        print(test_df)
    else:
        model = LEARNER.train(train_df)
        evaluate(model, test_df, save=True)


if __name__ == "__main__":
    rt()
    # evaluate(df=split_df(get_df(2024, 2024, 5), 2024)[0])
    # evaluate()
