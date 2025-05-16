import json
import os
import pandas as pd
import pickle
import ydf

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Optional

from .utils import (
    balance_classes,
    compute_success_rate,
    save_train_configs,
    get_train_test,
)
from ..config import (
    LEARNER_TYPE,
    MDPATH,
    MODEL_TYPE,
    MPATH,
    TARGET_LABEL,
)
from ..features.build_features import get_dataset, drop_features
from ..features.utils import PosCat
from ..hyperparam_tester import HyperParamTester

HYERPARAMS = {
    # "max_depth": 90,
    # "num_trees": 500,
    "max_depth": 5,
    "num_trees": 100,
    # "focal_loss_alpha": 0.8,
    "min_examples": 1,
    "growing_strategy": "BEST_FIRST_GLOBAL",
    "task": ydf.Task.CLASSIFICATION,
}
LEARNER: LEARNER_TYPE = LEARNER_TYPE(label=TARGET_LABEL, **HYERPARAMS)
TOP_RANGE = False


class EmptyDataFrame(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def train_model(
    pos_cat: PosCat,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    *,
    min_year: int = 2017,
    max_year: int = 2022,
    split_year: int = 2021,
    save_model: bool = False,
    model_name: str = "model_x",
) -> tuple[MODEL_TYPE, float]:
    global TOP_RANGE
    if train_df is None or test_df is None:
        train_df, test_df = get_train_test(
            pos_cat, min_year=min_year, max_year=max_year, split_year=split_year
        )

    if test_df.empty:
        raise EmptyDataFrame("Empty test dataset.")

    model: MODEL_TYPE = LEARNER.train(train_df)
    print("Features:", model.input_feature_names())

    success_rate = compute_success_rate(test_df, model, pos_cat, top_range=TOP_RANGE)

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


def evaluate_2024(
    pos_cat: PosCat, df: Optional[pd.DataFrame] = None, model=None
) -> tuple[float, pd.DataFrame]:
    global TOP_RANGE

    if df is None:
        raw_df = get_dataset(pos_cat)
        raw_df = raw_df[raw_df["year"] == 2024]
        df = drop_features(raw_df)

    success = compute_success_rate(df, model, pos_cat, top_range=TOP_RANGE)
    print(f"2024 success rate: {success:.2%}")
    return success, df


def train() -> MODEL_TYPE:
    global TOP_RANGE

    dataset_pos_cat = "loose"
    eval_pos_cat = "loose"
    kwargs = {
        "min_year": 2017,
        "max_year": 2023,
        "split_year": 2022,
    }

    raw_df = get_dataset(dataset_pos_cat)
    raw_df = balance_classes(raw_df).sort_values(["year", "round"])

    # training
    train_df, test_df = (
        raw_df[
            (raw_df["year"] >= kwargs["min_year"])
            & (raw_df["year"] <= kwargs["split_year"])
        ],
        raw_df[raw_df["year"] == kwargs["max_year"]],
    )
    df_2024 = raw_df[raw_df["year"] == 2024]

    train_df, test_df, df_2024 = (
        drop_features(train_df).dropna(),
        drop_features(test_df).dropna(),
        drop_features(df_2024).dropna(),
    )

    TOP_RANGE = False
    model, whole_test_success = train_model(
        eval_pos_cat, train_df=train_df, test_df=test_df
    )
    whole_2024_success, _ = evaluate_2024(eval_pos_cat, df_2024, model)

    TOP_RANGE = True
    model, top_range_test_success = train_model(
        eval_pos_cat, train_df=train_df, test_df=test_df
    )
    top_range_2024_success, eval_df = evaluate_2024(eval_pos_cat, df_2024, model)

    save_train_configs(
        model,
        eval_pos_cat,
        HYERPARAMS,
        top_range_test_success,
        top_range_2024_success,
        whole_test_success,
        whole_2024_success,
    )

    eval_df.to_csv("eval.csv", index=False)
    # df_2024.to_csv("2024.csv", index=False)

    # print(json.dumps(model.variable_importances(), indent=4))
    return model


def test_hyperparams() -> None:
    ht = HyperParamTester(TARGET_LABEL, LEARNER_TYPE)
    ht.run(
        "loose",
        {
            "max_depth": {"min": 3, "max": 100, "step": 1},
            "num_trees": {"min": 5, "max": 1000, "step": 5},
            "growing_strategy": {"value": "BEST_FIRST_GLOBAL"},
            # "focal_loss_alpha": {"min": 0.01, "max": 0.99, "step": 0.01},
        },
        True,
        2000,
        1,
    )


def train_log_regression():
    df = get_dataset("top3")

    df_2024 = drop_features(df[df["year"] == 2024])

    df = df[df["year"] < 2024]
    df = drop_features(df.dropna())
    df = balance_classes(df)

    Y = df.pop("target")
    X = df

    df_2024_Y = df_2024.pop("target")
    df_2024_X = df_2024

    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    y_encoded_2024 = le.fit_transform(df_2024_Y)

    print("Class distribution:\n", pd.Series(Y).value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    # 2024 for evaluation, validating the findings
    _, X_test_2024, _, y_test_2024 = train_test_split(
        df_2024_X,
        y_encoded_2024,
        test_size=0.95,
        random_state=42,
        stratify=y_encoded_2024,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_2024_scaled = scaler.transform(X_test_2024)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_pred_2024 = clf.predict(X_test_2024_scaled)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(
        "Classification Report:\n",
        classification_report(y_test, y_pred, target_names=le.classes_),
    )
    print("*" * 20)
    print("Confusion Matrix:\n", confusion_matrix(y_test_2024, y_pred_2024))
    print(
        "Classification Report:\n",
        classification_report(y_test_2024, y_pred_2024, target_names=le.classes_),
    )

    # pickle.dump(clf, open("log_reg_model.pkl", "wb"))

    return clf, scaler, le


if __name__ == "__main__":
    train()
    # train_log_regression()
    # test_hyperparams()
    # df = get_dataset("top3")
    # df = df[df["year"] == 2024]
    # df.to_csv("file.csv", index=False)
