import json
import os

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

from core.typing import F1ModelConfig, F1ModelStats
from .features import get_features_df, get_features_top3_df, get_features_winner_df


def train():
    df = get_features_winner_df()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train = df[df["year"] < 2023].copy()
    test = df[df["year"] >= 2023].copy()

    # # For simplicity, only predict top 5
    # test = test[test["target"].isin(("1", "2", "3", "4", "5"))]

    X_train, X_test, y_train, y_test = (
        train,
        test,
        train.pop("target"),
        test.pop("target"),
    )

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=10,
        verbose=100,
        cat_features=[col for col in X_train.columns if X_train[col].dtype == "object"],
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    
    folder = os.path.join(os.path.dirname(__file__), "models", "model-winner-2")
    os.makedirs(folder, exist_ok=True)

    params = model.get_params()
    params.pop("cat_features", None)

    conf = F1ModelConfig(
        params=params,
        features=list(X_test.columns),
        classes=list(model.classes_),
        stats=F1ModelStats(
            accuracy=report["accuracy"], precision=report["macro avg"]["precision"]
        ),
    )
    json.dump(conf.model_dump(), open(os.path.join(folder, "config.json"), "w"))
    model.save_model(os.path.join(folder, "model"))


if __name__ == "__main__":
    train()
