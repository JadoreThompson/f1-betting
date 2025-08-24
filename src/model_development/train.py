import os
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

from .features import get_features_df


def train():
    df = get_features_df()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train = df[df["year"] < 2023].copy()
    test = df[df["year"] >= 2023].copy()
    test = test[test["position_text"].isin(("1", "2", "3", "4", "5"))]

    X_train, X_test, y_train, y_test = (
        train,
        test,
        train.pop("position_text"),
        test.pop("position_text"),
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

    model.save_model(os.path.join(os.path.dirname(__file__), "models", "model-1"))
    # print(X_test.columns)
    # X_test.info()


if __name__ == "__main__":
    train()
