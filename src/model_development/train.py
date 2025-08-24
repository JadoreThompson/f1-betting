from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import get_features_df


def train():
    df = get_features_df()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train = df[df["year"] < 2024].copy()
    test = df[df["year"] >= 2024].copy()
    test = test[test["positionText"].isin(("1", "2", "3"))]

    X_train, X_test, y_train, y_test = (
        train,
        test,
        train.pop("positionText"),
        test.pop("positionText"),
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


if __name__ == "__main__":
    train()
