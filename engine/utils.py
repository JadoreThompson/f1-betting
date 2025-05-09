import os
import numpy as np

from collections import namedtuple
from matplotlib import pyplot as plt
from pandas import DataFrame
from typing import Iterable, Optional

from .config import MPATH, TARGET_LABEL, TRAINED_MODEL
from .features.build_features import drop_features, get_dataset
from .features.utils import PosCat

Prediction = namedtuple("Prediction", ("prediction", "percentage"))


def save_model(model, name: str) -> None:
    fpath = os.path.join(MPATH, name)

    if os.path.exists(fpath):
        os.remove(fpath)

    model.save(fpath)


def interact(data: DataFrame | Iterable, model=None) -> tuple[Prediction, ...]:
    if model is None:
        model = TRAINED_MODEL

    if isinstance(data, Iterable):

        d = DataFrame(data, columns=model.input_feature_names())
    else:
        d = data
    preds: list[list[float] | float] = model.predict(d).tolist()

    if len(model_classes := model.label_classes()) == 2:
        return tuple(
            Prediction(model_classes[0 if prob < 0.5 else 1], prob) for prob in preds
        )
    else:
        return tuple(
            Prediction(model.label_classes()[row.index(m := max(row))], m)
            for row in preds
        )


def plot_heatmap(df: DataFrame) -> None:
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = get_dataset("top3")
    df = df[df["year"] == 2024]
    df = drop_features(df)
    df["target"] = df["target"].astype("int")
    plot_heatmap(df)
