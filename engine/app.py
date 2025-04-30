import os
import pandas as pd

from .config import DPATH
from .train import add_last_n_races
from .utils import apply_types, plot_heatmap


def func():
    results_df = pd.read_csv(os.path.join(DPATH, "results.csv"))
    results_df["position"] = (
        pd.to_numeric(results_df["position"], errors="coerce").fillna(0).astype("int")
    )

    df = results_df
    df.columns = [
        col.replace("Id", "_id").replace("Text", "_text") for col in df.columns
    ]
    df = apply_types(results_df)
    df = add_last_n_races(df, 5)

    for col in df.columns:
        if "last_" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int")

    plot_heatmap(df)


if __name__ == "__main__":
    func()
