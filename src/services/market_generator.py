import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import insert, select, update

from config import BASE_PATH
from db_models import (
    Constructors,
    Drivers,
    Predictions,
    Qualifyings,
    ConstructorStandings,
    Races,
    Results,
)
from enums import PredictionStatus
from model_development.clean import get_clean_df
from model_development.features import get_features_df
from utils.db import get_db_session


class MarketGenerator:
    _model: CatBoostClassifier | None = None
    _model_classes: list[str] | None = None

    @classmethod
    def _load_models(cls) -> None:
        if cls._model is not None and cls._model_classes is None:
            return

        folder = os.path.join(BASE_PATH, "model_development")
        cls._model = CatBoostClassifier().load_model(
            os.path.join(folder, "models", "model-1", "model")
        )
        cls._model_classes = cls._model.classes_

    @classmethod
    async def generate(cls, year: int, _round: int):
        cls._load_models()
        dfs = None
        clean_df = get_clean_df(dfs)

        meta_cols = [
            "race_id",
            "driver_id",
            "driver_ref",
            "constructor_id",
            "constructor_ref",
            "nationality",
        ]

        meta_df = clean_df[meta_cols].copy()

        features_df = get_features_df(clean_df)
        target_df = features_df[
            (features_df["year"] == year) & (features_df["round"] == _round)
        ].copy()

        print(len(target_df))
        print(clean_df[(clean_df["year"] == year) & (clean_df["round"] == _round)])

        driver_details = {
            row["driver_id"]: {
                "driver_ref": row["driver_ref"],
                "constructor_id": row["constructor_id"],
                "constructor_ref": row["constructor_ref"],
                "nationality": row["nationality"],
            }
            for _, row in meta_df.iterrows()
        }

        target_df["race_id"] = meta_df["race_id"]
        race_id = target_df["race_id"].unique()[0]
        target_df.pop("race_id")

        # Predict probabilities
        pred_probs = cls._model.predict_proba(target_df)
        pred_probs = [list(p) for p in pred_probs]
        target_df["driver_id"] = meta_df["driver_id"]

        # Map predictions
        preds = [
            (cls._model_classes[p.index(max(p))], round(max(p), 3)) for p in pred_probs
        ]

        # Prepare rows for insertion
        rows = []
        for ind, (pred_cls, pred_prob) in enumerate(preds):
            driver_id = target_df.iloc[ind]["driver_id"]
            driver_detail = driver_details[driver_id]
            rows.append(
                {
                    "race_id": int(race_id),
                    "driver_id": int(driver_id),
                    "constructor_id": int(driver_detail["constructor_id"]),
                    "predicted_position": pred_cls,
                    "predicted_probability": pred_prob * 100,
                    "status": PredictionStatus.OPEN.value,
                }
            )

        # Insert into DB
        async with get_db_session() as sess:
            await sess.execute(
                update(Predictions).values(status=PredictionStatus.CLOSED.value)
            )
            await sess.execute(insert(Predictions), rows)
            await sess.commit()

        return rows

    @classmethod
    async def _fetch_data(cls) -> dict[str, pd.DataFrame]:
        async with get_db_session() as sess:
            # Drivers
            result = await sess.execute(select(Drivers))
            drivers = pd.DataFrame(result.mappings().all())

            # Qualifying
            result = await sess.execute(select(Qualifyings))
            qualifying = pd.DataFrame(result.mappings().all())

            # Constructors
            result = await sess.execute(select(Constructors))
            constructors = pd.DataFrame(result.mappings().all())

            # Constructor Standings
            result = await sess.execute(select(ConstructorStandings))
            constructor_standings = pd.DataFrame(result.mappings().all())

            # Races
            result = await sess.execute(select(Races))
            races = pd.DataFrame(result.mappings().all())

            # Results
            result = await sess.execute(select(Results))
            results = pd.DataFrame(result.mappings().all())

        return {
            "constructors": constructors,
            "constructor_standings": constructor_standings,
            "drivers": drivers,
            "qualifying": qualifying,
            "races": races,
            "results": results,
        }
