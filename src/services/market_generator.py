import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import insert, select

from config import BASE_PATH
from db_models import (
    Drivers,
    Predictions,
    Qualifyings,
    ConstructorStandings,
    Races,
    Results,
)
from model_development.clean import get_clean_df
from model_development.features import get_features_df
from utils.db import get_db_session


class MarketGenerator:
    _model: CatBoostClassifier | None = None
    _model_classes: list[str] | None = None
    _params: dict | None = None

    @classmethod
    def _load_models(cls) -> None:
        if (
            cls._model is not None
            or cls._model_classes is None
            or cls._params is not None
        ):
            return

        folder = os.path.join(BASE_PATH, "model_development")
        cls._model = CatBoostClassifier().load_model(
            os.path.join(folder, "models", "model-1")
        )
        cls._model_classes = cls._model.classes_
        cls._params = json.load(os.path.join(folder, "params", "params-1"))

    @classmethod
    async def generate(cls, year: int, _round: int):
        cls._load_models()
        dfs = await cls._fetch_data(year, _round)
        clean_df = get_clean_df(dfs)

        features_df = get_features_df(clean_df)
        target_df = features_df[
            (features_df["year"] == year) & (features_df["round"] == _round)
        ]
        race_id = target_df["race_id"].unique()[0]
        driver_details = {
            s["driverId"]: {
                "driver_ref": s["driver_ref"],
                "constructor_id": s["constructor_id"],
                "constructor_ref": s["constructor_ref"],
                "nationality": s["nationality"],
            }
            for _, s in target_df.iterrows()
        }

        pred_probs = cls._model.predict_proba(target_df)
        pred_probs = [list(p) for p in pred_probs]

        preds: list[tuple[str, float]] = []
        for probs in pred_probs:
            m_prob = max(probs)
            ind = probs.index(m_prob)
            _class = cls._model_classes[ind]
            preds.append((_class, round(m_prob, 3)))

        # Preparing
        rows: list[dict] = []
        for ind, (pred_cls, pred_prob) in enumerate(preds):
            driver_id = target_df.iloc[ind]["driver_id"]
            driver_detail = driver_details[driver_id]
            rows.append(
                {
                    "race_id": race_id,
                    "driver_id": driver_id,
                    "constructor_id": driver_detail["constructor_id"],
                    "predicted_position": pred_cls,
                    "predicted_probability": pred_prob,
                }
            )

        # Inserting
        async with get_db_session() as sess:
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
            "constructor_standings": constructor_standings,
            "drivers": drivers,
            "qualifying": qualifying,
            "races": races,
            "results": results,
        }
