import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import insert, select, update
from sqlalchemy.inspection import inspect

from config import BASE_PATH
from core.typing import F1ModelConfig
from db_models import (
    Base,
    Constructors,
    Drivers,
    Predictions,
    Qualifyings,
    ConstructorStandings,
    Races,
    Results,
)
from enums import PredictionOutcome, PredictionStatus
from model_development.clean import get_clean_df
from model_development.features import get_features_df
from utils.db import get_db_session


class MarketGenerator:
    # _model: CatBoostClassifier | None = None
    # _model_classes: list[str] | None = None

    _all_model: CatBoostClassifier | None = None
    _all_model_classes: list[str] | None = None
    _top3_model: CatBoostClassifier | None = None
    _top3_model_classes: list[str] | None = None
    _winer_model: CatBoostClassifier | None = None
    _winer_model_classes: list[str] | None = None

    @classmethod
    def _load_models(cls) -> None:
        def _load_model(
            outcome: PredictionOutcome,
        ) -> tuple[CatBoostClassifier, list[str]]:
            model_path = os.path.join(BASE_PATH, "model_development", "models")

            if outcome == PredictionOutcome.ALL:
                folder = os.path.join(model_path, "model-all-2")
            elif outcome == PredictionOutcome.TOP3:
                folder = os.path.join(model_path, "model-top3-2")
            elif outcome == PredictionOutcome.WINNER:
                folder = os.path.join(model_path, "model-winner-2")

            if not os.path.exists(folder):
                print(f"Model folder {folder} does not exist.")
                raise FileNotFoundError(f"Model folder {folder} does not exist.")

            model = CatBoostClassifier().load_model(os.path.join(folder, "model"))
            conf = json.load(open(os.path.join(folder, "config.json")))
            conf = F1ModelConfig(**conf)
            return model, conf.classes

        if (
            cls._all_model is None
            or cls._top3_model is None
            or cls._winer_model is None
        ):
            cls._all_model, cls._all_model_classes = _load_model(PredictionOutcome.ALL)
            cls._top3_model, cls._top3_model_classes = _load_model(
                PredictionOutcome.TOP3
            )
            cls._winer_model, cls._winer_model_classes = _load_model(
                PredictionOutcome.WINNER
            )

    @classmethod
    async def generate(cls, year: int, _round: int) -> dict[str, list[dict]]:
        cls._load_models()
        results = {}
        for outcome in PredictionOutcome:
            print(f"Generating predictions for {outcome.value}...")
            preds = await cls._generate_predictions(outcome, year, _round)
            results[outcome.value] = preds
        return results

    @classmethod
    async def _generate_predictions(
        cls, outcome: PredictionOutcome, year: int, _round: int
    ):
        if outcome == PredictionOutcome.ALL:
            model = cls._all_model
            model_classes = cls._all_model_classes
        elif outcome == PredictionOutcome.TOP3:
            model = cls._top3_model
            model_classes = cls._top3_model_classes
        elif outcome == PredictionOutcome.WINNER:
            model = cls._winer_model
            model_classes = cls._winer_model_classes
        else:
            raise ValueError(f"Invalid outcome: {outcome}")

        dfs = await cls._fetch_data()
        clean_df = get_clean_df(dfs)

        meta_cols = [
            "race_id",
            "driver_id",
            "driver_ref",
            "constructor_id",
            "constructor_ref",
            "nationality",
            "round",
        ]
        meta_df = clean_df[meta_cols].copy()

        features_df = get_features_df(clean_df)
        features_df["round"] = meta_df["round"]
        target_df = features_df[
            (features_df["year"] == year) & (features_df["round"] == _round)
        ].copy()

        driver_details = {
            row["driver_id"]: {
                "driver_ref": row["driver_ref"],
                "constructor_id": row["constructor_id"],
                "constructor_ref": row["constructor_ref"],
            }
            for _, row in meta_df.iterrows()
        }
        target_df["race_id"] = meta_df["race_id"]
        race_id = target_df["race_id"].unique()[0]
        target_df.pop("race_id")
        target_df.pop("created_at")

        # Predict probabilities
        pred_probs = model.predict_proba(target_df)
        pred_probs = [list(p) for p in pred_probs]
        target_df["driver_id"] = meta_df["driver_id"]

        # Map predictions
        preds = [(model_classes[p.index(max(p))], round(max(p), 3)) for p in pred_probs]

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
                    'outcome_class': outcome.value,
                }
            )

        # Insert into DB
        async with get_db_session() as sess:
            await sess.execute(
                update(Predictions)
                .values(status=PredictionStatus.CLOSED.value)
                .where(Predictions.outcome_class == outcome.value)
            )
            await sess.execute(insert(Predictions), rows)
            await sess.commit()

        return rows

    @classmethod
    async def _fetch_data(cls) -> dict[str, pd.DataFrame]:
        def to_dict(db_obj: Base) -> dict:
            return {
                c.key: getattr(db_obj, c.key)
                for c in inspect(db_obj).mapper.column_attrs
            }

        async with get_db_session() as sess:
            # Drivers
            result = await sess.execute(select(Drivers))
            drivers = pd.DataFrame([to_dict(row) for row in result.scalars().all()])

            # Qualifying
            result = await sess.execute(select(Qualifyings))
            qualifying = pd.DataFrame([to_dict(row) for row in result.scalars().all()])

            # Constructors
            result = await sess.execute(select(Constructors))
            constructors = pd.DataFrame([to_dict(row) for row in result.scalars().all()])

            # Constructor Standings
            result = await sess.execute(select(ConstructorStandings))
            constructor_standings = pd.DataFrame([to_dict(row) for row in result.scalars().all()])

            # Races
            result = await sess.execute(select(Races))
            races = pd.DataFrame([to_dict(row) for row in result.scalars().all()])

            # Results
            result = await sess.execute(select(Results))
            results = pd.DataFrame([to_dict(row) for row in result.scalars().all()])

        return {
            "constructors": constructors,
            "constructor_standings": constructor_standings,
            "drivers": drivers,
            "qualifying": qualifying,
            "races": races,
            "results": results,
        }
