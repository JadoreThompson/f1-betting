import asyncio
import json
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sqlalchemy import insert, select
from ydf import load_model

from db_models import (
    Circuits,
    ConstructorStandings,
    Constructors,
    DriverStandings,
    Drivers,
    GrandPrixResults,
    Markets,
    QualiResults,
)
from enums import MarketCategory
from model_development.config import BPATH, MPATH
from model_development.features import (
    get_position_category,
    append_elo,
    append_elo_change,
    append_elo_percentile,
    append_elo_rank_in_race,
    append_avg_position_move,
    append_constructor_encodings,
    append_last_n,
    append_last_n_podiums,
    append_last_season_wins,
)
from model_development.preprocessing import merge_datasets
from utils.db import get_db_session


class MarketPipeline:
    """Pipeline for generating F1 betting market predictions.

    This class loads models and datasets, processes features,
    makes predictions for race winners and top 3 positions,
    and persists market odds to the database.
    """

    def __init__(self) -> None:
        """Initialize the Pipeline with scaler and model placeholders."""
        self._winner_model = None
        self._top3_model = None
        self._params_folder = os.path.join(BPATH, "params")
        self._scaler = StandardScaler()

    def _init(self) -> None:
        """Initialize models from disk"""
        self._winner_model = load_model(os.path.join(MPATH, "winner_v1"))
        self._top3_model = load_model(os.path.join(MPATH, "top3_v1"))

    async def run(self, year: int, round_: int) -> None:
        """Start the pipeline process to generate and store market predictions.

        Loads datasets, applies feature engineering, makes predictions for
        winner and top 3 finishers, and stores the calculated betting odds
        into the database.
        """
        self._init()
        ds = await self._get_datasets()

        if any(ds[key].empty for key in ds):
            return

        merged_df = merge_datasets(ds)
        merged_df.to_csv("m.csv", index=False)

        result = self._get_winner_preds(merged_df)

        if result is not None:
            drivers, preds = result

            winner_markets = self._generate_markets_lr(
                preds, drivers, MarketCategory.WINNER, year, round_
            )
            await self._persist_markets(winner_markets)

        result = self._get_top3_preds(merged_df)
        if result is not None:
            drivers, preds = result
            top3_markets = self._generate_markets_lr(
                preds, drivers, MarketCategory.TOP3, year, round_
            )
            await self._persist_markets(top3_markets)

    async def _get_datasets(self) -> dict[str, pd.DataFrame]:
        """Retrieve datasets from the database and structure them for modeling.

        Returns:
            dict[str, pd.DataFrame]: Dictionary of structured DataFrames representing
            races, results, drivers, constructors, standings, and qualifying.
        """
        async with get_db_session() as sess:
            circuits = (await sess.execute(select(Circuits))).scalars().all()
            constructors = (await sess.execute(select(Constructors))).scalars().all()
            drivers = (await sess.execute(select(Drivers))).scalars().all()
            driver_standings = (
                (await sess.execute(select(DriverStandings))).scalars().all()
            )
            constructor_standings = (
                (await sess.execute(select(ConstructorStandings))).scalars().all()
            )
            results = (await sess.execute(select(GrandPrixResults))).scalars().all()
            qualifying = (await sess.execute(select(QualiResults))).scalars().all()

            datasets = {
                "circuits": pd.DataFrame(
                    [
                        {
                            "circuitId": circuit.circuit_id,
                            "circuitRef": circuit.circuit_ref,
                        }
                        for circuit in circuits
                    ]
                ),
                "constructors": pd.DataFrame(
                    [
                        {
                            "constructorId": constructor.constructor_id,
                            "constructorRef": constructor.constructor_ref,
                        }
                        for constructor in constructors
                    ]
                ),
                "constructor_standings": pd.DataFrame(
                    [
                        {
                            "raceId": self._create_race_id(
                                standing.year, standing.round
                            ),
                            "constructorId": standing.constructor_id,
                            "points": standing.points,
                            "position": standing.position,
                        }
                        for standing in constructor_standings
                    ]
                ),
                "drivers": pd.DataFrame(
                    [
                        {
                            "driverId": driver.driver_id,
                            "driverRef": driver.driver_ref,
                            "dob": driver.dob,
                            "nationality": driver.nationality,
                        }
                        for driver in drivers
                    ]
                ),
                "driver_standings": pd.DataFrame(
                    [
                        {
                            "raceId": self._create_race_id(
                                standing.year, standing.round
                            ),
                            "driverId": standing.driver_id,
                            "points": standing.points,
                            "position": standing.position,
                            "wins": standing.wins,
                        }
                        for standing in driver_standings
                    ]
                ),
                "results": pd.DataFrame(
                    [
                        {
                            "raceId": self._create_race_id(result.year, result.round),
                            "driverId": result.driver_id,
                            "constructorId": result.constructor_id,
                            "grid": result.grid,
                            "position": result.position,
                            "positionText": result.position_text,
                            "positionOrder": result.position,
                            "statusId": 1,
                        }
                        for result in results
                    ]
                ),
                "races": pd.DataFrame(
                    [
                        {
                            "raceId": self._create_race_id(result.year, result.round),
                            "circuitId": result.circuit_id,
                            "year": result.year,
                            "round": result.round,
                        }
                        for result in results
                    ]
                ).drop_duplicates(),
                "qualifying": pd.DataFrame(
                    [
                        {
                            "raceId": self._create_race_id(quali.year, quali.round),
                            "driverId": quali.driver_id,
                            "position": quali.position,
                        }
                        for quali in qualifying
                    ]
                ),
            }

            return datasets

    def _create_race_id(self, year: int, round_: int) -> int:
        """Create a unique race ID from year and round number.

        Args:
            year (int): The year of the race.
            round_ (int): The round number in the season.

        Returns:
            int: A unique integer identifier for the race.
        """
        return year * 100 + round_

    def _get_winner_preds(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[tuple[float, float]]] | None:
        """Generate predictions for race winners.

        Applies feature engineering and uses a trained model to predict
        winning probabilities for each driver.

        Args:
            df (pd.DataFrame): Merged dataset with race and driver features.

        Returns:
            tuple[list[str], list[tuple[float, float]]] | None:
                A tuple of driver references and their predicted probabilities,
                or None if the data is insufficient.
        """
        ptracker = json.load(
            open(
                os.path.join(
                    self._params_folder,
                    "forest",
                    "winner",
                    "param_tracker_winner_0.json",
                ),
                "rb",
            )
        )
        used_feats = ptracker["features"]

        df.to_csv("f.csv", index=False)
        df["target"] = df["positionText"].apply(
            lambda x: get_position_category(x, "loose")
        )

        df = append_elo(df)
        df = append_elo_change(df)
        df = append_last_n(df, "target", window=6)
        if df.empty:
            return  # Checking after last_n performs df.dropna

        df = append_last_n_podiums(df, window=0)
        df = df.dropna()

        driver_refs = df["driverRef"].tolist()

        df = df.drop(
            [col for col in df.columns if col not in used_feats and col != "target"],
            axis=1,
        )
        return driver_refs, self._winner_model.predict(df)

    def _get_top3_preds(self, df: pd.DataFrame) -> tuple[list[str], list[float, float]]:
        """Generate predictions for top 3 race positions.

        Applies feature engineering and uses a trained model to predict
        top 3 probabilities for each driver.

        Args:
            df (pd.DataFrame): Merged dataset with race and driver features.

        Returns:
            tuple[list[str], list[float, float]]: A tuple of driver references and their predicted top 3 probabilities.
        """
        ptracker = json.load(
            open(
                os.path.join(
                    self._params_folder,
                    "forest",
                    "top3",
                    "param_tracker_top3_5.json",
                ),
                "rb",
            )
        )
        used_feats = ptracker["features"]

        df.to_csv("f.csv", index=False)
        df["target"] = df["positionText"].apply(
            lambda x: get_position_category(x, "loose")
        )

        df = append_elo(df)
        df = append_elo_change(df)
        df = append_last_n(df, "target", window=6)
        if df.empty:
            return  # Checking after last_n performs df.dropna

        df = append_last_n_podiums(df, window=0)
        df = append_last_season_wins(df)
        df = df.dropna()

        driver_refs = df["driverRef"].tolist()

        df = df.drop(
            [col for col in df.columns if col not in used_feats and col != "target"],
            axis=1,
        )
        return driver_refs, self._winner_model.predict(df)

    def _generate_markets_lr(
        self,
        preds: list[tuple[float, float]],
        drivers: list[str],
        category: MarketCategory,
        year: int,
        round_: int,
    ) -> tuple[dict[str, str | float | int], ...]:
        """Convert predicted probabilities into betting odds.

        Args:
            preds (list[tuple[float, float]]): Predicted probabilities for each driver.
            drivers (list[str]): List of driver identifiers.
            category (MarketCategory): Betting market category.

        Returns:
            tuple[dict[str, str | float | int], ...]: Betting odds data for persistence.
        """
        return tuple(
            {
                "title": drivers[ind],
                "category": category.value,
                "numerator": round(1 / prob),
                "denomiator": 1,
                "year": year,
                "round": round_,
            }
            for ind, (_, prob) in preds
        )

    async def _persist_markets(self, markets: list[dict[str, str | float | int]]):
        """Persist generated betting markets to the database.

        Args:
            markets (list[dict[str, str | float | int]]): List of market records to insert.
        """
        async with get_db_session() as sess:
            await sess.execute(insert(Markets).values(markets))
            await sess.commit()
