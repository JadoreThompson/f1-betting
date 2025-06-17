import json
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sqlalchemy import insert, select
from typing import List
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
from model_development.enums import LoosePositionCategory
from model_development.features import (
    get_position_category,
    append_elo,
    append_elo_change,
    append_last_n,
    append_last_n_podiums,
    append_last_season_wins,
)
from model_development.preprocessing import merge_datasets
from utils.db import get_db_session


class MarketPipeline:
    """Generates F1 betting market predictions and saves them to the database."""

    def __init__(self) -> None:
        self._winner_model = None
        self._top3_model = None
        self._params_folder = os.path.join(BPATH, "params")
        self._scaler = StandardScaler()

    def _init(self) -> None:
        """Load models and their feature sets from disk."""
        self._winner_model = load_model(os.path.join(MPATH, "winner_v1"))
        self._winner_model_feats = json.load(
            open(
                os.path.join(
                    self._params_folder,
                    "forest",
                    "winner",
                    "param_tracker_winner_0.json",
                ),
                "rb",
            )
        )["features"]
        self._top3_model = load_model(os.path.join(MPATH, "top3_v1"))
        self._top3_model_feats = json.load(
            open(
                os.path.join(
                    self._params_folder,
                    "forest",
                    "top3",
                    "param_tracker_top3_5.json",
                ),
                "rb",
            )
        )["features"]

    async def run(
        self,
        year: int,
        round_: int,
        circuit_id: int,
        circuit_ref: str,
        driver_ids: List[int],
        quali_positions: List[int],
    ) -> None:
        """Execute the full pipeline for a given race.

        Args:
            year (int): Race year.
            round_ (int): Race round number.
            circuit_id (int): Circuit identifier.
            circuit_ref (str): Circuit reference name.
            driver_ids (List[int]): List of driver IDs.
            quali_positions (List[int]): Corresponding qualifying positions.
        """
        if round_ < 6:  # last 6 is a feature in dataset
            return

        if len(driver_ids) != len(quali_positions):
            raise ValueError("length of driver_ids must be the same as quali_pos.")

        self._init()
        ds: dict[str, pd.DataFrame] = await self._get_datasets()

        if any(ds[key].empty for key in ds):
            return

        merged_df = merge_datasets(ds)

        new_rows = self._generate_new_rows(
            merged_df,
            year,
            round_,
            circuit_id,
            circuit_ref,
            driver_ids,
            quali_positions,
        )

        merged_df = pd.concat([merged_df, new_rows], ignore_index=True)

        result = self._generate_winner_preds(merged_df, year, round_)
        if result is not None:
            drivers, preds = result
            markets = self._generate_winner_markets(preds, drivers, year, round_)
            if markets:
                await self._persist_markets(markets)

        result = self._generate_top3_preds(merged_df, year, round_)
        if result is not None:
            drivers, preds = result
            markets = self._generate_top3_markets(preds, drivers, year, round_)
            if markets:
                await self._persist_markets(markets)

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

        return {
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
                        "raceId": self._create_race_id(standing.year, standing.round),
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
                        "raceId": self._create_race_id(standing.year, standing.round),
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

    def _create_race_id(self, year: int, round_: int) -> int:
        """Create a unique race ID from year and round number.

        Args:
            year (int): The year of the race.
            round_ (int): The round number in the season.

        Returns:
            int: A unique integer identifier for the race.
        """
        return year * 100 + round_

    def _generate_new_rows(
        self,
        df: pd.DataFrame,
        year: int,
        round_: int,
        circuit_id: int,
        circuit_ref: str,
        driver_ids: List[int],
        quali_pos: List[int],
    ) -> pd.DataFrame:
        """Generate new data rows for the upcoming race.

        Args:
            df (pd.DataFrame): Historical data.
            year (int): Race year.
            round_ (int): Race round.
            circuit_id (int): Circuit ID.
            circuit_ref (str): Circuit reference.
            driver_ids (List[int]): Driver IDs.
            quali_pos (List[int]): Qualifying positions.

        Returns:
            pd.DataFrame: Updated rows with race features.
        """
        quali_pos_map = dict(list(zip(driver_ids, quali_pos)))
        new_rows = df[(df["year"] == year) & (df["round"] >= round_ - 1)].copy()

        new_rows["round"] = round_
        new_rows["circuitId"] = circuit_id
        new_rows["circuitRef"] = circuit_ref
        new_rows["grid"] = new_rows["position_quali"] = new_rows["driverId"].apply(
            lambda x: quali_pos_map.get(int(x))
        )

        new_rows["prev_points"] = new_rows["points"]
        new_rows["prev_position_driver_standings"] = new_rows[
            "position_driver_standings"
        ]
        new_rows["prev_wins"] = new_rows["wins"]
        new_rows["prev_points_constructor_standings"] = new_rows[
            "points_constructor_standings"
        ]
        new_rows["prev_position_constructor_standings"] = new_rows[
            "position_constructor_standings"
        ]

        return new_rows

    def _generate_winner_preds(
        self, df: pd.DataFrame, year: int, round_: int
    ) -> tuple[list[str], list[list[float]]] | None:
        """Predict race winners.

        Args:
            df (pd.DataFrame): Feature dataset.
            year (int): Race year.
            round_ (int): Race round.

        Returns:
            tuple[list[str], list[list[float]]]: Driver references and their prediction scores.
        """
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
        df = df[(df["year"] == year) & (df["round"] == round_)]
        driver_refs = df["driverRef"].tolist()
        df = df.drop(
            [
                col
                for col in df.columns
                if col not in self._winner_model_feats and col != "target"
            ],
            axis=1,
        )

        preds = [p.tolist() for p in self._winner_model.predict(df)]
        if not preds:
            return 
        
        return driver_refs, preds

    def _generate_top3_preds(
        self, df: pd.DataFrame, year: int, round_: int
    ) -> tuple[list[str], list[list[float]]] | None:
        """Predict top 3 finishers.

        Args:
            df (pd.DataFrame): Feature dataset.
            year (int): Race year.
            round_ (int): Race round.

        Returns:
            tuple[list[str], list[list[float]]]: Driver references and prediction scores
            for the passed year and round.
        """
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
        df = df[(df["year"] == year) & (df["round"] == round_)]
        driver_refs = df["driverRef"].tolist()
        df = df.drop(
            [
                col
                for col in df.columns
                if col not in self._top3_model_feats and col != "target"
            ],
            axis=1,
        )
        
        preds = [p.tolist() for p in self._winner_model.predict(df)]
        if not preds:
            return 
        
        return driver_refs, preds

    def _generate_winner_markets(
        self, preds: list[list[float]], drivers: list[str], year: int, round_: int
    ) -> list[dict]:
        """Convert winner predictions into market odds.

        Args:
            preds (list[list[float]]): Prediction probabilities.
            drivers (list[str]): Corresponding drivers.
            year (int): Race year.
            round_ (int): Race round.

        Returns:
            list[dict]: Market entries for betting.
        """
        classes = self._winner_model.label_classes()
        try:
            winner_index = classes.index(LoosePositionCategory.TOP_3.value)
        except ValueError:
            return []

        return [
            {
                "title": drivers[i],
                "category": MarketCategory.WINNER.value,
                "numerator": round(1 / prob),
                "denominator": 1,
                "year": year,
                "round": round_,
            }
            for i, driver_probs in enumerate(preds)
            if 0.7 > (prob := driver_probs[winner_index]) > 0.05
        ]

    def _generate_top3_markets(
        self, preds: list[list[float]], drivers: list[str], year: int, round_: int
    ) -> list[dict]:
        """Convert top 3 predictions into market odds.

        Args:
            preds (list[list[float]]): Prediction probabilities.
            drivers (list[str]): Corresponding drivers.
            year (int): Race year.
            round_ (int): Race round.

        Returns:
            list[dict]: Market entries for betting.
        """
        classes = self._top3_model.label_classes()

        try:
            top3_index = classes.index(LoosePositionCategory.TOP_3.value)
        except ValueError:
            return []

        return [
            {
                "title": drivers[i],
                "category": MarketCategory.TOP3.value,
                "numerator": round(1 / prob),
                "denominator": 1,
                "year": year,
                "round": round_,
            }
            for i, driver_probs in enumerate(preds)
            if 0.7 > (prob := driver_probs[top3_index]) > 0.05
        ]

    async def _persist_markets(
        self, markets: list[dict[str, str | float | int]]
    ) -> None:
        """Insert generated markets into the database.

        Args:
            markets (list[dict]): List of market entries.
        """
        async with get_db_session() as sess:
            await sess.execute(insert(Markets).values(markets))
            await sess.commit()
