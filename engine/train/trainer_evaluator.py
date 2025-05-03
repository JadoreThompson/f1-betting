import json
import numpy as np
import pandas as pd

from typing import Any, Dict, Generator, TypedDict
from engine.evaluate import (
    Dataset,
    ParsedRaceData,
    get_features,
    construct_parsed_race_data,
    fetch_race_results_v2,
)
from .utils import get_position_category, get_train_test, interact


class ParamSettings(TypedDict):
    min: float | None = None
    max: float | None = None
    step: float | None = None
    value: Any | None = None
    values: list[Any] | None = None


Params = Dict[str, ParamSettings]


class TrainerEvaluator:
    """
    Handles dataset construction, model training, hyperparameter tuning, and evaluation
    for a specified machine learning learner type.

    This class automates training and evaluating models using historical race data,
    tests multiple hyperparameter configurations, and stores the best-performing
    configuration based on evaluation success.

    Attributes:
        _learner_type (type): A callable class/type for the machine learning model to be trained.
        _learner: An instance of the learner, initialized with current parameters.
        _model: The trained model object.
        _target_label (str): The label name used as ground truth during training and evaluation.
        _best_params (dict): The best hyperparameter combination found during evaluation.
        _datasets (list[tuple[list, str]]): Feature/label pairs for evaluation.
    """

    def __init__(self, target_label: str, learner_type) -> None:
        """
        Initializes the TrainerEvaluator.

        Args:
            target_label (str): The column name of the target label to predict.
            learner_type: The learner class or factory used to instantiate models.
        """
        self._learner_type = learner_type
        self._learner: learner_type = None
        self._model = None
        self._target_label = target_label
        self._best_params: dict = {}
        self._datasets: list[tuple[list, str]] = []

    def _train_model(self) -> bool:
        """
        Trains the learner model on a historical dataset and evaluates it on a holdout test set.

        The training/test data is split by year, and a model is trained using the
        specified learner type. The trained model is then evaluated for immediate feedback.
        """
        train_df, test_df, _ = get_train_test(
            min_year=2017, max_year=2022, split_year=2021, sma_length=2
        )

        if test_df.empty:
            print("Empty test dataset.")
            return False

        self._model = self._learner.train(train_df, verbose=0)
        self._evaluate_model(
            self._model,
            test_df,
        )
        return True

    def _evaluate_model(self, model, df: pd.DataFrame) -> None:
        """
        Evaluates the trained model using accuracy against the target label.

        Args:
            model: The trained learner model.
            df (pd.DataFrame): The test dataset to evaluate on.
        """
        predictions = model.predict(df)
        success = 0

        for i, preds in enumerate(predictions):
            if len(model.label_classes()) == 2:
                pred_index = 0 if preds < 0.5 else 1
            else:
                pred_index = preds.tolist().index(max(preds))

            pred = model.label_classes()[pred_index]

            if pred == df.iloc[i][self._target_label]:
                success += 1

        if success:
            success /= len(predictions)

    def _build_datasets(self, *, rounds: int, last_n_races: int) -> None:
        """
        Constructs feature datasets from historical race data.

        Args:
            rounds (int): Number of race rounds to include per year.
            last_n_races (int): Number of past races to consider per driver.
        """
        years = [2024]
        race_data: Dataset[ParsedRaceData] = {}

        for i in range(len(years)):
            fetched_data = {
                j: fetch_race_results_v2(years[i], j + 1) for j in range(rounds)
            }

            for key in fetched_data:
                race_data[key + (i * rounds)] = fetched_data[key]

        race_data = construct_parsed_race_data(race_data, last_n_races)

        for round_, round_data in race_data.items():
            for driver_id, driver_data in round_data.items():
                self._datasets.append(
                    (
                        get_features(
                            race_data=race_data,
                            last_n_races=last_n_races,
                            round_=round_,
                            name=driver_id,
                        ),
                        driver_data.real,
                    )
                )

    def _yield_params(self, params: Params) -> Generator[dict[str, Any], None, None]:
        """
        Yields all valid hyperparameter combinations based on the search space.

        Args:
            params (Params): A dictionary specifying parameter search ranges or values.

        Yields:
            dict[str, Any]: A single hyperparameter configuration.
        """
        var_params = {}
        constant_params = {}

        for p, settings in params.items():
            if vals := settings.get("values"):
                var_params[p] = vals
            elif val := settings.get("value"):
                constant_params[p] = val
            else:
                var_params[p] = np.arange(
                    settings.get("min", 0),
                    settings.get("max", 1),
                    settings.get("step", 0.1),
                )

        keys = list(var_params.keys())
        total_combinations = len(var_params[keys[0]])

        for key in keys[1:]:
            total_combinations *= len(var_params[key])

        print("TC:", total_combinations)

        combinations: list[Any] = []
        yield_val = {
            **constant_params,
            **{key: val[0] for key, val in var_params.items()},
        }

        for _ in range(total_combinations):
            for key, vals in var_params.items():
                for val in vals:
                    temp_obj = {**yield_val, key: val}

                    if temp_obj in combinations:
                        continue

                    yield_val[key] = val
                    yvc = yield_val.copy()
                    combinations.append(yvc)
                    yield yvc

    def _test_unseen(self) -> float:
        """
        Evaluates the trained model on unseen datasets and computes accuracy
        based on position category matches.

        Returns:
            float: The success rate of predictions falling into the target category range.
        """
        success = 0.0
        total = 0

        for dataset, real in self._datasets:
            pred = interact([dataset], self._model)[0]

            if (
                "1" <= pred.prediction < "3"
                or "1" <= get_position_category(real, "loose") < "3"
            ):
                total += 1
                if pred.prediction == get_position_category(real, "loose"):
                    success += 1

        if success and total:
            success /= total

        return success

    def test_hyperparameters(
        self,
        *,
        rounds: int,
        last_n_races: int,
        params: Params,
    ) -> None:
        """
        Runs a hyperparameter search by training and evaluating models over all combinations.

        Args:
            rounds (int): Number of race rounds to use in the dataset.
            last_n_races (int): Number of historical races per driver for feature construction.
            params (Params): Dictionary defining hyperparameter search space.

        Side Effects:
            Trains and evaluates multiple models.
            Updates self._best_params with the top-performing configuration.
        """
        self._build_datasets(rounds=rounds, last_n_races=last_n_races)

        best_avg_success = 0.0
        for param_combination in self._yield_params(params):
            self._learner = self._learner_type(
                **param_combination, label=self._target_label
            )

            if self._train_model():
                result = self._test_unseen()

                if result > best_avg_success:
                    self._best_params = param_combination
                    best_avg_success = result
                    print("New AVG success:", best_avg_success)

        try:
            self._best_params = {"average": best_avg_success, "params": self._best_params}
            json.dump(self._best_params, open("params.json", "w"), indent=4)
        finally:
            print(f"Best Average: {best_avg_success}", json.dumps(self._best_params))
