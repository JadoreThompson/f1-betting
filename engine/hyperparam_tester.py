import json
import numpy as np

from typing import Any, Dict, Generator, TypedDict
from .utils import (
    compute_success_rate,
    drop_features,
    get_df,
    get_train_test,
)


class ParamSettings(TypedDict):
    min: float | None = None
    max: float | None = None
    step: float | None = None
    value: Any | None = None
    values: list[Any] | None = None


Params = Dict[str, ParamSettings]


class HyperParamTester:
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
        self._dataset = None

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

    def run(
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
        self._dataset = drop_features(get_df(2024, 2024, 2))

        best_avg_success = 0.0

        try:
            for param_combination in self._yield_params(params):
                self._learner = self._learner_type(
                    **param_combination, label=self._target_label
                )

                train_df, test_df, _ = get_train_test(
                    min_year=2017, max_year=2022, split_year=2021, sma_length=2
                )

                if test_df.empty or train_df.empty:
                    continue

                self._model = self._learner.train(train_df)

                result = compute_success_rate(self._dataset, self._model)

                if result > best_avg_success:
                    self._best_params = param_combination
                    best_avg_success = result
                    print("New AVG success:", best_avg_success)
        finally:
            self._best_params = {
                "average": best_avg_success,
                "params": self._best_params,
            }
            json.dump(self._best_params, open("params.json", "w"), indent=4)
            print(f"Best Average: {best_avg_success} - Params saved to 'params.json'")
