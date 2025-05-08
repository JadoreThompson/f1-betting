import json
import optuna
import tqdm  # For optuna
import ydf

from pandas import DataFrame
from typing import Any, Dict, Type, TypedDict

from .features.build_features import (
    drop_features,
    get_dataset,
)
from .features.utils import PosCat
from .train.utils import compute_success_rate, get_train_test


class ParamSettings(TypedDict, total=False):
    min: float
    max: float
    step: float
    value: Any
    values: list[Any]


Params = Dict[str, ParamSettings]


class HyperParamTester:
    """
    Handles dataset construction, model training, hyperparameter tuning, and evaluation
    for a specified machine learning learner type, using Optuna for optimization.
    """

    def __init__(self, target_label: str, learner_type: Type) -> None:
        self._learner_type: Type = learner_type
        self._target_label: str = target_label

        self._best_hyperparams: dict = {}
        self._best_score: float = float("-inf")

        self._train_df: DataFrame | None = None
        self._eval_df: DataFrame | None = None
        self._pos_cat: PosCat | None = None
        self._top_range: bool | None = None
        self._hparams: Params | None = None  # Not to be mutated once set

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna.
        A single trial trains and evaluates the model with a set of hyperparameters
        suggested by Optuna.
        """
        current_params = {}

        for p_name, settings in self._hparams.items():
            if "value" in settings:
                current_params[p_name] = settings["value"]
            elif "values" in settings:
                current_params[p_name] = trial.suggest_categorical(
                    p_name, settings["values"]
                )
            elif "min" in settings and "max" in settings:
                if isinstance(settings["min"], float) or isinstance(
                    settings["max"], float
                ):
                    suggest_func = trial.suggest_float
                elif isinstance(settings["min"], int) and isinstance(
                    settings["max"], int
                ):
                    suggest_func = trial.suggest_int
                else:
                    raise ValueError(
                        f"Parameter '{p_name}' has unsupported min/max types for Optuna suggestion."
                    )

                current_params[p_name] = suggest_func(
                    p_name,
                    settings["min"],
                    settings["max"],
                    step=settings.get("step"),
                )
            else:
                raise ValueError(
                    f"Parameter '{p_name}' in params_def is not configured correctly for Optuna."
                )

        try:
            learner = self._learner_type(
                **current_params, label=self._target_label, task=ydf.Task.CLASSIFICATION
            )
            model = learner.train(self._train_df)
            success_rate = compute_success_rate(
                self._eval_df, model, self._pos_cat, top_range=self._top_range
            )
            return success_rate
        except Exception as e:
            print(type(e), str(e))

    def run(
        self,
        pos_cat: PosCat,
        hparams: Params,
        top_range: bool,
        n_trials: int = 100,
        n_jobs: int = 1,
    ) -> None:
        """
        Runs hyperparameter search using Optuna.

        Args:
            pos_cat: Position category for data processing.
            params_definition: Dict defining hyperparameter search space for Optuna.
            top_range: Passed to compute_success_rate.
            n_trials: Total number of hyperparameter combinations to test.
            n_jobs: Number of processes for parallel execution. Use -1 for all CPUs.
        """
        self._pos_cat = pos_cat
        self._top_range = top_range
        self._hparams = hparams

        df = get_dataset(self._pos_cat)
        self._eval_df = drop_features(df[df["year"] == 2024])
        self._train_df, _ = get_train_test(self._pos_cat, 2017, 2023, 2022)

        if self._train_df.empty:
            raise ValueError("Training dataset is empty. Check data and date ranges.")
        if self._eval_df.empty:
            raise ValueError("Evaluation dataset (for 2024) is empty. Check data.")

        study = optuna.create_study(direction="maximize")

        print(
            f"Starting Optuna hyperparameter search for {n_trials} trials using {n_jobs} parallel jobs."
        )

        study.optimize(
            self._objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True
        )

        if study.best_trial:
            self._best_hyperparams = study.best_params
            self._best_score = study.best_value
        else:
            self._best_hyperparams = {}
            self._best_score = float("-inf")
            print("Warning: Optuna study completed without any successful trials.")

        # self._save_results()

    def _save_results(self) -> None:
        """Helper method to save the results to a JSON file."""
        results_to_save = {
            "best_score": (
                self._best_score if self._best_score > float("-inf") else None
            ),
            "best_params": self._best_hyperparams,
            "notes": "Optuna search results.",
        }
        if self._best_score <= float("-inf") or not self._best_hyperparams:
            results_to_save["notes"] = (
                "Optuna search did not find a successful set of parameters or all trials failed."
            )

        with open("optuna_params.json", "w") as f:
            json.dump(results_to_save, f, indent=4)

        print("\n--- Optuna Hyperparameter Search Complete ---")
        if results_to_save["best_score"] is not None:
            print(f"Best score achieved: {results_to_save['best_score']:.4f}")
            print(f"Best params: {results_to_save['best_params']}")
        else:
            print("No best score found (all trials might have failed).")
        print(f"Results saved to 'optuna_params.json'")
