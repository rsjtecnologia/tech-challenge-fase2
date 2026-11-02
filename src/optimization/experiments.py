"""Dataset preparation and repeatable genetic-optimization experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.optimization.genetic import (
    EXPERIMENT_CONFIGS,
    GAConfig,
    GeneticOptimizer,
)


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("breast_cancer", "pcos")
MODEL_NAMES = (
    "Regressao Logistica",
    "Arvore de Decisao",
    "SVM",
    "Gradient Boosting",
    "Random Forest",
)
SEED = 42

SEARCH_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "Regressao Logistica": {
        "C": (0.01, 0.1, 1.0, 10.0, 100.0),
        "class_weight": (None, "balanced"),
    },
    "Arvore de Decisao": {
        "max_depth": (None, 3, 5, 8, 12, 20),
        "min_samples_split": (2, 5, 10, 20),
        "min_samples_leaf": (1, 2, 4, 8),
        "criterion": ("gini", "entropy"),
        "class_weight": (None, "balanced"),
    },
    "SVM": {
        "C": (0.1, 1.0, 10.0, 100.0),
        "gamma": ("scale", "auto", 0.001, 0.01, 0.1),
        "kernel": ("rbf", "linear"),
        "class_weight": (None, "balanced"),
    },
    "Gradient Boosting": {
        "n_estimators": (50, 100, 150, 200, 250),
        "learning_rate": (0.01, 0.03, 0.05, 0.1, 0.2),
        "max_depth": (1, 2, 3, 4),
        "subsample": (0.7, 0.85, 1.0),
    },
    "Random Forest": {
        "n_estimators": (100, 200, 300, 400),
        "max_depth": (None, 5, 10, 20, 30),
        "min_samples_leaf": (1, 2, 4, 8),
        "max_features": ("sqrt", 0.5, 1.0),
        "class_weight": (None, "balanced"),
    },
}


def load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load source data and reproduce the Phase 1 feature transformations."""
    if dataset == "breast_cancer":
        path = (
            REPOSITORY_ROOT
            / "src"
            / "db"
            / "models"
            / "breast-cancer-wisconsin-data"
            / "data.csv"
        )
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        target = frame.pop("diagnosis").map({"M": 1, "B": 0})
        features = frame.drop(columns=["id", "Unnamed: 32"], errors="ignore")
        return features, target.astype(int), "Câncer de mama (Wisconsin)"

    if dataset == "pcos":
        path = (
            REPOSITORY_ROOT
            / "src"
            / "db"
            / "models"
            / "polycystic-ovary-syndrome-pcos"
            / "PCOS_infertility.csv"
        )
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        frame = frame.rename(
            columns={
                "I   beta-HCG(mIU/mL)": "beta_hcg_i",
                "II    beta-HCG(mIU/mL)": "beta_hcg_ii",
                "AMH(ng/mL)": "amh",
                "PCOS (Y/N)": "pcos",
            }
        )
        for column in ("beta_hcg_i", "beta_hcg_ii", "amh"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ("beta_hcg_i", "beta_hcg_ii"):
            frame.loc[frame[column] == 1.99, column] = np.nan

        frame["beta_hcg_max"] = frame[["beta_hcg_i", "beta_hcg_ii"]].max(axis=1)
        frame["beta_hcg_diff"] = (
            frame["beta_hcg_i"] - frame["beta_hcg_ii"]
        ).abs()
        frame["beta_hcg_ratio"] = frame["beta_hcg_i"] / frame[
            "beta_hcg_ii"
        ].replace(0, np.nan)
        frame["amh_alto"] = (frame["amh"] >= 4.5).astype(int)
        frame["beta_i_censurado"] = frame["beta_hcg_i"].isna().astype(int)
        frame["beta_ii_censurado"] = frame["beta_hcg_ii"].isna().astype(int)
        target = frame.pop("pcos").astype(int)
        features = frame.drop(
            columns=["Sl. No", "Patient File No."], errors="ignore"
        )
        return features, target, "Síndrome do ovário policístico (PCOS)"

    raise ValueError(f"Estudo desconhecido: {dataset}")


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numericas",
                Pipeline(
                    [
                        ("imputacao", SimpleImputer(strategy="median")),
                        ("padronizacao", StandardScaler()),
                    ]
                ),
                slice(None),
            )
        ],
        remainder="drop",
    )


def _classifier(
    name: str,
    params: dict[str, Any],
    dataset: str,
    seed: int = SEED,
) -> Any:
    if name == "Regressao Logistica":
        defaults: dict[str, Any] = {
            "max_iter": 2000,
            "random_state": seed,
            "class_weight": "balanced" if dataset == "pcos" else None,
        }
        return LogisticRegression(**(defaults | params))
    if name == "Arvore de Decisao":
        defaults = {
            "random_state": seed,
            "class_weight": "balanced" if dataset == "pcos" else None,
        }
        return DecisionTreeClassifier(**(defaults | params))
    if name == "SVM":
        svc = SVC(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel=params.get("kernel", "rbf"),
            class_weight=params.get("class_weight", "balanced"),
        )
        return CalibratedClassifierCV(
            estimator=svc,
            method="sigmoid",
            cv=3 if params else 5,
            ensemble=False,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            random_state=seed,
            **(
                {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 1.0,
                }
                | params
            ),
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
            **(
                {
                    "n_estimators": 300,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                }
                | params
            ),
        )
    raise ValueError(f"Modelo desconhecido: {name}")


def _pipeline(name: str, params: dict[str, Any], dataset: str) -> Pipeline:
    return Pipeline(
        [
            ("preprocessador", _preprocessor()),
            ("classificador", _classifier(name, params, dataset)),
        ]
    )


def _scores(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    folds: StratifiedKFold,
) -> dict[str, float]:
    result = cross_validate(
        estimator,
        features,
        target,
        cv=folds,
        scoring={
            "accuracy": "accuracy",
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "precision": make_scorer(precision_score, zero_division=0),
            "balanced_accuracy": "balanced_accuracy",
        },
        n_jobs=1,
        error_score="raise",
    )
    return {
        metric: float(np.mean(result[f"test_{metric}"]))
        for metric in (
            "accuracy",
            "recall",
            "f1",
            "precision",
            "balanced_accuracy",
        )
    }


def _holdout_metrics(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    prediction = estimator.predict(features)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction)
        ),
    }


def run_optimization(
    datasets: tuple[str, ...] = DATASETS,
    model_names: tuple[str, ...] = MODEL_NAMES,
    configs: tuple[GAConfig, ...] = EXPERIMENT_CONFIGS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run all configured genetic experiments and compare on one fixed holdout."""
    if not datasets or not model_names or not configs:
        raise ValueError("É necessário informar estudos, modelos e experimentos.")
    unknown_datasets = set(datasets).difference(DATASETS)
    unknown_models = set(model_names).difference(MODEL_NAMES)
    if unknown_datasets or unknown_models:
        raise ValueError(
            f"Configuração inválida: estudos={sorted(unknown_datasets)}, "
            f"modelos={sorted(unknown_models)}"
        )

    output: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "fitness": {
            "formula": "0.50*recall + 0.30*f1 + 0.20*accuracy",
            "selection_data": "stratified cross-validation on training split only",
        },
        "studies": {},
    }

    for dataset in datasets:
        features, target, label = load_dataset(dataset)
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=seed,
            stratify=target,
        )
        study_result: dict[str, Any] = {
            "dataset": dataset,
            "label": label,
            "samples": len(features),
            "features": list(features.columns),
            "positive_class": "malignant" if dataset == "breast_cancer" else "PCOS",
            "split": {
                "train": len(x_train),
                "test": len(x_test),
                "test_size": 0.2,
                "stratified": True,
            },
            "models": {},
        }

        for model_name in model_names:
            LOGGER.info(
                "model_start study=%s model=%s train=%d test=%d",
                dataset,
                model_name,
                len(x_train),
                len(x_test),
            )
            started = time.perf_counter()
            baseline = _pipeline(model_name, {}, dataset)
            baseline.fit(x_train, y_train)
            baseline_test = _holdout_metrics(baseline, x_test, y_test)

            experiment_results: list[dict[str, Any]] = []
            for config in configs:
                folds = StratifiedKFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=seed,
                )

                def evaluate(params: dict[str, Any]) -> dict[str, float]:
                    candidate = _pipeline(model_name, params, dataset)
                    return _scores(candidate, x_train, y_train, folds)

                search = GeneticOptimizer(
                    search_space=SEARCH_SPACES[model_name],
                    config=config,
                    evaluator=evaluate,
                    seed=seed,
                )
                result = search.run()
                experiment_results.append(
                    {
                        "name": config.name,
                        "best_params": result.best_params,
                        "best_fitness": result.best_fitness,
                        "best_cv_metrics": result.best_metrics,
                        "history": result.history,
                        "unique_candidates_evaluated": result.evaluations,
                        "config": result.config,
                    }
                )
                LOGGER.info(
                    "experiment_complete study=%s model=%s experiment=%s "
                    "fitness=%.5f evaluations=%d",
                    dataset,
                    model_name,
                    config.name,
                    result.best_fitness,
                    result.evaluations,
                )

            winning_experiment = max(
                experiment_results, key=lambda item: item["best_fitness"]
            )
            optimized = _pipeline(
                model_name,
                winning_experiment["best_params"],
                dataset,
            )
            optimized.fit(x_train, y_train)
            optimized_test = _holdout_metrics(optimized, x_test, y_test)
            study_result["models"][model_name] = {
                "baseline": {
                    "hyperparameters": "Phase 1 defaults",
                    "test_metrics": baseline_test,
                },
                "optimized": {
                    "selected_experiment": winning_experiment["name"],
                    "hyperparameters": winning_experiment["best_params"],
                    "cv_metrics": winning_experiment["best_cv_metrics"],
                    "test_metrics": optimized_test,
                },
                "experiments": experiment_results,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            LOGGER.info(
                "model_complete study=%s model=%s duration_seconds=%.3f "
                "baseline_recall=%.4f optimized_recall=%.4f",
                dataset,
                model_name,
                study_result["models"][model_name]["elapsed_seconds"],
                baseline_test["recall"],
                optimized_test["recall"],
            )

        output["studies"][dataset] = study_result
    return output


def save_results(results: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)


def configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
