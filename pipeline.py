from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_DB = ARTIFACT_DIR / "predictions.db"
TARGET = "price"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Phase 2 engineered train and held-out test data."""
    train = pd.read_csv(DATA_DIR / "train_data.csv")
    test = pd.read_csv(DATA_DIR / "test_data.csv")
    for name, frame in (("train", train), ("test", test)):
        if TARGET not in frame:
            raise ValueError(f"{name}_data.csv must contain a '{TARGET}' column")
    return train, test


def train_model(train: pd.DataFrame):
    """Tune candidate regressors with CV and return the lowest-MAE estimator."""
    x_train, y_train = train.drop(columns=TARGET), train[TARGET]
    candidates = {
        "ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]}),
        "random_forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [50], "max_depth": [10], "min_samples_split": [2, 5]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [50], "learning_rate": [0.05, 0.1], "max_depth": [3]},
        ),
    }
    best_name, best_search = None, None
    for name, (estimator, grid) in candidates.items():
        # Keep the pipeline reliable on constrained CI runners and local machines.
        search = GridSearchCV(estimator, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
        search.fit(x_train, y_train)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search
    return best_name, best_search.best_estimator_, best_search.best_params_, -best_search.best_score_


def evaluate(model, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    features, actual = test.drop(columns=TARGET), test[TARGET]
    predicted = model.predict(features)
    results = features.copy()
    results.insert(0, "prediction_id", range(1, len(results) + 1))
    results["actual_price"] = actual.to_numpy()
    results["predicted_price"] = predicted
    results["absolute_error"] = np.abs(actual.to_numpy() - predicted)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }
    return results, metrics


def save_predictions(predictions: pd.DataFrame) -> None:
    """Persist final inference output to a project-owned SQLite database."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(PREDICTIONS_DB) as connection:
        predictions.to_sql("model_predictions", connection, if_exists="replace", index=False)


def run_training() -> dict[str, float]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    # MLflow 3 requires a database-backed tracking store; SQLite keeps the
    # experiment history local, reproducible, and dependency-free.
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("used-car-price-estimation")
    train, test = load_data()
    with mlflow.start_run(run_name="train-and-evaluate"):
        model_name, model, parameters, cv_mae = train_model(train)
        predictions, metrics = evaluate(model, test)
        metrics["cv_mae"] = cv_mae
        metrics["train_rows"] = len(train)
        metrics["test_rows"] = len(test)
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps({"model": model_name, "parameters": parameters, **metrics}, indent=2), encoding="utf-8")
        save_predictions(predictions)
        mlflow.log_param("model", model_name)
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.sklearn.log_model(model, artifact_path="model")
    print(f"Selected model: {model_name}")
    print(f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved {len(predictions)} predictions to: {PREDICTIONS_DB.relative_to(ROOT)} (model_predictions table)")
    return metrics


def run_prediction() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model. Run `python pipeline.py train` first.")
    _, test = load_data()
    predictions, _ = evaluate(joblib.load(MODEL_PATH), test)
    save_predictions(predictions)
    print(f"Saved {len(predictions)} predictions to: {PREDICTIONS_DB.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "predict", "run"), default="run", nargs="?")
    command = parser.parse_args().command
    if command in ("train", "run"):
        run_training()
    else:
        run_prediction()
