from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_final_dataset() -> pd.DataFrame:
    """Load the merged final modeling dataset for training and evaluation."""
    path = DATA_PROCESSED / "togo_final_modeling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run the preprocessing pipeline first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    return df


def train_and_predict(test_df: pd.DataFrame | None = None):
    """Train the forecasting model and predict 2023 district incidence values."""
    df = load_final_dataset()
    target_col = "malaria_incidence"
    exclude_cols = ["district_id", "district_name", "region", "date", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_df = df[df["date"] < "2023-01-01"].copy()
    eval_df = test_df if test_df is not None else df[df["date"] >= "2023-01-01"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_eval = eval_df[feature_cols]
    y_eval = eval_df[target_col]

    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)

    metrics = {
        "mae": mean_absolute_error(y_eval, predictions),
        "rmse": np.sqrt(mean_squared_error(y_eval, predictions)),
        "r2": r2_score(y_eval, predictions),
    }

    pred_frame = eval_df[["district_id", "district_name", "date"]].copy()
    pred_frame["predicted_incidence"] = predictions
    pred_frame["observed_incidence"] = y_eval.values

    return model, pred_frame, metrics


if __name__ == "__main__":
    _, predictions, metrics = train_and_predict()
    print("Model evaluation metrics:")
    print(metrics)
    print(predictions.head())
