"""Model training and walk-forward evaluation pipeline for district-level malaria forecasting."""

from __future__ import annotations

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.config import (
    DISTRICT_ID_COL,
    INCIDENCE_COL,
    RANDOM_SEED,
    REPORTS_DIR,
    WALK_FORWARD_TEST_YEARS,
    YEAR_COL,
)
from src.logging_utils import get_logger
from src.modeling.baselines import (
    DampedDriftBaseline,
    DistrictMeanBaseline,
    DriftPersistenceBaseline,
    LocalDistrictTrendBaseline,
    PersistenceBaseline,
)
from src.modeling.features import feature_columns, prepare_annual_features
from src.modeling.metrics import compute_metrics, paired_block_bootstrap_diff

logger = get_logger(__name__)


def train_and_evaluate() -> dict[str, dict[str, float | str | bool]]:
    """Train ML models on incidence delta (Delta_t) using expanding window cross-validation.
    
    Evaluates ML models and baselines across multiple test years without data leakage.
    Reconstructs forecasts as: y_hat = y_{t-1} + Delta_hat.
    """
    logger.info("Preparing annual features dataset...")
    df = prepare_annual_features()

    if "target_delta" not in df.columns:
        df["target_delta"] = df[INCIDENCE_COL] - df[f"{INCIDENCE_COL}_lag_1"]

    years = sorted(df[YEAR_COL].unique())
    if len(years) <= WALK_FORWARD_TEST_YEARS:
        raise ValueError(
            f"Not enough temporal depth ({len(years)} years) for walk-forward evaluation "
            f"with {WALK_FORWARD_TEST_YEARS} test years."
        )

    test_years = years[-WALK_FORWARD_TEST_YEARS:]
    logger.info("Walk-forward evaluation across test years: %s", test_years)

    X_cols = feature_columns(df)
    logger.info("Selected %s predictive features via whitelist.", len(X_cols))

    all_eval_rows: list[pd.DataFrame] = []

    # Iterate over expanding windows
    for test_year in test_years:
        train_df = df[df[YEAR_COL] < test_year].copy()
        test_df = df[df[YEAR_COL] == test_year].copy()

        if test_df.empty:
            continue

        X_train = train_df[X_cols]
        y_train_delta = train_df["target_delta"]

        X_test = test_df[X_cols]
        y_test_true = test_df[INCIDENCE_COL].to_numpy(dtype=float)
        y_test_lag1 = test_df[f"{INCIDENCE_COL}_lag_1"].to_numpy(dtype=float)

        fold_eval = test_df[[DISTRICT_ID_COL, YEAR_COL, INCIDENCE_COL, f"{INCIDENCE_COL}_lag_1"]].copy()

        # 1. Baseline Predictions
        pers = PersistenceBaseline().fit(train_df)
        pred_pers = pers.predict(test_df)
        fold_eval["pred_Persistence"] = pred_pers
        fold_eval["err_Persistence"] = y_test_true - pred_pers

        drift = DriftPersistenceBaseline().fit(train_df)
        pred_drift = drift.predict(test_df)
        fold_eval["pred_Drift_Persistence"] = pred_drift
        fold_eval["err_Drift_Persistence"] = y_test_true - pred_drift

        damped = DampedDriftBaseline().fit(train_df)
        pred_damped = damped.predict(test_df)
        fold_eval["pred_Damped_Drift"] = pred_damped
        fold_eval["err_Damped_Drift"] = y_test_true - pred_damped

        dist_mean = DistrictMeanBaseline().fit(train_df)
        pred_mean = dist_mean.predict(test_df)
        fold_eval["pred_District_Mean"] = pred_mean
        fold_eval["err_District_Mean"] = y_test_true - pred_mean

        dist_trend = LocalDistrictTrendBaseline().fit(train_df)
        pred_trend = dist_trend.predict(test_df)
        fold_eval["pred_District_Trend"] = pred_trend
        fold_eval["err_District_Trend"] = y_test_true - pred_trend

        # 2. Machine Learning Regressors
        ml_models = {
            "Random_Forest": RandomForestRegressor(
                n_estimators=150,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbosity=-1,
            ),
        }

        for name, model in ml_models.items():
            model.fit(X_train, y_train_delta)
            delta_pred = model.predict(X_test)
            y_pred = np.clip(y_test_lag1 + delta_pred, a_min=0.0, a_max=None)

            fold_eval[f"pred_{name}"] = y_pred
            fold_eval[f"err_{name}"] = y_test_true - y_pred

        all_eval_rows.append(fold_eval)

    # Combine all out-of-sample forecast folds
    full_eval_df = pd.concat(all_eval_rows, ignore_index=True)
    y_true_all = full_eval_df[INCIDENCE_COL].to_numpy(dtype=float)

    # Compute reference baseline RMSE for normalized metrics
    pers_metrics = compute_metrics(y_true_all, full_eval_df["pred_Persistence"].to_numpy())
    ref_rmse = pers_metrics["rmse"]

    results: dict[str, dict[str, float | str | bool]] = {
        "Persistence": pers_metrics,
    }

    model_names = [
        "Drift_Persistence",
        "Damped_Drift",
        "District_Mean",
        "District_Trend",
        "Random_Forest",
        "XGBoost",
        "LightGBM",
    ]

    for name in model_names:
        y_pred = full_eval_df[f"pred_{name}"].to_numpy()
        metrics = compute_metrics(y_true_all, y_pred, ref_rmse)

        # Paired District Block Bootstrap against Persistence
        obs_diff, (ci_low, ci_high) = paired_block_bootstrap_diff(
            full_eval_df,
            err_model=f"err_{name}",
            err_baseline="err_Persistence",
            group_col=DISTRICT_ID_COL,
            n_boot=1000,
            seed=RANDOM_SEED,
        )

        metrics["mae_diff_vs_pers"] = float(obs_diff)
        metrics["mae_diff_ci_95"] = f"[{ci_low:.5f}, {ci_high:.5f}]"
        metrics["statistically_significant"] = bool(ci_low > 0 or ci_high < 0)

        results[name] = metrics
        logger.info(
            "[%s] RMSE=%.4f | MAE=%.4f | R2=%.4f | MAE_Diff=%.5f (CI: %s)",
            name,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
            obs_diff,
            metrics["mae_diff_ci_95"],
        )

    # Save detailed evaluation report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "model_training_metrics.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved walk-forward model training metrics to %s", report_file)
    return results


if __name__ == "__main__":
    train_and_evaluate()