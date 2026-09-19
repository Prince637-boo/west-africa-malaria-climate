"""Evaluation metrics, including skill scores and bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import N_BOOTSTRAP, RANDOM_SEED


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Pooled MAE, RMSE, and R². R² is reported but is not the primary skill metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def skill_score(model_mae: float, reference_mae: float) -> float:
    """MAE skill relative to a reference (climatology). 1 is perfect, 0 matches the reference."""
    if reference_mae == 0:
        return float("nan")
    return 1.0 - model_mae / reference_mae


def bootstrap_mae_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile 95% interval for MAE by resampling test rows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_obs, n_obs)
        stats.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def grouped_mae(frame: pd.DataFrame, group_col: str, y_true: str, y_pred: str) -> pd.DataFrame:
    """MAE by district or by calendar month."""
    rows = []
    for key, part in frame.groupby(group_col, sort=True):
        rows.append(
            {
                group_col: key,
                "n": int(len(part)),
                "mae": float(mean_absolute_error(part[y_true], part[y_pred])),
            }
        )
    return pd.DataFrame(rows)
