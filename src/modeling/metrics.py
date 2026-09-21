"""Evaluation metrics and statistical significance testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, baseline_rmse: float | None = None
) -> dict[str, float]:
    """Compute standard regression metrics (RMSE, MAE, R2) and optional forecast skill score."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    if baseline_rmse is not None and baseline_rmse > 0:
        # Forecast Skill Score relative to persistence RMSE
        metrics["skill_vs_persistence"] = float(1.0 - (rmse / baseline_rmse))

    return metrics


def paired_block_bootstrap_diff(
    df: pd.DataFrame,
    err_model: str,
    err_baseline: str,
    group_col: str = "district_id",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute mean(|err_model|) - mean(|err_baseline|) with a district-level block bootstrap 95% CI.
    
    A negative difference indicates the candidate model outperforms the baseline.
    If the 95% confidence interval crosses zero, the performance difference is NOT 
    statistically significant at alpha = 0.05.
    """
    diff = df[err_model].abs() - df[err_baseline].abs()
    grouped = diff.groupby(df[group_col])

    block_sums = grouped.sum().to_numpy()
    block_counts = grouped.count().to_numpy()

    num_blocks = len(block_sums)
    if num_blocks == 0:
        raise ValueError("Cannot perform block bootstrap on an empty DataFrame or grouping.")

    rng = np.random.default_rng(seed)

    # Resample district blocks with replacement across bootstrap iterations
    sample_indices = rng.integers(0, num_blocks, size=(n_boot, num_blocks))
    boot_means = block_sums[sample_indices].sum(axis=1) / block_counts[sample_indices].sum(axis=1)

    obs_diff = float(diff.mean())
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    return obs_diff, (float(ci_lower), float(ci_upper))