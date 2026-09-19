"""Permutation importance and optional SHAP summaries."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FIGURE_DIR, RANDOM_SEED, REPORTS_DIR, TARGET_PREFIX, ensure_data_dirs
from src.features.engineering import prepare_horizon_frame
from src.logging_utils import get_logger
from src.modeling.evaluation import walk_forward_splits
from src.modeling.features import feature_columns
from src.modeling.training import fit_xgboost, load_final_dataset

logger = get_logger(__name__)


def permutation_importance_table(
    df: pd.DataFrame | None = None,
    horizon: int = 1,
    n_repeats: int = 8,
) -> pd.DataFrame:
    """Permutation importance on the last walk-forward test fold (MAE decrease)."""
    ensure_data_dirs()
    dataset = df if df is not None else load_final_dataset()
    ready = prepare_horizon_frame(dataset, horizon)
    splits = list(walk_forward_splits(ready, horizon))
    if not splits:
        raise RuntimeError("No walk-forward folds available for interpretability.")
    _, train_df, test_df = splits[-1]
    target_col = f"{TARGET_PREFIX}{horizon}"
    cols = feature_columns(ready)
    model = fit_xgboost(train_df, cols, target_col)
    result = permutation_importance(
        model,
        test_df[cols],
        test_df[target_col],
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    table.to_csv(REPORTS_DIR / f"permutation_importance_h{horizon}.csv", index=False)
    logger.info("Top permutation importances:\n%s", table.head(12).to_string(index=False))
    return table


def shap_summary_plot(df: pd.DataFrame | None = None, horizon: int = 1, max_samples: int = 400) -> str | None:
    """Write a SHAP beeswarm if the shap package is installed."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        logger.warning("shap is not installed; skipping SHAP plot.")
        return None

    ensure_data_dirs()
    dataset = df if df is not None else load_final_dataset()
    ready = prepare_horizon_frame(dataset, horizon)
    splits = list(walk_forward_splits(ready, horizon))
    if not splits:
        return None
    _, train_df, test_df = splits[-1]
    target_col = f"{TARGET_PREFIX}{horizon}"
    cols = feature_columns(ready)
    model = fit_xgboost(train_df, cols, target_col)
    sample = test_df[cols].sample(n=min(max_samples, len(test_df)), random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    path = FIGURE_DIR / f"shap_summary_h{horizon}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Wrote SHAP summary to %s", path)
    return str(path)


if __name__ == "__main__":
    permutation_importance_table()
    shap_summary_plot()
