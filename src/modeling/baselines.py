"""Forecast baselines: climatology, seasonal naive, and lagged-incidence regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import INCIDENCE_COL
from src.modeling.features import autoregressive_feature_columns, feature_columns


def district_month_climatology(train_df: pd.DataFrame, origin_month_col: str = "month") -> pd.Series:
    """Mean incidence by district and calendar month of the *origin* is not the target month.

    Callers should pass a frame that already contains the target column and the
    calendar month of the target date.
    """
    return train_df.groupby(["district_id", origin_month_col])[INCIDENCE_COL].mean()


def predict_climatology(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_month_col: str,
    target_col: str,
) -> np.ndarray:
    """Predict using historical mean of the target calendar month in each district.

    The climatology is computed from observed incidence in the training period,
    grouped by district and calendar month (the month being forecast).
    """
    hist = train_df.copy()
    hist["_clim_month"] = hist["date"].dt.month
    means = hist.groupby(["district_id", "_clim_month"])[INCIDENCE_COL].mean()
    global_month = hist.groupby("_clim_month")[INCIDENCE_COL].mean()
    global_mean = hist[INCIDENCE_COL].mean()

    preds = []
    for _, row in test_df.iterrows():
        key = (row["district_id"], int(row[target_month_col]))
        if key in means.index:
            preds.append(float(means.loc[key]))
        elif int(row[target_month_col]) in global_month.index:
            preds.append(float(global_month.loc[int(row[target_month_col])]))
        else:
            preds.append(float(global_mean))
    _ = target_col
    return np.asarray(preds, dtype=float)


def predict_seasonal_naive(test_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Same calendar month last year, aligned to the origin row."""
    col = f"seasonal_naive_h{horizon}"
    if col not in test_df.columns:
        raise KeyError(f"Missing {col}; run feature engineering first.")
    return test_df[col].to_numpy(dtype=float)


def predict_global_mean(train_df: pd.DataFrame, n_test: int) -> np.ndarray:
    """Weak pooled-mean baseline (kept only as a straw man)."""
    return np.full(n_test, float(train_df[INCIDENCE_COL].mean()), dtype=float)


def fit_autoregressive_ridge(train_df: pd.DataFrame, target_col: str) -> Ridge:
    """Ridge on incidence lags only (hard operational competitor)."""
    cols = autoregressive_feature_columns(feature_columns(train_df))
    model = Ridge(alpha=1.0)
    model.fit(train_df[cols], train_df[target_col])
    model.feature_names_in_ = np.asarray(cols)
    return model


def predict_autoregressive_ridge(model: Ridge, test_df: pd.DataFrame) -> np.ndarray:
    cols = list(model.feature_names_in_)
    return model.predict(test_df[cols])
