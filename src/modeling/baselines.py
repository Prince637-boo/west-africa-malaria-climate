"""Baseline models for district-level malaria forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DISTRICT_ID_COL, INCIDENCE_COL


class BasePredictor:
    """Base class for baseline predictors."""

    def fit(self, train_df: pd.DataFrame, target_col: str = INCIDENCE_COL) -> BasePredictor:
        """Fit model parameters on training data if applicable."""
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict target values for testing data."""
        raise NotImplementedError


class PersistenceBaseline(BasePredictor):
    """Naive Persistence Baseline: predicts y(t) = y(t-1)."""

    def fit(self, train_df: pd.DataFrame, target_col: str = INCIDENCE_COL) -> PersistenceBaseline:
        self.target_col = target_col
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        lag1_col = f"{self.target_col}_lag_1"
        if lag1_col not in test_df.columns:
            raise KeyError(f"Required lag column '{lag1_col}' is missing from test_df.")
        return test_df[lag1_col].to_numpy(dtype=float)


class DriftPersistenceBaseline(BasePredictor):
    """Persistence with Linear Drift: predicts y(t) = y(t-1) + (y(t-1) - y(t-2))."""

    def fit(self, train_df: pd.DataFrame, target_col: str = INCIDENCE_COL) -> DriftPersistenceBaseline:
        self.target_col = target_col
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        lag1_col = f"{self.target_col}_lag_1"
        lag2_col = f"{self.target_col}_lag_2"
        drift_col = f"{self.target_col}_drift"

        if lag1_col not in test_df.columns:
            raise KeyError(f"Required lag column '{lag1_col}' is missing from test_df.")

        if drift_col in test_df.columns:
            drift = test_df[drift_col]
        elif lag2_col in test_df.columns:
            drift = test_df[lag1_col] - test_df[lag2_col]
        else:
            raise KeyError(f"Neither '{drift_col}' nor '{lag2_col}' found in test_df.")

        pred = test_df[lag1_col] + drift
        return np.clip(pred.to_numpy(dtype=float), a_min=0.0, a_max=None)


class DampedDriftBaseline(BasePredictor):
    """Persistence with Damped Drift: predicts y(t) = y(t-1) + phi * drift(t)."""

    def __init__(self) -> None:
        self.target_col = INCIDENCE_COL
        self.phi: float = 1.0

    def fit(self, train_df: pd.DataFrame, target_col: str = INCIDENCE_COL) -> DampedDriftBaseline:
        self.target_col = target_col
        lag1_col = f"{self.target_col}_lag_1"
        drift_col = f"{self.target_col}_drift"

        if lag1_col in train_df.columns and drift_col in train_df.columns:
            y = train_df[self.target_col].to_numpy(dtype=float)
            y_prev = train_df[lag1_col].to_numpy(dtype=float)
            drift = train_df[drift_col].to_numpy(dtype=float)

            diff = y - y_prev
            mask = ~np.isnan(diff) & ~np.isnan(drift) & (drift != 0)
            if np.any(mask):
                # Fit damping factor phi via least squares clipped to [0, 1]
                self.phi = float(np.clip(np.dot(diff[mask], drift[mask]) / np.dot(drift[mask], drift[mask]), 0.0, 1.0))
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        lag1_col = f"{self.target_col}_lag_1"
        drift_col = f"{self.target_col}_drift"

        if lag1_col not in test_df.columns or drift_col not in test_df.columns:
            raise KeyError(f"Required columns '{lag1_col}' or '{drift_col}' missing from test_df.")

        pred = test_df[lag1_col] + self.phi * test_df[drift_col]
        return np.clip(pred.to_numpy(dtype=float), a_min=0.0, a_max=None)


class LocalDistrictTrendBaseline(BasePredictor):
    """Recent 3-year local trend baseline: y(t) = y(t-1) + (y(t-1) - y(t-3)) / 2."""

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self.target_col = INCIDENCE_COL

    def fit(self, train_df: pd.DataFrame, target_col: str = INCIDENCE_COL) -> LocalDistrictTrendBaseline:
        self.target_col = target_col
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        lag1_col = f"{self.target_col}_lag_1"
        lag3_col = f"{self.target_col}_lag_3"

        if lag1_col not in test_df.columns or lag3_col not in test_df.columns:
            raise KeyError(f"Required columns '{lag1_col}' and '{lag3_col}' missing from test_df.")

        trend_3y = (test_df[lag1_col] - test_df[lag3_col]) / 2.0
        pred = test_df[lag1_col] + trend_3y
        return np.clip(pred.to_numpy(dtype=float), a_min=0.0, a_max=None)


class DistrictMeanBaseline(BasePredictor):
    """Historical District Mean Baseline."""

    def __init__(self) -> None:
        self.means_: dict[str, float] = {}
        self.global_mean_: float = 0.0
        self.target_col = INCIDENCE_COL

    def fit(self, train_df: pd.DataFrame, target_col: str = INCIDENCE_COL) -> DistrictMeanBaseline:
        self.target_col = target_col
        self.global_mean_ = float(train_df[target_col].mean())
        grouped = train_df.groupby(DISTRICT_ID_COL)[target_col].mean()
        self.means_ = {str(k): float(v) for k, v in grouped.items()}
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.means_.get(str(d), self.global_mean_) for d in test_df[DISTRICT_ID_COL]],
            dtype=float,
        )