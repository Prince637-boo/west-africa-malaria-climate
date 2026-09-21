"""Visualization module for malaria forecasting model performance and feature importance."""

from __future__ import annotations

import json
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.config import DATA_PROCESSED, INCIDENCE_COL, RANDOM_SEED, REPORTS_DIR
from src.logging_utils import get_logger
from src.modeling.features import feature_columns, prepare_annual_features

logger = get_logger(__name__)

FIGURES_DIR = REPORTS_DIR / "figures"


def setup_style() -> None:
    """Configure publication-ready Matplotlib and Seaborn styles."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_model_performance_comparison(metrics_file: Path | None = None) -> None:
    """Generate a horizontal bar plot comparing MAE, RMSE, and Forecast Skill across models."""
    setup_style()
    if metrics_file is None:
        metrics_file = REPORTS_DIR / "model_training_metrics.json"

    if not metrics_file.exists():
        logger.error("Metrics file not found at %s. Skipping plot.", metrics_file)
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Exclude District_Mean for clean visualization due to extreme outlier scale
    filtered_data = {k: v for k, v in data.items() if k != "District_Mean"}
    df_metrics = pd.DataFrame(filtered_data).T.reset_index().rename(columns={"index": "Model"})
    df_metrics = df_metrics.sort_values(by="mae", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. MAE Comparison
    palette = ["#1f77b4" if "LightGBM" in m or "XGBoost" in m else "#aec7e8" for m in df_metrics["Model"]]
    sns.barplot(data=df_metrics, x="mae", y="Model", palette=palette, ax=axes[0])
    axes[0].set_title("Mean Absolute Error (MAE) - Lower is Better")
    axes[0].set_xlabel("MAE (Incidence Rate)")
    axes[0].set_ylabel("")

    for p in axes[0].patches:
        width = p.get_width()
        axes[0].annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )

    # 2. Skill Score vs Persistence
    df_skill = df_metrics.dropna(subset=["skill_vs_persistence"]).sort_values(by="skill_vs_persistence", ascending=False)
    skill_palette = ["#2ca02c" if v > 0.25 else "#98df8a" for v in df_skill["skill_vs_persistence"]]
    sns.barplot(data=df_skill, x="skill_vs_persistence", y="Model", palette=skill_palette, ax=axes[1])
    axes[1].set_title("Forecast Skill Score vs. Persistence - Higher is Better")
    axes[1].set_xlabel("Skill Score (1 - RMSE / RMSE_pers)")
    axes[1].set_ylabel("")

    for p in axes[1].patches:
        width = p.get_width()
        axes[1].annotate(
            f"{width*100:.1f}%",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )

    plt.suptitle("Walk-Forward Evaluation Performance (2018–2025)", y=1.02)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "model_performance_comparison.png"
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved performance comparison plot to %s", out_path)


def plot_shap_feature_importance() -> None:
    """Compute and plot SHAP feature importance values for the top LightGBM model."""
    setup_style()
    df = prepare_annual_features()
    
    if "target_delta" not in df.columns:
        df["target_delta"] = df[INCIDENCE_COL] - df[f"{INCIDENCE_COL}_lag_1"]

    X_cols = feature_columns(df)
    X = df[X_cols]
    y_delta = df["target_delta"]

    # Fit LightGBM on the entire dataset for feature importance insight
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )
    model.fit(X, y_delta)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=15, show=False)
    plt.title("Top 15 Feature Contributions to Malaria Incidence Delta (SHAP)", pad=15)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "shap_feature_importance.png"
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved SHAP feature importance plot to %s", out_path)


def generate_all_plots() -> None:
    """Entry point to generate all pipeline visual reports."""
    logger.info("Generating publication-ready figures...")
    plot_model_performance_comparison()
    plot_shap_feature_importance()
    logger.info("All visual figures generated successfully.")


if __name__ == "__main__":
    generate_all_plots()