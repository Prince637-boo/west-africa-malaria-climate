# Methodological pipeline for climate-informed malaria incidence forecasting

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Prince637-boo/west-africa-malaria-climate/actions/workflows/ci.yml/badge.svg)](https://github.com/Prince637-boo/west-africa-malaria-climate/actions/workflows/ci.yml)

**This repository is a methods and software study.** The prediction target is a **simulated** monthly incidence panel. It is **not** Malaria Atlas Project (MAP) data, **not** DHIS2/SNIS surveillance, and **must not** be reported as malaria incidence in Togo.

MAP does not provide monthly incidence by health district. Its modelled products are typically annual rasters. Until DHIS2/SNIS (or an equivalent partnership with the national malaria programme) is wired in, every metric in this repo only shows whether the pipeline recovers a **known** seasonal and lagged-climate signal under honest validation.

## What the pipeline does

For each forecast origin month \(t\) it predicts simulated incidence at horizons **h = 1, 2, 3 months** using only information dated \(t\) or earlier:

- Open-Meteo ERA5 archive climate at one in-polygon representative point per GADM Admin-2 unit
- trailing climate lags and a 3-month rolling mean that ends at \(t\)
- lagged incidence (including \(y_t\))
- cyclic encodings of the origin month and of the **known** calendar month of \(t+h\)

It does **not** predict month \(t\) using the climate of month \(t\) as if that were a 1–3 month forecast.

## What it does not do

- It does not implement SARIMA, LightGBM-only papers, or spatiotemporal neural networks as the published model card (LightGBM is included as a tree baseline; there is no graph/CNN sequence model).
- It does not use Copernicus CDS / `cdsapi`. Climate is downloaded from the **Open-Meteo Archive API** (ERA5 as served by Open-Meteo).
- GADM Admin-2 polygons are **prefectures**, not official health districts. The code logs the actual polygon count; it is not “40 health districts” unless that happens to match GADM.
- Climate is **not** a zonal mean over the polygon.
- A single pooled \(R^2\) is not treated as proof of outbreak prediction. Reports include MAE, RMSE, skill versus district-month climatology, bootstrap intervals, per-district and per-month MAE, leave-one-district-out, and ablations.

## Repository structure

```text
west-africa-malaria-climate/
├── src/config.py                 # Paths, seed, horizons, date range
├── src/data/download_data.py     # GADM Admin-2 download
├── src/data/climate_data.py      # Open-Meteo ERA5 point extraction
├── src/data/simulate_incidence.py
├── src/data/merge_datasets.py
├── src/features/engineering.py   # Horizon-aware lags and targets
├── src/modeling/                 # Features, baselines, training, evaluation
├── src/visualization/            # Error map and district time series
├── tests/                        # Leakage, merge, split, and map tests
├── DATA_SOURCES.md               # Versions, licenses, spatial definitions
├── requirements.txt              # Minimum versions
├── requirements.lock             # Pinned tree (uv pip compile)
└── run_pipeline.py               # Local evaluation after climate download
```

## Installation

```bash
git clone https://github.com/Prince637-boo/west-africa-malaria-climate.git
cd west-africa-malaria-climate
pip install uv
uv venv
source .venv/bin/activate
uv pip sync requirements.lock
# or: uv pip install -r requirements.txt
```

The global random seed is `42` (`src.config.RANDOM_SEED`).

## Data pipeline

```bash
python -m src.data.download_data
python -m src.data.climate_data
python -m src.data.simulate_incidence
python -m src.data.merge_datasets
python -m src.features.engineering
python -m src.modeling.evaluation
python -m src.modeling.interpretability
python -m src.visualization.spatial_prediction_map
```

Or, after the climate file exists:

```bash
python run_pipeline.py
```

Default climate window: **2015-01-01 to 2023-12-31**. Re-download if you only have a shorter cached extract. Even nine years of *simulated* monthly series is a methods experiment, not an epidemiological conclusion.

Manifest: `data/processed/data_manifest.json`. Details: [DATA_SOURCES.md](DATA_SOURCES.md).

## Models and validation

Walk-forward folds use an expanding training window and a 12-month test block defined on the **target** date \(t+h\), not the origin date.

Baselines (the global mean is kept only as a weak straw man):

- district-by-calendar-month climatology
- seasonal naive (\(y_{t+h-12}\))
- ridge regression on incidence lags

Comparators: Random Forest, XGBoost (small inner time-series grid on training origins only), LightGBM.

Spatial check: leave-one-district-out on the last walk-forward test year.

Interpretability: permutation importance (MAE); optional SHAP if `shap` is installed.

Tables are written under `reports/`. Maps and series under `notebooks/figures/`. Map files do **not** contain the caption “Figure 1”; put figure numbers in the manuscript caption. The map is a mean over the test year plus a **mean absolute error** panel; use the time-series figure for within-year dynamics.

Tree models do not extrapolate a linear trend beyond the training range. That limitation remains even with a longer panel.

## Tests and CI

```bash
python -m pytest tests/ -q
```

Unit tests do not require `data/` extracts. They check lag alignment, absence of lead targets in the feature list, inner-merge behaviour, temporal order of walk-forward splits, and map writing from a tiny GeoJSON.

## Using real incidence later

Replace `togo_malaria_incidence_simulated.csv` with monthly rates aligned to the **official health-district** geography, keep the horizon-aware features and walk-forward code, and rewrite the paper as an epidemiological study. Until then, cite this work only as a methodological pipeline on simulated data.

## Citation

```bibtex
@software{tchagodomou_malaria_methods_pipeline,
  title={Methodological pipeline for horizon-aware malaria incidence forecasting (simulated target)},
  author={Tchagodomou, Issa Prince},
  year={2026},
  note={Simulated incidence; not observational surveillance for Togo}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
