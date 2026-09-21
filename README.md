# West Africa Malaria and Climate Forecasting

Reproducible pipeline for **one-year-ahead forecasting of *Plasmodium falciparum* incidence at district level** (GADM Admin-2 units) in Togo, Benin, Ghana and Burkina Faso, using ERA5 climate predictors and Malaria Atlas Project (MAP) incidence surfaces.

> **Read this first.** The target variable is *modelled* incidence: MAP raster surfaces aggregated per district. It is **not** observed case counts (no DHIS2 / SNIS / PNLP data are used). MAP surfaces are themselves estimated with environmental covariates and are very smooth in time, so results here describe how well climate-informed models extrapolate MAP surfaces, not how well they predict clinical malaria. See [Limitations](#limitations).

## What this repository does

1. Extracts district-level annual incidence from MAP GeoTIFF rasters (zonal statistics over GADM Admin-2 polygons).
2. Downloads daily ERA5 reanalysis via the Open-Meteo Archive API and aggregates it to calendar-year climate variables per district.
3. Merges both sources on `district_id` and `year` (one-to-one, validated).
4. Builds leakage-safe lagged features (whitelist-based selection).
5. Compares machine-learning models with several autoregressive baselines under expanding-window walk-forward validation.
6. Quantifies differences with a paired, district-level block bootstrap.

## Method

**Panel.** One row per district-year. `district_id` is the GADM `GID_2` identifier; the merge key is `(district_id, year)`.

**Climate variables** (annual, per district): total precipitation, rainy-season precipitation (May–October), mean / max / min temperature, mean relative humidity, precipitation anomaly. ERA5 is sampled at one point per district (its centroid), not as a zonal mean.

**Features.** All predictors for year *t* use information from year *t−1* or earlier:

- lags 1 and 2 of every climate variable;
- rolling means (2 and 3 years) computed on the series shifted by one year;
- lags of the target incidence and the previous-year change `drift = y(t−1) − y(t−2)`;
- a hydro-thermal index built from *t−1* climate.

Columns are selected with a **whitelist** (`_lag_<k>`, `_roll_<k>y`, plus the two explicit indices above). The target and its min / max proxies are never eligible.

**Target and reconstruction.** Models predict the annual change `Δ(t) = y(t) − y(t−1)`. The incidence forecast is `ŷ(t) = max(0, y(t−1) + Δ̂(t))`.

**Baselines.**

| Name | Definition |
| --- | --- |
| Persistence | `ŷ(t) = y(t−1)` |
| Drift persistence | `ŷ(t) = y(t−1) + (y(t−1) − y(t−2))` |
| Damped drift | `ŷ(t) = y(t−1) + φ·(y(t−1) − y(t−2))`, φ ∈ [0, 1] fitted by least squares on the training years |
| District trend (3 years) | `ŷ(t) = y(t−1) + (y(t−1) − y(t−3)) / 2` |
| District mean | mean incidence of the district over the training years |

**Models.** Random Forest (150 trees, depth 6), XGBoost (100 trees, depth 3, learning rate 0.03), LightGBM (100 trees, depth 3, learning rate 0.03). Seed 42. Hyperparameters are set manually in `src/modeling/training.py`; they are not selected by nested validation.

**Validation.** Expanding-window walk-forward over the last `WALK_FORWARD_TEST_YEARS` years available (`src/config.py`). For each test year, models are trained on all earlier years only. Out-of-sample forecasts from all folds are pooled before computing metrics.

**Metrics and uncertainty.** RMSE, MAE and R² on incidence levels; skill = `1 − RMSE / RMSE_persistence`. The MAE difference against persistence is reported with a 95 % percentile interval from a paired block bootstrap that resamples whole districts (1,000 resamples). An interval that excludes 0 is not a p-value.

## Results

Pooled out-of-sample forecasts (`reports/model_training_metrics.json`). Skill and ΔMAE are relative to **naive persistence**.

| Model | RMSE | MAE | R² | Skill vs. persistence (RMSE) | ΔMAE vs. persistence [95 % CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| Persistence | 0.0246 | 0.0190 | 0.9254 | reference | reference |
| Drift persistence | 0.0192 | 0.0151 | 0.9542 | +21.7 % | −0.00393 [−0.00446, −0.00340] |
| Damped drift | 0.0176 | 0.0139 | 0.9616 | +28.2 % | −0.00509 [−0.00557, −0.00461] |
| District trend (3 years) | 0.0245 | 0.0190 | 0.9255 | +0.1 % | −0.00002 [−0.00062, +0.00056] |
| District mean | 0.1212 | 0.1071 | −0.8145 | −393.1 % | +0.08811 [+0.08383, +0.09206] |
| Random Forest (Δ target) | 0.0183 | 0.0145 | 0.9585 | +25.5 % | −0.00454 [−0.00514, −0.00397] |
| XGBoost (Δ target) | 0.0176 | 0.0137 | 0.9618 | +28.5 % | −0.00529 [−0.00583, −0.00477] |
| LightGBM (Δ target) | 0.0174 | 0.0136 | 0.9625 | +29.1 % | −0.00540 [−0.00594, −0.00489] |

**How to read this table.**

- Every drift-based baseline and every ML model improves on naive persistence; the 3-year district trend is indistinguishable from it; the district mean is far worse.
- The strongest non-ML baseline (damped drift) already reaches about 28 % RMSE skill, close to the best ML model (about 29 %). The gap between the ML models and damped drift is small, and **a paired comparison of the ML models against damped drift is not reported yet**. This table therefore does not demonstrate that the ML models outperform a strong baseline.
- R² is computed on incidence levels and is dominated by between-district variance; it says little about the quality of year-to-year forecasts.

## Limitations

- **Target is modelled, not observed.** MAP incidence is derived from geostatistical models that use environmental covariates, and it is smooth in time. Skill on this target may overstate skill on observed cases.
- **Publication lag.** MAP surfaces are released with a delay, so a one-year-ahead forecast built from `y(t−1)` is a retrospective experiment, not an operational forecast.
- **Administrative units.** GADM Admin-2 polygons are not the official health districts used by national health ministries.
- **Climate aggregation.** ERA5 is sampled at one point per unit; the rainy season is fixed to May–October for all four countries; the precipitation anomaly is computed relative to the district mean over the full 2000–2025 period.
- **Spatial dependence.** The bootstrap resamples districts independently and does not account for spatial autocorrelation. No spatially blocked cross-validation is performed.
- **No tuning protocol.** Hyperparameters are set manually and were not selected by nested time-series validation.
- **No intervention covariates.** Bed-net, treatment and other intervention coverage, which drive long-term trends, are not included.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── config.py
│   ├── logging_utils.py
│   ├── data/
│   │   ├── extract_district_data.py   # MAP rasters -> district-year incidence
│   │   ├── climate_data.py            # ERA5 via Open-Meteo -> district-year climate
│   │   └── provenance.py
│   ├── features/
│   │   └── build_features.py          # merge on (district_id, year)
│   └── modeling/
│       ├── features.py                # lag / rolling features, whitelist
│       ├── baselines.py
│       ├── metrics.py                 # metrics, paired block bootstrap
│       └── training.py                # walk-forward evaluation
├── tests/
│   └── test_leakage.py
├── data/
│   ├── raw/                           # boundaries, MAP GeoTIFF rasters (not versioned)
│   └── processed/
└── reports/
    └── model_training_metrics.json
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

Data preparation (MAP GeoTIFF files must be placed manually in `data/raw/map_rasters/`, one file per year with the year in the file name):

```bash
python -m src.data.extract_district_data
python -m src.data.climate_data
python -m src.features.build_features
```

Training and evaluation:

```bash
python -m src.modeling.training
```

Metrics are written to `reports/model_training_metrics.json`.

## Data sources

| Data | Source |
| --- | --- |
| Incidence | Malaria Atlas Project *P. falciparum* incidence rasters (GeoTIFF, downloaded separately) |
| Climate | ERA5 reanalysis (Hersbach et al., 2020) accessed through the Open-Meteo Archive API |
| Boundaries | GADM 4.1, Admin-2 level |

Licences and terms of use of each source apply to the derived data; see `DATA_SOURCES.md` before redistributing anything.

## License

The code is distributed under the Apache License 2.0. See [LICENSE](LICENSE).