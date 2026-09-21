# West Africa Malaria and Climate Forecasting

This repository builds a reproducible forecasting workflow for malaria incidence and climate signals across West African districts. The code is structured to support climate-informed forecasting, spatial validation, and feature engineering in a clean Python project.

The project is designed for real data work, not for synthetic-only demonstrations. The repository keeps the scientific pipeline explicit: data acquisition, preprocessing, feature generation, model training, evaluation, and reporting.

## Project goals

- aggregate climate and epidemiological information by district and time period
- create horizon-aware forecasting features for monthly or annual prediction targets
- compare baselines and machine-learning models under time-aware validation
- generate diagnostics for model quality, spatial error, and feature importance
- keep the workflow reproducible, testable, and easy to extend

## Repository layout

```text
west-africa-malaria-climate/
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_pipeline.py
├── DATA_SOURCES.md
├── CONTRIBUTING.md
├── LICENSE
├── src/
│   ├── config.py
│   ├── logging_utils.py
│   ├── data/
│   │   ├── climate_data.py
│   │   ├── download_data.py
│   │   ├── extract_district_data.py
│   │   ├── merge_datasets.py
│   │   ├── provenance.py
│   │   └── resample.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── feature_engine.py
│   ├── modeling/
│   │   ├── baselines.py
│   │   ├── evaluation.py
│   │   ├── features.py
│   │   ├── interpretability.py
│   │   ├── metrics.py
│   │   └── training.py
│   └── visualization/
│       └── spatial_prediction_map.py
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_forecast_integrity.py
│   ├── test_no_fake_map.py
│   └── test_spatial_prediction_map.py
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── notebooks/
└── .gitignore
```

## Suggested setup

```bash
cd west-africa-malaria-climate
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run the validation suite:

```bash
pytest -q
```

## Data flow

The project expects the following flow:

1. raw boundary and raster data are placed under `data/raw/`
2. climate data are downloaded and aggregated under `data/processed/`
3. epidemiological and climate datasets are merged
4. forecasting features are built and validated
5. model evaluation and figures are written to `reports/` and `notebooks/figures/`

A minimal local run is:

```bash
python run_pipeline.py
```

## Scientific notes

- Climate variables are treated as exogenous predictors and are aligned to the forecast origin.
- Forecast targets are built with explicit horizons and tested with temporal validation.
- Model comparisons should focus on out-of-sample performance rather than in-sample fit.
- Map outputs and error summaries should support interpretation, not be used as a substitute for full epidemiological validation.

## Cleaning and maintenance rules

- keep English-only naming for modules, functions, variables, and documentation
- avoid stale simulated-data references in the active code path
- keep local caches such as `data/processed/climate_cache/` and `.openmeteo_cache.sqlite` out of version control
- keep code connected to a single data schema rather than maintaining parallel legacy modules

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
