# Spatiotemporal Malaria Outbreak Prediction using Climate & Satellite Data in West Africa

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-fast%20package%20manager-de5b40.svg)](https://github.com/astral-sh/uv)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Data Source: ERA5](https://img.shields.io/badge/Climate-ERA5%20%2F%20Copernicus-blue)](https://cds.climate.copernicus.eu/)
[![Data Source: MAP](https://img.shields.io/badge/Health-Malaria%20Atlas%20Project-green)](https://malariaatlas.org/)

## Abstract & Operational Objective

Malaria transmission in West Africa exhibits strong climate-driven seasonal patterns. However, operational early warning systems capable of predicting local outbreaks weeks in advance remain limited.

This repository hosts a reproducible, end-to-end spatiotemporal machine learning framework designed to forecast malaria incidence at the health-district level (Admin-2) in West Africa (initial pilot focus: **Togo**, 40 health districts). By combining climate reanalysis datasets (ERA5 / Open-Meteo) with epidemiological data from the **Malaria Atlas Project (MAP)** and GADM administrative boundaries, this tool provides a **1 to 3-month lead-time prediction system** to assist public health authorities in proactive resource allocation (e.g., ACT treatments, mosquito net distribution).

This research project is designed for submission to a peer-reviewed scientific journal and includes an interactive web dashboard for real-time risk visualization.

---

## Key Features

- **Spatial Granularity:** District-level analysis (Admin-2) using EPSG:32631 (UTM Zone 31N) projected centroids.
- **Climate Data Extraction:** High-resolution precipitation, mean, min, and max 2-meter air temperature series.
- **Lag Feature Engineering:** Computation of 1, 2, and 3-month temporal lags reflecting the biological mosquito-vector cycle (*Anopheles*) and *Plasmodium* incubation.
- **Reproducible Machine Learning Pipeline:** Strict time-series cross-validation (Walk-Forward Validation) comparing autoregressive baselines (SARIMA) against gradient boosted trees (XGBoost, LightGBM) and spatiotemporal neural architectures.
- **Interactive Decision Support Tool:** API backend (FastAPI) coupled with a user-facing dashboard for epidemic risk mapping.

---

## Repository Structure

```text
west-africa-malaria-climate/
│
├── .github/               # CI/CD workflows and branch protection configuration
├── data/                  # Excluded from version control via .gitignore
│   ├── raw/               # Raw geospatial and epidemiological data (GADM, MAP)
│   └── processed/         # Aligned monthly spatiotemporal climate-health datasets
│
├── src/                   # Core Python package modules
│   ├── fetch_malaria_data.py   # Ingestion of GADM boundaries & MAP data
│   ├── fetch_climate_data.py   # ERA5 climate data extraction with rate-limit handling
│   ├── features.py             # Feature engineering (lags, rolling averages, seasonality)
│   ├── models.py               # Model definitions & training pipelines
│   └── evaluation.py           # Time-series cross-validation & evaluation metrics
│
├── notebooks/             # Exploratory Data Analysis (EDA) and figure generation
├── app/                   # Web dashboard & API service (FastAPI / Streamlit)
├── tests/                 # Unit and integration tests
├── requirements.txt       # Project dependencies
├── requirements.lock      # Locked dependencies for full reproducibility (uv)
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # Apache License 2.0
└── README.md              # Project documentation
```

---

## Installation & Setup

This project uses **`uv`** for fast and reproducible dependency management.

### Prerequisites

- Python 3.10 or higher
- Git

### Quickstart

1. **Clone the repository:**

```bash
   git clone https://github.com/Prince637-boo/west-africa-malaria-climate.git
   cd west-africa-malaria-climate
```

2. **Set up the virtual environment with `uv`:**

```bash
   # Install uv if not already installed
   pip install uv

   # Create and activate the virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
   uv pip install -r requirements.txt
```

---

## Data Pipeline Execution

Run the data extraction and processing pipeline in order:

1. **Fetch administrative boundaries & epidemiological base data:**

```bash
   python src/fetch_malaria_data.py
```

2. **Extract ERA5 climate variables (precipitation & temperature):**

```bash
   python src/fetch_climate_data.py
```

3. **Generate lag features & prepare the modeling dataset:**

```bash
   python src/features.py
```

---

## Data Sources & Acknowledgements

This project relies on open-access epidemiological, climate, and administrative spatial data:

- **Epidemiological Data:** [Malaria Atlas Project (MAP)](https://malariaatlas.org/) — *Plasmodium falciparum* prevalence (PfPR) surveys and spatial incidence layers supported by the Telethon Kids Institute and the University of Oxford.
- **Climate & Reanalysis Data:** [Copernicus Climate Change Service (C3S)](https://cds.climate.copernicus.eu/) / [Open-Meteo API](https://open-meteo.com/) — ERA5-Land historical monthly and daily precipitation and 2-meter temperature reanalysis.
- **Administrative Boundaries:** [GADM (Global Administrative Areas)](https://gadm.org/) — High-resolution administrative shapefiles (Admin-2 / Health Districts).

*We acknowledge the open-science initiatives of ECMWF, MAP, and GADM for making these datasets publicly available to support global public health research.*
---

## Scientific Reproducibility & Citation

To ensure strict scientific reproducibility:

- All climate and geospatial data extraction functions are fully automated and deterministic.
- Model evaluations enforce temporal integrity to prevent data leakage (no future data used in training sets).
- Final code releases associated with publication manuscripts are archived on **Zenodo** with a persistent Digital Object Identifier (DOI).

If you use this codebase or methodology in your research, please cite:

```bibtex
@article{tchagodomou2026malaria,
  title={Spatiotemporal Prediction of Malaria Incidence using Climate Lags and Machine Learning in West Africa},
  author={Tchagodomou, Issa Prince},
  journal={In Preparation},
  year={2026}
}
```

---

## License & Contribution

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

Contributions via Pull Requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first, and ensure that unit tests pass and code adheres to PEP 8 standards before submitting a PR.