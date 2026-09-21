"""Write a neutral, machine-readable manifest for the annual district dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import (
    CLIMATE_API_URL,
    CLIMATE_PROVIDER,
    CLIMATE_REANALYSIS,
    CLIMATE_TIMEZONE,
    DATA_PROCESSED,
    RANDOM_SEED,
    ensure_data_dirs,
)


def write_data_manifest(extra: dict | None = None) -> None:
    """Persist a manifest describing the annual climate and incidence source material."""
    ensure_data_dirs()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "boundaries": {
            "dataset": "GADM Admin-2",
            "unit_definition": "District polygons used to identify annual district identifiers.",
            "license_note": "Review the applicable GADM licensing terms before redistribution.",
        },
        "climate": {
            "provider": CLIMATE_PROVIDER,
            "reanalysis": CLIMATE_REANALYSIS,
            "endpoint": CLIMATE_API_URL,
            "timezone": CLIMATE_TIMEZONE,
            "temporal_resolution": "annual aggregate",
            "notes": "Annual district summaries are derived from daily ERA5-like archive data.",
        },
        "incidence": {
            "type": "annual_district_panel",
            "file": "west_africa_malaria_incidence_2000_2025.csv",
            "notes": [
                "This repository uses the annual district panel available in data/processed.",
                "The pipeline is designed for reproducible modeling, not for claiming observational inference beyond the provided dataset.",
            ],
        },
        "quality": {
            "leakage_policy": "Contemporaneous climate and target data are excluded from the year t feature matrix.",
            "target_definition": "Annual delta: pf_incidence_rate - lagged pf_incidence_rate.",
        },
    }
    if extra:
        payload.update(extra)
    path = DATA_PROCESSED / "data_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
