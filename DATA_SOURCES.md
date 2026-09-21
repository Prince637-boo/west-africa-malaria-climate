# Data Sources, Versions, and Licenses

This repository integrates spatial epidemiological targets with climate reanalysis and observation datasets across West Africa (Togo, Benin, Ghana, Burkina Faso) for the 2000–2025 period.

---

## 1. Administrative Boundaries

Administrative level 2 (districts / prefectures) boundaries are used to spatialise regional malaria incidence and climate metrics.

| Field | Value |
|---|---|
| **Dataset** | GADM (Database of Global Administrative Areas) |
| **Version** | 4.1 |
| **Layer** | `gadm41_{ISO3}_2` (Admin-2) for TGO, BEN, GHA, BFA |
| **Source URL** | `https://geodata.ucdavis.edu/gadm/gadm4.1/json/` |
| **Spatial Reference** | EPSG:4326 (WGS84) |
| **Polygon Definition** | Prefectures, districts, and municipalities (Admin-2) |
| **License** | Non-commercial academic use ([GADM License](https://gadm.org/license.html)) |

---

## 2. Malaria Incidence Target (Observational / MAP)

Malaria incidence data are extracted directly from official high-resolution raster datasets provided by the **Malaria Atlas Project (MAP)**.

| Field | Value |
|---|---|
| **Dataset** | Malaria Atlas Project — *Plasmodium falciparum* Incidence Rate |
| **Release / Model** | GBD 2025 (Global Burden of Disease 2025 Release) |
| **Format** | Annual GeoTIFF rasters (`2026_GBD2025_Global_Pf_Incidence_Rate_{YEAR}.tif`) |
| **Time Period** | 2000 – 2025 (Annual temporal resolution) |
| **Spatial Resolution** | ~5 km x 5 km (0.04166° grid) |
| **Extraction Method** | Polygonal Zonal Statistics (`mean`, `min`, `max`, `std`) per Admin-2 unit using `rasterstats` |
| **Target Variable** | `pf_incidence_rate` (Estimated cases per 1,000 population at risk) |
| **Output File** | `data/processed/west_africa_malaria_incidence_2000_2025.csv` |
| **License / Citation** | Open Access (Creative Commons Attribution 4.0 International — CC BY 4.0). Courtesy of Malaria Atlas Project / IHME. |

---

## 3. Climate Variables (ERA5 / Open-Meteo Archive)

Historical meteorological predictors integrated into the feature engineering pipeline.

| Field | Value |
|---|---|
| **Source API** | Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`) |
| **Underlying Dataset** | ECMWF ERA5 Reanalysis |
| **Variables** | Total Precipitation (mm), 2m Mean Temperature (°C), Relative Humidity (%) |
| **Temporal Grain** | Daily aggregated to monthly metrics |
| **Spatial Aggregation**| Zonal polygon means / centroid-based spatial joining for Admin-2 units |
| **Timezone** | `Africa/Lome` (UTC+0) |
| **License** | Copernicus Climate Change Service (C3S) Open Data License / Open-Meteo Terms |

---

## Data Provenance & Reproducibility Statement

- Raw GeoTIFF rasters (15 GB) are stored locally under `data/raw/map_rasters/` and backed up off-repository.
- GeoJSON administrative boundaries are cached in `data/raw/boundaries/`.
- All extraction logic is fully deterministic and versioned in `src/data/extract_district_data.py`.