# Data sources, versions, and licenses

All timestamps in `data/processed/data_manifest.json` are written when the local pipeline runs.

## Administrative boundaries

| Field | Value |
|---|---|
| Dataset | GADM |
| Version | 4.1 |
| Layer | `gadm41_TGO_2` (Admin-2) |
| URL | https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_2.json |
| What the polygons are | Prefectures / second-level administrative units |
| What they are **not** | Official Togo Ministry of Health / DHIS2 health districts |
| License | GADM license; review https://gadm.org/license.html before redistribution |

## Climate

| Field | Value |
|---|---|
| Access API | Open-Meteo Archive (`https://archive-api.open-meteo.com/v1/archive`) |
| Underlying reanalysis | ERA5, as served by Open-Meteo |
| Not used | Copernicus CDS `cdsapi` downloads |
| Native download | Daily precipitation and 2 m temperature |
| Analysis grain | Calendar-month sums (precipitation) and means (temperature) |
| Spatial support | One in-polygon representative point per Admin-2 unit (EPSG:32631 → WGS84) |
| Not implemented | Zonal mean over the polygon |
| Timezone | `Africa/Lome` |

## Incidence (target)

| Field | Value |
|---|---|
| Type | **Simulated** panel (`togo_malaria_incidence_simulated.csv`) |
| Not a source | Malaria Atlas Project monthly district incidence (MAP products are not used as the target; MAP incidence rasters are annual) |
| Not a source | DHIS2 / SNIS / PNLP case counts |
| Generator | `src/data/simulate_incidence.py` |
| Seed | `src.config.RANDOM_SEED` (42) |
| License | Simulated values are produced by this repository; they are not observational health data |

Coefficients of the data-generating process are stored in `simulate_incidence.DGP` and copied into the manifest.

## Replacing the target with observations

To move from a methods study to an epidemiological paper, replace the simulated file with monthly incidence (or cases + population) from DHIS2/SNIS, aligned to the official health-district geography, and keep the same horizon-aware feature and validation code.
