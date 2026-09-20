import glob
import os
from pathlib import Path
import re
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RASTERS_DIR = DATA_RAW_DIR / "map_rasters"
BOUNDARIES_DIR = DATA_RAW_DIR / "boundaries"

# URLs des frontières Admin-2 pour les 4 pays
COUNTRIES = {
    "TGO": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_2.json",
    "BEN": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BEN_2.json",
    "GHA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GHA_2.json",
    "BFA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BFA_2.json",
}


def load_all_boundaries() -> gpd.GeoDataFrame:
  """Charge ou télécharge les frontières Admin-2 (districts) pour les 4 pays."""
  BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
  gdfs = []

  for iso, url in COUNTRIES.items():
    geojson_path = BOUNDARIES_DIR / f"{iso}_admin2.json"
    if not geojson_path.exists():
      print(f"Téléchargement des frontières Admin-2 pour {iso}...")
      gdf = gpd.read_file(url)
      gdf.to_file(geojson_path, driver="GeoJSON")
    else:
      gdf = gpd.read_file(geojson_path)

    gdf["iso3"] = iso
    gdf["district_id"] = gdf.get("GID_2", gdf.get("GID_1"))
    gdf["district_name"] = gdf.get("NAME_2", gdf.get("NAME_1"))
    gdf["region_name"] = gdf.get("NAME_1", "N/A")

    gdfs.append(
        gdf[[
            "iso3",
            "district_id",
            "district_name",
            "region_name",
            "geometry",
        ]]
    )

  combined_gdf = pd.concat(gdfs, ignore_index=True)
  return gpd.GeoDataFrame(combined_gdf, crs="EPSG:4326")


def process_map_rasters() -> pd.DataFrame:
  """Extrait les statistiques zonales des rasters GeoTIFF MAP locaux."""
  DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  gdf_districts = load_all_boundaries()

  raster_files = sorted(glob.glob(str(RASTERS_DIR / "*.tif")))
  if not raster_files:
    raise FileNotFoundError(
        f"Aucun fichier .tif trouvé dans {RASTERS_DIR}. Place tes GeoTIFF"
        " téléchargés à cet endroit."
    )

  print(f"Trouvé {len(raster_files)} rasters GeoTIFF à traiter.")
  records = []

  for tif_path in raster_files:
    filename = Path(tif_path).name

    # REGEX CORRIGÉE : Extraction des 4 chiffres situés juste avant la fin '.tif'
    match = re.search(r"_(\d{4})\.tif$", filename)
    if not match:
      match = re.search(r"(\d{4})\.tif$", filename)

    if not match:
      print(f"Nom de fichier ignoré (année non détectée) : {filename}")
      continue

    year = int(match.group(1))
    print(f"Traitement du raster pour l'année {year} : {filename}...")

    with rasterio.open(tif_path) as src:
      affine = src.transform
      array = src.read(1)
      nodata = src.nodata

      stats = zonal_stats(
          gdf_districts,
          array,
          affine=affine,
          stats=["mean", "min", "max", "std"],
          nodata=nodata,
      )

      for idx, row in gdf_districts.iterrows():
        mean_val = stats[idx]["mean"]
        min_val = stats[idx]["min"]
        max_val = stats[idx]["max"]

        records.append({
            "iso3": row["iso3"],
            "district_id": row["district_id"],
            "district_name": row["district_name"],
            "region": row["region_name"],
            "year": year,
            "pf_incidence_rate": (
                round(float(mean_val), 5) if mean_val is not None else None
            ),
            "pf_incidence_min": (
                round(float(min_val), 5) if min_val is not None else None
            ),
            "pf_incidence_max": (
                round(float(max_val), 5) if max_val is not None else None
            ),
        })

  df_out = pd.DataFrame(records)

  # Tri chronologique par pays, district et année
  df_out.sort_values(by=["iso3", "district_id", "year"], inplace=True)

  output_path = (
      DATA_PROCESSED_DIR / "west_africa_malaria_incidence_2000_2025.csv"
  )
  df_out.to_csv(output_path, index=False)
  print(f"Extraction terminée ! Fichier sauvegardé ({len(df_out)} lignes) -> {output_path}")
  return df_out


if __name__ == "__main__":
  process_map_rasters()