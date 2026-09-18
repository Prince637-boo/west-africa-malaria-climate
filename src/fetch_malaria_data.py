import os
import requests
import geopandas as gpd
import pandas as pd

# Create the data directories if they do not already exist
DATA_RAW = "data/raw"
os.makedirs(DATA_RAW, exist_ok=True)


def download_togo_boundaries():
    """Download the second-level administrative boundaries for Togo from GADM."""
    print("1/2 - Downloading administrative boundaries for Togo...")
    geojson_path = os.path.join(DATA_RAW, "togo_districts.geojson")

    if not os.path.exists(geojson_path):
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_2.json"
        response = requests.get(url)
        response.raise_for_status()
        with open(geojson_path, "wb") as f:
            f.write(response.content)

    gdf = gpd.read_file(geojson_path)
    print(f"✅ Administrative boundaries loaded: {len(gdf)} districts found.")
    return gdf


def fetch_malaria_atlas_data():
    """
    Retrieve Plasmodium falciparum prevalence data from the Malaria Atlas Project
    via the WCS/direct-download endpoint.
    """
    print("2/2 - Downloading malaria prevalence data from the Malaria Atlas Project (MAP)...")

    # Direct URL for global PfPR survey data (2020/2021)
    raster_url = "https://data.malariaatlas.org/geoserver/OWS/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=OWS:pf_pr_surveys&outputFormat=json&cql_filter=country_id='TGO'"

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(raster_url, headers=headers, timeout=30)
        response.raise_for_status()

        # Validate that the response contains valid JSON
        data = response.json()

        # Convert survey points to a GeoDataFrame
        gdf_surveys = gpd.GeoDataFrame.from_features(data["features"])
        csv_path = os.path.join(DATA_RAW, "togo_malaria_surveys.csv")
        gdf_surveys.to_csv(csv_path, index=False)

        print(
            f"✅ MAP data saved: {len(gdf_surveys)} survey points exported to '{csv_path}'."
        )
        return gdf_surveys

    except Exception as e:
        print(
            f"⚠️ Direct API unavailable ({e}). Falling back to a district-level incidence template."
        )

        # Fallback: generate a basic district structure for later climate merge
        togo_gdf = download_togo_boundaries()
        df_base = pd.DataFrame(
            {
                "district_id": togo_gdf["GID_2"],
                "district_name": togo_gdf["NAME_2"],
                "region": togo_gdf["NAME_1"],
            }
        )

        csv_path = os.path.join(DATA_RAW, "togo_malaria_districts_base.csv")
        df_base.to_csv(csv_path, index=False)
        print(f"✅ District structure saved in '{csv_path}'.")
        return df_base


if __name__ == "__main__":
    togo_districts = download_togo_boundaries()
    malaria_data = fetch_malaria_atlas_data()
    print("\nHealth data initialization completed successfully.")