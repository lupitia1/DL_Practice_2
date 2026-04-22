import os
import time
import zipfile
from io import BytesIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://mis.nyiso.com/public/csv/pal/"

OUTPUT_DIR = "newyork_system_operator_data_2021_2025"
PARQUET_DIR = os.path.join(OUTPUT_DIR, "monthly_parquet")
FINAL_CSV = os.path.join(OUTPUT_DIR, "nyiso_hourly_load.csv")

# Expected NYISO zones used in the practice
ZONE_COLUMNS = [
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
]

os.makedirs(PARQUET_DIR, exist_ok=True)


# ============================================================
# HTTP SESSION WITH RETRIES
# ============================================================

session = requests.Session()

retries = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_month(url, timeout=60):
    """
    Download one monthly ZIP file from the NYISO public site.
    """
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def process_zip(content):
    """
    Read all CSV files inside a monthly ZIP file, pivot them by zone,
    convert timestamps properly, resample to hourly frequency,
    and fill small gaps by interpolation.
    """
    dfs = []

    with zipfile.ZipFile(BytesIO(content)) as z:
        for file in z.namelist():
            if file.endswith(".csv"):
                with z.open(file) as f:
                    df = pd.read_csv(f)
                    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])
                    dfs.append(df)

    if not dfs:
        raise ValueError("No CSV files found inside ZIP file.")

    df = pd.concat(dfs, ignore_index=True)

    # Convert long format to wide format:
    # rows -> timestamps, columns -> zones, values -> load
    df_pivot = df.pivot_table(
        index="Time Stamp",
        columns="Name",
        values="Load",
        aggfunc="mean"
    ).sort_index()

    # The original timestamps correspond to New York local time.
    # We localize them carefully to handle DST transitions and
    # convert them to UTC to avoid duplicated / ambiguous timestamps.
    localized_index = df_pivot.index.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward"
    )

    mask = ~localized_index.isna()
    df_pivot = df_pivot[mask].copy()
    df_pivot.index = localized_index[mask].tz_convert("UTC")

    # Resample to hourly frequency
    df_hourly = df_pivot.resample("1h").mean().sort_index()

    # Fill remaining gaps
    df_hourly = df_hourly.interpolate(method="time").ffill().bfill()

    return df_hourly


# ============================================================
# DOWNLOAD AND PROCESS MONTHLY FILES
# ============================================================

for year in range(2021, 2026):
    for month in range(1, 13):
        ym = f"{year}{month:02d}"
        url = f"{BASE_URL}{ym}01pal_csv.zip"
        parquet_path = os.path.join(PARQUET_DIR, f"{ym}.parquet")

        if os.path.exists(parquet_path):
            print(f"Skipping {ym}, already processed")
            continue

        print(f"Processing {ym}...")

        try:
            content = download_month(url)
            df_hourly = process_zip(content)

            df_hourly.to_parquet(parquet_path)
            print(f"Finished {ym}, shape={df_hourly.shape}")

            # Gentle pause to avoid stressing the server
            time.sleep(1)

        except Exception as e:
            print(f"Failed {ym}: {e}")
            continue


# ============================================================
# REBUILD FINAL CSV FROM MONTHLY PARQUETS
# ============================================================

print("Rebuilding final CSV from monthly parquet files...")

parquet_files = sorted(
    f for f in os.listdir(PARQUET_DIR)
    if f.endswith(".parquet")
)

dfs = []
for fname in parquet_files:
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)
    dfs.append(df)

if not dfs:
    raise ValueError("No parquet files found. Nothing to rebuild.")

final_df = pd.concat(dfs).sort_index()

# Keep only the expected zones
missing = [c for c in ZONE_COLUMNS if c not in final_df.columns]
if missing:
    raise ValueError(f"Missing expected zones in final dataset: {missing}")

final_df = final_df[ZONE_COLUMNS].copy()

# Compute total load (target variable for the practice)
final_df["total_load"] = final_df.sum(axis=1)

# Make Time Stamp an explicit column
final_df = final_df.reset_index().rename(columns={"index": "Time Stamp"})

# Save final CSV
final_df.to_csv(FINAL_CSV, index=False)

print(f"Final CSV written: {FINAL_CSV}")
print(f"Final shape: {final_df.shape}")

# Quick sanity checks
print("\nFirst rows:")
print(final_df.head())

print("\nLast rows:")
print(final_df.tail())

print("\nRows per year:")
print(pd.to_datetime(final_df["Time Stamp"]).dt.year.value_counts().sort_index())

print("\nAll done!")
