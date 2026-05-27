# features_tsfresh.py

import os
import re
import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

# === CONFIGURATION ===
RAW_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_oversampled"
PROCESSED_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed"
TSFRESH_FEATURES_CSV = os.path.join(PROCESSED_DIR, "tsfresh_features.csv")

FS = 100  # Hz

# Correct sensor labels (16 GRF sensors)
SENSOR_LABELS = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"
]


def load_vgrf_data(raw_dir):
    """Loads VGRF signals and converts to long format suitable for TSFresh."""
    long_dfs = []
    subject_info = []  # Store subject metadata

    # Files to explicitly skip (non-signal files)
    skip_files = {
        "demographics.html", "demographics.txt", "demographics.xls",
        "format.txt", "gaitpd.png", "SHA256SUMS.txt"
    }

    for fname in os.listdir(raw_dir):
        # Skip only the specific non-data files
        if fname in skip_files:
            print(f"⚠️ Skipping non-signal file: {fname}")
            continue

        # Skip non-text files
        if not fname.endswith('.txt'):
            print(f"⚠️ Skipping non-text file: {fname}")
            continue

        file_path = os.path.join(raw_dir, fname)
        print(f"📄 Loading: {file_path}")

        try:
            # Replace delim_whitespace with sep='\s+' to fix the deprecation warning
            df = pd.read_csv(file_path, sep='\s+', header=None)
        except Exception as e:
            print(f"❌ Failed to load {fname}: {e}")
            continue

        # Check if file has at least 16 columns (VGRF sensors)
        if df.shape[1] < 16:
            print(f"⚠️ Skipping {fname} — does not contain 16 VGRF channels (has {df.shape[1]})")
            continue

        df = df.iloc[:, :16]  # Keep first 16 force sensors

        time_idx = np.arange(len(df))

        # Extract subject info from filename
        subject_id = len(subject_info)  # Use index as ID
        filename_no_ext = fname.replace('.txt', '')

        # Determine group and site from filename
        if filename_no_ext.startswith('GaCo'):
            group = 'control'
            site = 'Ga'
        elif filename_no_ext.startswith('GaPt'):
            group = 'patient'
            site = 'Ga'
        elif filename_no_ext.startswith('JuCo'):
            group = 'control'
            site = 'Ju'
        elif filename_no_ext.startswith('JuPt'):
            group = 'patient'
            site = 'Ju'
        elif filename_no_ext.startswith('SiCo'):
            group = 'control'
            site = 'Si'
        elif filename_no_ext.startswith('SiPt'):
            group = 'patient'
            site = 'Si'
        else:
            group = 'unknown'
            site = 'unknown'

        # Store subject metadata - ADDED filename WITH EXTENSION AND subject COLUMN
        subject_info.append({
            'id': subject_id,
            'group': group,
            'site': site,
            'filename': fname,  # Original filename with .txt extension
            'subject': filename_no_ext  # Subject ID without extension (GaCo01, GaPt03, etc.)
        })

        for i, sensor in enumerate(SENSOR_LABELS):
            temp_df = pd.DataFrame({
                "id": subject_id,
                "time": time_idx,
                "value": df.iloc[:, i],
                "sensor": sensor
            })
            long_dfs.append(temp_df)

    return pd.concat(long_dfs, ignore_index=True), subject_info


def extract_tsfresh_features(long_format_df, subject_info):
    """Runs TSFresh and saves results to CSV with group, site, id as first columns."""
    print("🚀 Running TSFresh feature extraction...")

    # Use proper feature extraction parameters
    features = extract_features(
        long_format_df,
        column_id="id",
        column_sort="time",
        column_kind="sensor",
        column_value="value",
        n_jobs=2,  # Reduced to prevent memory issues
        disable_progressbar=False,
        show_warnings=False,
        # Use the proper MinimalFCParameters object instead of string
        default_fc_parameters=MinimalFCParameters()
    )

    impute(features)

    # Reset index to get subject IDs as a column
    features = features.reset_index()
    features = features.rename(columns={'index': 'id'})

    # Create metadata DataFrame - ADDED filename AND subject COLUMNS
    metadata_df = pd.DataFrame(subject_info)[['id', 'group', 'site', 'filename', 'subject']]

    # Merge metadata with features
    features_with_metadata = pd.merge(metadata_df, features, on='id', how='right')

    # Reorder columns to have group, site, id, filename, subject as first five columns
    column_order = ['group', 'site', 'id', 'filename', 'subject'] + [col for col in features_with_metadata.columns if
                                              col not in ['group', 'site', 'id', 'filename', 'subject']]
    features_with_metadata = features_with_metadata[column_order]

    features_with_metadata.to_csv(TSFRESH_FEATURES_CSV, index=False)
    print(f"✅ TSFresh Features saved → {TSFRESH_FEATURES_CSV}")
    print(f"📊 Feature matrix shape: {features_with_metadata.shape}")

    return features_with_metadata


if __name__ == "__main__":
    print("📥 Loading VGRF long-format dataset...")
    df_long, subject_info = load_vgrf_data(RAW_DIR)

    if df_long.empty:
        print("❌ No data loaded! Check file filtering logic.")
        exit(1)

    print(f"✅ Data loaded: {len(df_long['id'].unique())} subjects, {len(df_long)} total observations")
    print("Starting TSFresh...")

    features_df = extract_tsfresh_features(df_long, subject_info)

    print("✅ Features are stored in processed folder.")