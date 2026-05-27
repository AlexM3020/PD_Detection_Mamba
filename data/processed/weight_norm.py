import os
import pandas as pd
import numpy as np

# -----------------------
# CONFIG
# -----------------------

DEMOGRAPHICS_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\demographics\demographics.xls"
GAIT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_oversampled"
OUTPUT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\normalized_gait_data_enlarged"

# -----------------------
# LOAD DEMOGRAPHICS AND CREATE WEIGHT MAPPING
# -----------------------

# Load demographics
df_demo = pd.read_excel(DEMOGRAPHICS_PATH)
df_demo.columns = df_demo.columns.str.strip()

# Create weight mapping (subject ID -> weight in kg)
weight_map = {}
for _, row in df_demo.iterrows():
    subject_id = str(row["ID"]).strip()
    weight = row["Weight (kg)"]

    # Only add to map if weight is available
    if not pd.isna(weight) and weight > 0:
        weight_map[subject_id] = weight

print(f"Loaded weight data for {len(weight_map)} subjects")


# -----------------------
# EXTRACT SUBJECT ID FROM FILENAME (HANDLING OVERSAMPLED FILES)
# -----------------------

def extract_subject_id(filename):
    # Handle regular files: GaPt07_01.txt → GaPt07
    # Handle oversampled files: GaCo02_01 (2).txt → GaCo02
    base_name = filename.split("_")[0]

    # Remove any "(2)" or similar oversampling indicators
    if " (" in base_name:
        base_name = base_name.split(" (")[0]

    return base_name


# -----------------------
# PROCESS AND NORMALIZE GAIT FILES
# -----------------------

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all text files
txt_files = [f for f in os.listdir(GAIT_FOLDER) if f.endswith(".txt")]

processed_count = 0
skipped_count = 0

for file in txt_files:
    file_path = os.path.join(GAIT_FOLDER, file)
    subject_id = extract_subject_id(file)

    # Skip if we don't have weight data for this subject
    if subject_id not in weight_map:
        print(f"WARNING: No weight data for subject {subject_id}, skipping file {file}")
        skipped_count += 1
        continue

    weight = weight_map[subject_id]

    try:
        # Read the gait data file
        # Assuming the file has multiple columns of numerical data
        df_gait = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # Normalize all columns by weight
        df_normalized = df_gait / weight

        # Save normalized data
        output_path = os.path.join(OUTPUT_FOLDER, file)
        df_normalized.to_csv(output_path, sep='\t', header=False, index=False)

        processed_count += 1

    except Exception as e:
        print(f"ERROR processing file {file}: {str(e)}")
        skipped_count += 1

print(f"\nNormalization complete!")
print(f"Processed: {processed_count} files")
print(f"Skipped: {skipped_count} files")
print(f"Output folder: {OUTPUT_FOLDER}")

# -----------------------
# CREATE SUMMARY CSV
# -----------------------

# Create a summary file showing which files were processed
summary_records = []
for file in txt_files:
    subject_id = extract_subject_id(file)
    has_weight = subject_id in weight_map
    weight_value = weight_map.get(subject_id, "N/A")

    summary_records.append({
        "filename": file,
        "subject_id": subject_id,
        "weight_available": has_weight,
        "weight_kg": weight_value,
        "processed": has_weight  # File was processed if weight was available
    })

df_summary = pd.DataFrame(summary_records)
summary_csv_path = os.path.join(OUTPUT_FOLDER, "normalization_summary.csv")
df_summary.to_csv(summary_csv_path, index=False)

print(f"Summary saved to: {summary_csv_path}")