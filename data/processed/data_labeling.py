import os
import pandas as pd
import numpy as np

# -----------------------
# CONFIG
# -----------------------

DEMOGRAPHICS_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\demographics\demographics.xls"
GAIT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset"
OUTPUT_CSV = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\dataset_labels.csv"

# -----------------------
# LOAD DEMOGRAPHICS
# -----------------------

# MUST use read_excel for XLS files
df_demo = pd.read_excel(DEMOGRAPHICS_PATH)

# Clean column names
df_demo.columns = df_demo.columns.str.strip()

# Ensure numeric clinical variables
df_demo["HoehnYahr"] = pd.to_numeric(df_demo["HoehnYahr"], errors="coerce")
df_demo["UPDRS"] = pd.to_numeric(df_demo["UPDRS"], errors="coerce")


# -----------------------
# DETECT CONTROL SUBJECTS
# -----------------------

def is_control(row):
    # Typical pattern: GaCoXX = Control
    if "Co" in str(row["ID"]):
        return True
    # Sometimes demographics include Group column
    if "Group" in df_demo.columns and str(row["Group"]).strip().lower() == "co":
        return True
    return False


df_demo["IS_CONTROL"] = df_demo.apply(is_control, axis=1)

# Assign stage=0 for controls with NaN H&Y
df_demo.loc[df_demo["IS_CONTROL"] & df_demo["HoehnYahr"].isna(), "HoehnYahr"] = 0.0


# -----------------------
# COMBO-BASED SEVERITY LABELING
# -----------------------

def get_pd_label(hoehn, updrs, is_control_flag):
    if is_control_flag:
        return 0  # CONTROL CLASS

    # Mild PD
    if (1 <= hoehn <= 2.0) or (not np.isnan(updrs) and updrs < 20):
        return 1

    # Moderate PD
    if (hoehn == 2.5) or (not np.isnan(updrs) and 20 <= updrs <= 35):
        return 2

    # Severe PD
    if hoehn >= 3.0 or (not np.isnan(updrs) and updrs > 35):
        return 2




df_demo["LABEL"] = df_demo.apply(
    lambda r: get_pd_label(r["HoehnYahr"], r["UPDRS"], r["IS_CONTROL"]),
    axis=1
)

# -----------------------
# MAP SUBJECT → LABEL
# -----------------------

label_map = {row["ID"]: row["LABEL"] for _, row in df_demo.iterrows()}

# -----------------------
# SCAN GAIT FOLDER
# -----------------------

txt_files = [f for f in os.listdir(GAIT_FOLDER) if f.endswith(".txt")]


def extract_subject_id(filename):
    # Example:
    #   GaPt07_01.txt → GaPt07
    #   GaCo03_02.txt → GaCo03
    return filename.split("_")[0]


# -----------------------
# CREATE OUTPUT CSV
# -----------------------

records = []

for file in txt_files:
    subj_id = extract_subject_id(file)

    if subj_id not in label_map:
        print(f"WARNING: Subject ID {subj_id} not found in demographics.xls")
        continue

    records.append({
        "filename": file,
        "subject_id": subj_id,
        "label": label_map[subj_id]
    })

df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False)

print("\nLabeling complete.")
print(f"Saved dataset_labels.csv to:\n{OUTPUT_CSV}")
