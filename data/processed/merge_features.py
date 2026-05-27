# merge_features.py

import pandas as pd
import numpy as np
import os
import re

# === CONFIG ===
PROCESSED_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed"
TSFRESH_CSV = os.path.join(PROCESSED_DIR, "tsfresh_features.csv")
HANDCRAFTED_CSV = os.path.join(PROCESSED_DIR, "improved_gait_features.csv")
MERGED_CSV = os.path.join(PROCESSED_DIR, "merged_features.csv")


def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., 'GaCo02_01.txt' -> 'GaCo02')"""
    if pd.isna(filename):
        return None
    # Remove extension and trial number (_01, _02, etc.)
    base_name = filename.replace('.txt', '')
    # Keep only the subject part (GaCo02, GaPt03, etc.)
    subject_id = base_name.split('_')[0]
    return subject_id


def average_tsfresh_features(tsfresh_df):
    """Average TSFresh features per subject (matching handcrafted approach)"""
    print("🔄 Averaging TSFresh features per subject...")

    # Extract subject ID from filename
    tsfresh_df = tsfresh_df.copy()
    tsfresh_df['subject_id'] = tsfresh_df['filename'].apply(extract_subject_id)

    # Count trials per subject
    trial_counts = tsfresh_df['subject_id'].value_counts()
    print(f"📊 Found {len(trial_counts)} unique subjects:")
    for subject, count in trial_counts.head(10).items():
        print(f"   {subject}: {count} trials")
    if len(trial_counts) > 10:
        print(f"   ... and {len(trial_counts) - 10} more subjects")

    # Group by subject and average all feature columns
    feature_cols = [col for col in tsfresh_df.columns
                    if col not in ['group', 'site', 'id', 'filename', 'subject', 'subject_id']]

    # Aggregate features by subject
    aggregation_dict = {col: 'mean' for col in feature_cols}
    aggregation_dict.update({
        'group': 'first',
        'site': 'first',
        'filename': lambda x: ', '.join(x),  # Keep track of source files
        'subject': 'first'  # Use the first subject name (they should be consistent)
    })

    tsfresh_averaged = tsfresh_df.groupby('subject_id').agg(aggregation_dict).reset_index()

    print(f"✅ Averaged {len(tsfresh_df)} trials → {len(tsfresh_averaged)} subjects")
    return tsfresh_averaged


def merge_feature_sets(tsfresh_path, handcrafted_path):
    """Merge averaged TSFresh features with handcrafted features"""
    print("📥 Loading feature files...")

    # Load TSFresh features
    tsfresh_df = pd.read_csv(tsfresh_path)
    print(f"   TSFresh features: {tsfresh_df.shape}")

    # Load handcrafted features (already averaged)
    handcrafted_df = pd.read_csv(handcrafted_path)
    print(f"   Handcrafted features: {handcrafted_df.shape}")

    # Average TSFresh features per subject
    tsfresh_averaged = average_tsfresh_features(tsfresh_df)

    print("\n🔍 Checking subject overlap...")
    tsfresh_subjects = set(tsfresh_averaged['subject_id'])
    handcrafted_subjects = set(handcrafted_df['subject_id'])

    common_subjects = tsfresh_subjects.intersection(handcrafted_subjects)
    only_tsfresh = tsfresh_subjects - handcrafted_subjects
    only_handcrafted = handcrafted_subjects - tsfresh_subjects

    print(f"   Common subjects: {len(common_subjects)}")
    print(f"   Only in TSFresh: {len(only_tsfresh)}")
    print(f"   Only in Handcrafted: {len(only_handcrafted)}")

    if only_tsfresh:
        print(f"   ⚠️ Subjects only in TSFresh: {list(only_tsfresh)[:5]}...")
    if only_handcrafted:
        print(f"   ⚠️ Subjects only in Handcrafted: {list(only_handcrafted)[:5]}...")

    # Identify feature columns before merging
    tsfresh_feature_cols = [col for col in tsfresh_averaged.columns
                            if col not in ['subject_id', 'group', 'site', 'filename', 'subject']]
    handcrafted_feature_cols = [col for col in handcrafted_df.columns
                                if col not in ['subject_id', 'group', 'site', 'filename']]

    print(f"\n🔢 Feature counts before merging:")
    print(f"   TSFresh features: {len(tsfresh_feature_cols)}")
    print(f"   Handcrafted features: {len(handcrafted_feature_cols)}")

    # Merge on subject_id
    print("\n🔄 Merging feature sets...")
    merged_df = pd.merge(
        tsfresh_averaged,
        handcrafted_df,
        on=['subject_id', 'group', 'site'],
        how='inner',  # Keep only subjects present in both datasets
        suffixes=('_tsfresh', '_handcrafted')
    )

    print(f"✅ Merged features: {merged_df.shape}")
    print(f"   Final subjects: {len(merged_df)}")

    return merged_df, tsfresh_feature_cols, handcrafted_feature_cols


def count_features_accurately(merged_df, tsfresh_feature_cols, handcrafted_feature_cols):
    """Accurately count features in the merged dataset"""

    # Count TSFresh features (columns that came from TSFresh)
    tsfresh_features_in_merged = []
    for col in tsfresh_feature_cols:
        if f"{col}_tsfresh" in merged_df.columns:
            tsfresh_features_in_merged.append(f"{col}_tsfresh")
        elif col in merged_df.columns:
            tsfresh_features_in_merged.append(col)

    # Count Handcrafted features (columns that came from Handcrafted)
    handcrafted_features_in_merged = []
    for col in handcrafted_feature_cols:
        if f"{col}_handcrafted" in merged_df.columns:
            handcrafted_features_in_merged.append(f"{col}_handcrafted")
        elif col in merged_df.columns:
            handcrafted_features_in_merged.append(col)

    # Metadata columns (everything else)
    all_columns = set(merged_df.columns)
    feature_columns = set(tsfresh_features_in_merged + handcrafted_features_in_merged)
    metadata_columns = all_columns - feature_columns

    return (len(tsfresh_features_in_merged),
            len(handcrafted_features_in_merged),
            len(metadata_columns),
            tsfresh_features_in_merged,
            handcrafted_features_in_merged)


def main():
    print("🚀 Starting Feature Merging Pipeline")
    print("=" * 50)

    # Merge the features
    merged_df, tsfresh_feature_cols, handcrafted_feature_cols = merge_feature_sets(TSFRESH_CSV, HANDCRAFTED_CSV)

    # Count features accurately
    tsfresh_count, handcrafted_count, metadata_count, tsfresh_cols, handcrafted_cols = count_features_accurately(
        merged_df, tsfresh_feature_cols, handcrafted_feature_cols
    )

    # Save merged file
    merged_df.to_csv(MERGED_CSV, index=False)
    print(f"💾 Saved merged features: {MERGED_CSV}")

    # Print final statistics
    print(f"\n📊 FINAL MERGED DATASET:")
    print(f"   • Subjects: {len(merged_df)}")
    print(f"   • Total columns: {merged_df.shape[1]}")
    print(f"   • Controls: {len(merged_df[merged_df['group'] == 'control'])}")
    print(f"   • Patients: {len(merged_df[merged_df['group'] == 'patient'])}")

    print(f"   • TSFresh features: {tsfresh_count}")
    print(f"   • Handcrafted features: {handcrafted_count}")
    print(f"   • Metadata columns: {metadata_count}")

    # Show some example features
    print(f"\n🔍 Example TSFresh features: {tsfresh_cols[:5]}...")
    print(f"🔍 Example Handcrafted features: {handcrafted_cols[:5]}...")


if __name__ == "__main__":
    main()