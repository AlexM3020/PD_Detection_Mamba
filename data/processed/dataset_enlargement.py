import os
import pandas as pd
import numpy as np

# -----------------------
# CONFIG
# -----------------------

GAIT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\oversampling"
OUTPUT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_500"
CHUNK_SIZE = 500  # 2k timesteps per chunk

# -----------------------
# CREATE OUTPUT FOLDER
# -----------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -----------------------
# PROCESSING FUNCTION
# -----------------------

def split_gait_file(input_file_path, output_folder):
    """
    Split a gait data file into chunks of CHUNK_SIZE timesteps
    """
    try:
        # Read the gait data file (assuming tab/space separated values)
        df = pd.read_csv(input_file_path, delim_whitespace=True, header=None, comment='#')

        # Get total number of timesteps
        total_timesteps = len(df)

        if total_timesteps < CHUNK_SIZE:
            print(
                f"WARNING: File {os.path.basename(input_file_path)} has only {total_timesteps} timesteps (< {CHUNK_SIZE}). Skipping.")
            return 0

        # Calculate number of chunks
        num_chunks = total_timesteps // CHUNK_SIZE
        chunks_created = 0

        # Create chunks
        for i in range(num_chunks):
            start_idx = i * CHUNK_SIZE
            end_idx = start_idx + CHUNK_SIZE

            # Extract chunk
            chunk_df = df.iloc[start_idx:end_idx]

            # Create output filename
            base_name = os.path.splitext(os.path.basename(input_file_path))[0]
            output_filename = f"{base_name}_chunk{i + 1:02d}.txt"
            output_path = os.path.join(output_folder, output_filename)

            # Save chunk (preserving original format)
            chunk_df.to_csv(output_path, sep='\t', header=False, index=False)
            chunks_created += 1

        return chunks_created

    except Exception as e:
        print(f"ERROR processing {os.path.basename(input_file_path)}: {str(e)}")
        return 0


# -----------------------
# MAIN PROCESSING
# -----------------------

# Get all text files
txt_files = [f for f in os.listdir(GAIT_FOLDER) if f.endswith(".txt")]
print(f"Found {len(txt_files)} text files in the dataset")

total_original_files = 0
total_chunks_created = 0

# Process each file
for file in txt_files:
    input_path = os.path.join(GAIT_FOLDER, file)

    if os.path.getsize(input_path) > 0:  # Skip empty files
        chunks_created = split_gait_file(input_path, OUTPUT_FOLDER)

        if chunks_created > 0:
            total_original_files += 1
            total_chunks_created += chunks_created
            print(f"Processed {file}: created {chunks_created} chunks")
        else:
            print(f"Skipped {file}: insufficient data or error")

# -----------------------
# SUMMARY
# -----------------------

print("\n" + "=" * 50)
print("SPLITTING SUMMARY")
print("=" * 50)
print(f"Original files processed: {total_original_files}")
print(f"Total chunks created: {total_chunks_created}")
print(f"Chunk size: {CHUNK_SIZE} timesteps")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Enlargement factor: {total_chunks_created / total_original_files:.2f}x")

# Create a summary CSV file
summary_data = []
for file in os.listdir(OUTPUT_FOLDER):
    if file.endswith(".txt"):
        file_path = os.path.join(OUTPUT_FOLDER, file)
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            timesteps = len(df)
            channels = df.shape[1] if len(df.shape) > 1 else 1

            # Extract original subject ID from filename
            original_file = "_".join(file.split("_chunk")[0]) + ".txt"

            summary_data.append({
                'filename': file,
                'original_file': original_file,
                'timesteps': timesteps,
                'channels': channels,
                'subject_type': 'PD' if 'Pt' in file else 'CO'  # PD patients vs Controls
            })
        except:
            continue

if summary_data:
    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(OUTPUT_FOLDER, "dataset_split_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to: {summary_csv_path}")

print("\nDataset enlargement complete!")