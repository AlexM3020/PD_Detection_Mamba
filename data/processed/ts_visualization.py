import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, signal
from math import inf

# -----------------------
# CONFIG
# -----------------------

GAIT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_oversampled "
OUTPUT_FOLDER = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\average_analysis_with_cycles"
SEQUENCE_LENGTH = 2000  # use only the first 2000 timesteps (approx 20s per your note)
CYCLE_POINTS = 100      # normalize each gait cycle to 100 points (0-100% gait cycle)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------
# HELPERS: DTW and NRMSE
# -----------------------

def dtw_distance(x, y, window=None):
    """
    Simple DTW (no external dependency).
    x, y: 1D numpy arrays
    window: maximum warping window (int). If None uses full matrix.
    Returns the DTW distance (sum of squared distances) and the normalized distance (sqrt / length)
    """
    n, m = len(x), len(y)
    if window is None:
        window = max(n, m)
    w = max(window, abs(n - m))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        start_j = max(1, i - w)
        end_j = min(m, i + w)
        for j in range(start_j, end_j + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j],    # insertion
                                   dtw[i, j - 1],    # deletion
                                   dtw[i - 1, j - 1])# match

    dist = dtw[n, m]
    normalized = np.sqrt(dist) / (n + m)  # normalized by sequence size (simple)
    return dist, normalized

def nrmse(x, y):
    """
    Normalized RMSE between two waveforms x and y.
    Normalization is done by dividing RMSE by (max(x,y) - min(x,y)) across both signals.
    If range is zero, divide by mean absolute value (fallback).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mse = np.mean((x - y) ** 2)
    rmse = np.sqrt(mse)
    combined = np.concatenate([x, y])
    rng = combined.max() - combined.min()
    if rng == 0:
        denom = np.mean(np.abs(combined)) if np.mean(np.abs(combined)) != 0 else 1.0
    else:
        denom = rng
    return rmse / denom

# -----------------------
# FILE CLASSIFICATION AND DATA EXTRACTION
# -----------------------

def classify_subject_and_extract_data(filename):
    filename_lower = filename.lower()
    if 'pt' in filename_lower:
        subject_type = 'PD'
    elif 'co' in filename_lower:
        subject_type = 'Control'
    else:
        subject_type = 'Unknown'
    return subject_type

def extract_foot_data(filepath):
    """
    Returns left_data, right_data as arrays shaped (timesteps, 8)
    Uses only first SEQUENCE_LENGTH timesteps. Interpolates/truncates to SEQUENCE_LENGTH.
    """
    try:
        df = pd.read_csv(filepath, delim_whitespace=True, header=None, comment='#')
        df = df.dropna(axis=1, how='all')

        if df.shape[1] < 19:
            print(f"Warning: File {os.path.basename(filepath)} has only {df.shape[1]} columns, expected 19")
            return None, None

        left_data = df.iloc[:, 1:9].values
        right_data = df.iloc[:, 9:17].values

        # Use only first SEQUENCE_LENGTH timesteps
        if len(left_data) >= SEQUENCE_LENGTH:
            left_data = left_data[:SEQUENCE_LENGTH, :]
            right_data = right_data[:SEQUENCE_LENGTH, :]
        else:
            # if shorter, upsample to SEQUENCE_LENGTH by linear interpolation
            orig = np.arange(len(left_data))
            new = np.linspace(0, len(left_data) - 1, SEQUENCE_LENGTH)
            left_cols = []
            right_cols = []
            for c in range(8):
                f_left = interpolate.interp1d(orig, left_data[:, c], kind='linear', fill_value='extrapolate')
                left_cols.append(f_left(new))
                f_right = interpolate.interp1d(orig, right_data[:, c], kind='linear', fill_value='extrapolate')
                right_cols.append(f_right(new))
            left_data = np.column_stack(left_cols)
            right_data = np.column_stack(right_cols)

        return left_data, right_data
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return None, None

# -----------------------
# CYCLE DETECTION & NORMALIZATION
# -----------------------

def detect_cycles(force_signal, sampling_rate_est=100.0):
    """
    Detect gait cycles (heel strikes) from a single foot total VGRF signal.
    force_signal: 1D array of summed sensors for a foot.
    sampling_rate_est is only used for default min distance heuristic.
    Returns list of peak indices (heel strikes).
    """
    # A robust approach: find peaks above a dynamic threshold; min distance between peaks ~ 0.4s (cadence ~1.5-2.5Hz)
    # We assume ~50-100 Hz sampling; since we fixed 2000 timesteps ~= 20s -> 100 Hz.
    # So min distance ~ int(0.4 * sampling_rate_est) ~ 40
    min_distance = int(0.4 * sampling_rate_est)
    # threshold: fraction of the signal max
    thr = max(0.2 * np.max(force_signal), np.mean(force_signal) + 0.1 * np.std(force_signal))
    peaks, _ = signal.find_peaks(force_signal, height=thr, distance=min_distance)
    # Ensure peaks are within range and not too few — fallback to peaks on smoothed derivative if none
    if len(peaks) < 2:
        # smoothing then peaks with lower threshold
        smooth = signal.savgol_filter(force_signal, 31 if len(force_signal)>31 else max(5,len(force_signal)//3*2+1), 3)
        peaks, _ = signal.find_peaks(smooth, distance=min_distance)
    return peaks

def extract_and_normalize_cycles(foot_data, cycle_points=CYCLE_POINTS):
    """
    foot_data: (timesteps, 8) array for one trial (one file, one foot)
    Returns list_of_cycles_per_sensor: list (len=8) of arrays shape (n_cycles, cycle_points)
    """
    total_force = foot_data.sum(axis=1)
    # estimate sampling rate from your note: 2000 timesteps ~ 20s -> fs ~ 100 Hz
    peaks = detect_cycles(total_force, sampling_rate_est=100)
    cycles_by_sensor = [ [] for _ in range(8) ]
    if len(peaks) < 2:
        # cannot segment cycles; treat entire trace as single 'cycle' (fallback)
        for s in range(8):
            col = foot_data[:, s]
            x_orig = np.arange(len(col))
            x_new = np.linspace(0, len(col)-1, cycle_points)
            f = interpolate.interp1d(x_orig, col, kind='linear', fill_value='extrapolate')
            cycles_by_sensor[s].append(f(x_new))
        return cycles_by_sensor

    # define cycles between consecutive peaks (peak_i -> peak_{i+1})
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        if end - start < 5:  # too short, skip
            continue
        idx_orig = np.arange(start, end+1)
        x_new = np.linspace(start, end, cycle_points)
        for s in range(8):
            col = foot_data[:, s]
            f = interpolate.interp1d(np.arange(len(col)), col, kind='linear', fill_value='extrapolate')
            cycle_resampled = f(x_new)
            cycles_by_sensor[s].append(cycle_resampled)
    return cycles_by_sensor

# -----------------------
# MAIN PROCESSING LOOP
# -----------------------

txt_files = [f for f in os.listdir(GAIT_FOLDER) if f.endswith(".txt")]
print(f"Found {len(txt_files)} text files")

# We'll store cycles per subject then average cycles per subject then across subjects
# Structure: cycles_data[subject_type]['Left' or 'Right'][subject_id] = list_of_cycles_per_sensor (list of length 8)
cycles_data = {'PD': {'Left': [], 'Right': []}, 'Control': {'Left': [], 'Right': []}}
file_counts = {'PD': 0, 'Control': 0}

for file in txt_files:
    filepath = os.path.join(GAIT_FOLDER, file)
    subject_type = classify_subject_and_extract_data(file)

    if subject_type in ['PD', 'Control']:
        left_data, right_data = extract_foot_data(filepath)

        if left_data is not None and right_data is not None:
            # For each foot, extract cycles_by_sensor (list of 8 lists of cycles)
            left_cycles = extract_and_normalize_cycles(left_data, cycle_points=CYCLE_POINTS)
            right_cycles = extract_and_normalize_cycles(right_data, cycle_points=CYCLE_POINTS)

            # Convert to numpy arrays per sensor: shape (n_cycles, cycle_points)
            # For cases with single fallback cycle, this will be (1, cycle_points)
            left_cycles_np = [np.vstack(left_cycles[s]) if len(left_cycles[s])>0 else np.empty((0, CYCLE_POINTS)) for s in range(8)]
            right_cycles_np = [np.vstack(right_cycles[s]) if len(right_cycles[s])>0 else np.empty((0, CYCLE_POINTS)) for s in range(8)]

            cycles_data[subject_type]['Left'].append(left_cycles_np)
            cycles_data[subject_type]['Right'].append(right_cycles_np)
            file_counts[subject_type] += 1

print("\nFile counts by category:")
print(f"PD Patients: {file_counts['PD']} files")
print(f"Controls: {file_counts['Control']} files")
print(f"Total processed files: {file_counts['PD'] + file_counts['Control']}")

# -----------------------
# BUILD PER-SUBJECT AVERAGE CYCLE (then group average)
# -----------------------

# For each subject (each entry in cycles_data[type][foot]) we have a list of 8 arrays (n_cycles, CYCLE_POINTS).
# First compute per-subject average cycle per sensor (if a subject has zero cycles for a sensor skip).
def compute_subject_avg_cycle(cycles_np_list):
    # cycles_np_list: list of 8 arrays (n_cycles, cycle_points)
    subj_avg = []
    for s in range(8):
        arr = cycles_np_list[s]
        if arr.size == 0:
            subj_avg.append(None)  # missing
        else:
            subj_avg.append(np.nanmean(arr, axis=0))  # average cycles for that subject, sensor
    return subj_avg  # list of length 8

# Collect subject averages per group and foot
group_avg_cycles = {'PD_Left': [None]*8, 'PD_Right': [None]*8, 'Control_Left': [None]*8, 'Control_Right': [None]*8}
group_subject_counts = {'PD_Left': 0, 'PD_Right': 0, 'Control_Left': 0, 'Control_Right': 0}

# For each group and foot produce a list of subject-average cycles per sensor, then compute group mean
for group in ['PD', 'Control']:
    for foot in ['Left', 'Right']:
        subj_list = cycles_data[group][foot]  # list of subjects
        per_sensor_subject_avgs = [[] for _ in range(8)]
        for subj_cycles in subj_list:
            subj_avg = compute_subject_avg_cycle(subj_cycles)
            for s in range(8):
                if subj_avg[s] is not None:
                    per_sensor_subject_avgs[s].append(subj_avg[s])
        # now compute group average per sensor (mean across subjects)
        for s in range(8):
            if len(per_sensor_subject_avgs[s]) > 0:
                arr = np.vstack(per_sensor_subject_avgs[s])  # (n_subjects, CYCLE_POINTS)
                group_mean = np.nanmean(arr, axis=0)
                group_avg_cycles[f'{group}_{foot}'][s] = group_mean
            else:
                group_avg_cycles[f'{group}_{foot}'][s] = np.full((CYCLE_POINTS,), np.nan)
        group_subject_counts[f'{group}_{foot}'] = len(subj_list)

# Save group average cycles to CSVs (one CSV per group+foot with columns Sensor_1...Sensor_8 and rows 0..CYCLE_POINTS-1)
for key, sensor_list in group_avg_cycles.items():
    df = pd.DataFrame({f'Sensor_{i+1}': sensor_list[i] for i in range(8)})
    df.to_csv(os.path.join(OUTPUT_FOLDER, f'group_average_cycle_{key.lower()}.csv'), index=False)
    print(f"Saved {key} average cycles to CSV with shape {df.shape}")

# -----------------------
# COMPUTE DISTANCES (NRMSE and DTW) per sensor and foot
# -----------------------

sensors = [f'Sensor_{i+1}' for i in range(8)]
results = []
for foot in ['Left', 'Right']:
    pd_key = f'PD_{foot}'
    co_key = f'Control_{foot}'
    for s_idx in range(8):
        pd_wave = group_avg_cycles[pd_key][s_idx]
        co_wave = group_avg_cycles[co_key][s_idx]
        # if both waves are not all-NaN, compute distances
        if np.all(np.isnan(pd_wave)) or np.all(np.isnan(co_wave)):
            results.append({'Foot': foot, 'Sensor': s_idx+1, 'NRMSE': np.nan, 'DTW_norm': np.nan, 'DTW_raw': np.nan})
            continue

        # For DTW window, use 20% of length as Sakoe-Chiba band
        window = int(0.2 * CYCLE_POINTS)
        dtw_raw, dtw_norm = dtw_distance(pd_wave, co_wave, window=window)
        nr = nrmse(pd_wave, co_wave)
        results.append({'Foot': foot, 'Sensor': s_idx+1, 'NRMSE': nr, 'DTW_norm': dtw_norm, 'DTW_raw': dtw_raw})

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_FOLDER, 'sensor_discriminative_metrics.csv'), index=False)
print(f"Saved sensor discriminative metrics to {os.path.join(OUTPUT_FOLDER, 'sensor_discriminative_metrics.csv')}")

# -----------------------
# RANKING SENSORS
# -----------------------

# We will rank sensors for each foot by both NRMSE and DTW_norm. To combine them into a single ranking
# compute rank (1 best discriminative -> highest distance -> so we rank descending), then average ranks.
ranking_rows = []
for foot in ['Left', 'Right']:
    dff = df_results[df_results['Foot'] == foot].copy()
    # higher distance => more discriminative -> we invert ranks so higher distance gets rank 1
    dff['rank_nrmse'] = dff['NRMSE'].rank(ascending=False, method='min')
    dff['rank_dtw'] = dff['DTW_norm'].rank(ascending=False, method='min')
    dff['combined_rank'] = (dff['rank_nrmse'] + dff['rank_dtw']) / 2.0
    dff_sorted = dff.sort_values('combined_rank')
    for _, row in dff_sorted.iterrows():
        ranking_rows.append({
            'Foot': foot,
            'Sensor': int(row['Sensor']),
            'NRMSE': row['NRMSE'],
            'DTW_norm': row['DTW_norm'],
            'rank_nrmse': int(row['rank_nrmse']) if not np.isnan(row['rank_nrmse']) else np.nan,
            'rank_dtw': int(row['rank_dtw']) if not np.isnan(row['rank_dtw']) else np.nan,
            'combined_rank': row['combined_rank']
        })

df_ranking = pd.DataFrame(ranking_rows)
df_ranking.to_csv(os.path.join(OUTPUT_FOLDER, 'sensor_ranking_by_discriminative_power.csv'), index=False)
print(f"Saved sensor ranking to {os.path.join(OUTPUT_FOLDER, 'sensor_ranking_by_discriminative_power.csv')}")

# -----------------------
# PLOTTING: Heatmaps and per-sensor plots (NRMSE / DTW)
# -----------------------

plt.style.use('seaborn-v0_8')

# Heatmap of NRMSE (sensors x feet)
heat_nrmse = df_results.pivot(index='Sensor', columns='Foot', values='NRMSE').sort_index()
heat_dtw = df_results.pivot(index='Sensor', columns='Foot', values='DTW_norm').sort_index()

# Plot NRMSE heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(heat_nrmse, annot=True, fmt='.3f', cmap='rocket', cbar_kws={'label': 'NRMSE'})
plt.title('Sensor NRMSE (PD vs Control) per Foot')
plt.ylabel('Sensor')
plt.xlabel('Foot')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'heatmap_nrmse.png'), dpi=300)
plt.close()

# Plot DTW heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(heat_dtw, annot=True, fmt='.4f', cmap='mako', cbar_kws={'label': 'DTW_norm'})
plt.title('Sensor DTW (PD vs Control) per Foot (normalized)')
plt.ylabel('Sensor')
plt.xlabel('Foot')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'heatmap_dtw.png'), dpi=300)
plt.close()

# Combined panel heatmap (NRMSE + DTW side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(heat_nrmse, annot=True, fmt='.3f', cmap='rocket', cbar_kws={'label': 'NRMSE'}, ax=axes[0])
axes[0].set_title('NRMSE')
axes[0].set_ylabel('Sensor')
axes[0].set_xlabel('Foot')

sns.heatmap(heat_dtw, annot=True, fmt='.4f', cmap='mako', cbar_kws={'label': 'DTW_norm'}, ax=axes[1])
axes[1].set_title('DTW_norm')
axes[1].set_ylabel('')
axes[1].set_xlabel('Foot')

plt.suptitle('Sensor Discriminative Metrics (PD vs Control)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_FOLDER, 'heatmap_combined_metrics.png'), dpi=300)
plt.close()

# Also create per-sensor plot comparing PD vs Control average cycle on Left and Right
for s in range(8):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for i, foot in enumerate(['Left', 'Right']):
        pd_wave = group_avg_cycles[f'PD_{foot}'][s]
        co_wave = group_avg_cycles[f'Control_{foot}'][s]
        axes[i].plot(np.arange(CYCLE_POINTS), pd_wave, label='PD', linewidth=2)
        axes[i].plot(np.arange(CYCLE_POINTS), co_wave, label='Control', linewidth=2, alpha=0.8)
        axes[i].set_title(f'Sensor {s+1} - {foot}')
        axes[i].set_xlabel('Gait cycle (%)')
        axes[i].set_ylabel('VGRF (resampled)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.suptitle(f'Sensor {s+1} Average Cycle (PD vs Control)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'sensor_{s+1}_pd_vs_control_cycle.png'), dpi=300)
    plt.close()

print("Generated heatmaps and per-sensor PD-vs-Control cycle plots.")

# -----------------------
# SAVE SUMMARY
# -----------------------

with open(os.path.join(OUTPUT_FOLDER, 'summary_readme.txt'), 'w') as f:
    f.write("Analysis summary\n")
    f.write("================\n")
    f.write(f"Files processed: PD={file_counts['PD']}, Control={file_counts['Control']}\n")
    f.write("Per-group average cycles saved for PD_Left, PD_Right, Control_Left, Control_Right\n")
    f.write("Sensor discriminative metrics saved: sensor_discriminative_metrics.csv\n")
    f.write("Sensor ranking saved: sensor_ranking_by_discriminative_power.csv\n")
    f.write("Heatmaps: heatmap_nrmse.png, heatmap_dtw.png, heatmap_combined_metrics.png\n")

print(f"\nAnalysis complete! Results saved to: {OUTPUT_FOLDER}")
