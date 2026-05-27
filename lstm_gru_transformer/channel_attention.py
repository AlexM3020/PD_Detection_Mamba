import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------
# CONFIG
# -----------------------
RAW_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_oversampled"
FEATURE_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\merged_features.csv"

# No feature vector concatenation - using only time series data
SELECTED_FEATURES = []  # Empty list - no static features

# All available channels
ALL_CHANNELS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
CHANNELS = ALL_CHANNELS  # Using all channels initially

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-3
SEQ_LEN = 2000
HIDDEN_DIM = 64
NUM_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 42


# -----------------------
# DATASET (Simplified - no feature vector)
# -----------------------
class GaitDataset(Dataset):
    def __init__(self, file_paths, seq_len=2000, channels=None):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.channels = channels if channels is not None else ALL_CHANNELS

    def __len__(self):
        return len(self.file_paths)

    def pad_or_truncate(self, arr):
        T, C = arr.shape
        if T > self.seq_len:
            return arr[:self.seq_len, :]
        elif T < self.seq_len:
            pad = np.zeros((self.seq_len - T, C))
            return np.vstack([arr, pad])
        return arr

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        fname = os.path.basename(file_path)
        label = 0 if "Co" in fname else 1  # Co=control, Pt=parkinson

        # Load and preprocess data
        data = np.loadtxt(file_path)
        data = data[:, [i - 1 for i in self.channels]]  # Select specified channels

        # Per-sample normalization (no data leakage)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # Pad/truncate to fixed length
        data = self.pad_or_truncate(data)

        # Return only time series data and label (no feature vector)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# -----------------------
# TEMPORALLY-AWARE CHANNEL ATTENTION MODEL
# -----------------------
class TemporalChannelAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, seq_len=2000):
        super().__init__()

        # ----- TEMPORAL FEATURE EXTRACTION -----
        # Extract temporal patterns using 1D convolutions
        # This allows the attention to "see" temporal dynamics, not just averages
        self.temporal_encoder = nn.Sequential(
            # First conv layer: capture local patterns (stride 2 for downsampling)
            nn.Conv1d(input_dim, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second conv layer: capture mid-range patterns
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Third conv layer: capture broader patterns
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Calculate the size after convolutions
        # After 3 stride-2 convolutions: seq_len / 8
        conv_output_len = seq_len // 8
        self.temporal_feature_dim = 128 * conv_output_len

        # ----- CHANNEL ATTENTION (now temporally informed) -----
        # This attention mechanism sees temporal features, not just averages
        self.channel_attention = nn.Sequential(
            nn.Linear(self.temporal_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output one weight per channel
            nn.Sigmoid()  # Weights between 0-1
        )

        # Alternative: Multi-head attention over channels (more sophisticated)
        # Uncomment if you want to try this instead
        """
        self.channel_multihead_attn = nn.MultiheadAttention(
            embed_dim=self.temporal_feature_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )
        """

        # ----- LSTM for sequence processing -----
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        lstm_out_dim = hidden_dim * 2  # Bidirectional

        # ----- CLASSIFIER -----
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # For visualization and analysis
        self.last_channel_weights = None

    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        batch_size = x.shape[0]

        # ----- STEP 1: Extract temporal features per channel -----
        # Transpose for Conv1d: (batch, channels, seq_len)
        x_t = x.transpose(1, 2)

        # Extract temporal patterns
        temporal_features = self.temporal_encoder(x_t)  # (batch, 128, seq_len/8)

        # ----- STEP 2: Compute channel attention based on temporal patterns -----
        # Flatten temporal features for each channel to get a rich representation
        temporal_features_flat = temporal_features.reshape(batch_size, -1)  # (batch, 128 * seq_len/8)

        # Compute attention weights per channel based on their temporal patterns
        channel_weights = self.channel_attention(temporal_features_flat)  # (batch, channels)

        # Store for visualization
        self.last_channel_weights = channel_weights.detach()

        # ----- STEP 3: Apply attention weights to original signal -----
        # Reshape for broadcasting: (batch, 1, channels)
        channel_weights = channel_weights.unsqueeze(1)

        # Apply attention: amplify important channels, suppress less important ones
        x_weighted = x * channel_weights

        # ----- STEP 4: Process with LSTM -----
        lstm_out, _ = self.lstm(x_weighted)
        lstm_features = lstm_out[:, -1, :]  # Last timestep features

        # ----- STEP 5: Final classification -----
        output = self.fc(lstm_features)

        return torch.sigmoid(output).squeeze(), channel_weights.squeeze(1)


# -----------------------
# ALTERNATIVE: FULLY ATTENTION-BASED MODEL
# -----------------------
class FullTemporalChannelAttention(nn.Module):
    """
    More sophisticated model using multi-head attention for both temporal and channel dimensions
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3, seq_len=2000):
        super().__init__()

        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(input_dim, dropout, max_len=seq_len)

        # Temporal self-attention (which time points matter?)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Channel attention (which sensors matter?) - now using temporal context
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=seq_len,  # Treat time dimension as features for channel attention
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(seq_len)

        # LSTM for final sequence processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        lstm_out_dim = hidden_dim * 2

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.temp_weights = None
        self.chan_weights = None

    def forward(self, x):
        # x: (batch, seq_len, channels)

        # Add positional encoding
        x = self.pos_encoder(x)

        # ----- Temporal attention (which time steps are important?) -----
        x_temp, temp_weights = self.temporal_attention(x, x, x)
        x_temp = self.norm1(x + x_temp)  # Residual connection
        self.temp_weights = temp_weights

        # ----- Channel attention (which sensors are important?) -----
        # Transpose to treat time as features: (batch, channels, seq_len)
        x_chan = x_temp.transpose(1, 2)
        x_chan, chan_weights = self.channel_attention(x_chan, x_chan, x_chan)
        x_chan = self.norm2(x_temp.transpose(1, 2) + x_chan)  # Residual
        self.chan_weights = chan_weights

        # Transpose back: (batch, seq_len, channels)
        x_out = x_chan.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(x_out)
        lstm_features = lstm_out[:, -1, :]

        output = self.fc(lstm_features)

        return torch.sigmoid(output).squeeze(), chan_weights.mean(dim=1)  # Average channel weights across heads


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -----------------------
# CHANNEL ANALYSIS FUNCTIONS (Updated for temporal attention)
# -----------------------
def visualize_channel_attention(model, dataloader, channels, save_path="."):
    """
    Collect and visualize attention weights for each channel
    Now includes temporal pattern visualization
    """
    model.eval()
    all_weights = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out, weights = model(X)
            all_weights.append(weights.cpu().numpy())

    # Combine all batch weights
    if all_weights:
        all_weights = np.vstack(all_weights)
        avg_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)
    else:
        avg_weights = np.zeros(len(channels))
        std_weights = np.zeros(len(channels))

    # Create enhanced visualizations
    plt.figure(figsize=(18, 12))

    # 1. Bar chart with error bars (showing variability across samples)
    plt.subplot(2, 3, 1)
    bars = plt.bar(range(len(channels)), avg_weights, yerr=std_weights,
                   capsize=3, alpha=0.8, ecolor='black')
    plt.xticks(range(len(channels)), channels, rotation=45)
    plt.xlabel("Channel Number")
    plt.ylabel("Attention Weight")
    plt.title("Channel Importance (with variability across samples)")
    plt.grid(True, alpha=0.3, axis='y')

    # 2. Sorted channel importance
    plt.subplot(2, 3, 2)
    sorted_indices = np.argsort(avg_weights)[::-1]
    sorted_channels = [channels[i] for i in sorted_indices]
    sorted_weights = avg_weights[sorted_indices]
    sorted_stds = std_weights[sorted_indices]

    bars = plt.bar(range(len(channels)), sorted_weights, yerr=sorted_stds,
                   capsize=3, alpha=0.8)
    plt.xticks(range(len(channels)), sorted_channels, rotation=45)
    plt.xlabel("Channel (Sorted by Importance)")
    plt.ylabel("Attention Weight")
    plt.title("Channels Sorted by Importance (with variability)")

    # Color code: top 5 in green, bottom 5 in red
    for i in range(len(bars)):
        if i < 5:
            bars[i].set_color('darkgreen')
        elif i >= len(bars) - 5:
            bars[i].set_color('darkred')
        else:
            bars[i].set_color('steelblue')

    # 3. Cumulative importance
    plt.subplot(2, 3, 3)
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    plt.plot(range(1, len(cumulative) + 1), cumulative, 'o-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
    plt.xlabel("Number of Channels")
    plt.ylabel("Cumulative Importance")
    plt.title("How Many Channels Needed?")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Find how many channels give 80% and 90% of importance
    n_80 = np.argmax(cumulative >= 0.8) + 1
    n_90 = np.argmax(cumulative >= 0.9) + 1
    plt.axvline(x=n_80, color='g', linestyle='--', alpha=0.5,
                label=f'Top {n_80} = 80%')
    plt.axvline(x=n_90, color='purple', linestyle='--', alpha=0.5,
                label=f'Top {n_90} = 90%')

    # 4. Distribution of channel importance
    plt.subplot(2, 3, 4)
    plt.hist(avg_weights, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=np.mean(avg_weights), color='r', linestyle='--',
                label=f'Mean: {np.mean(avg_weights):.3f}')
    plt.axvline(x=np.median(avg_weights), color='g', linestyle='--',
                label=f'Median: {np.median(avg_weights):.3f}')
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.title("Distribution of Channel Importance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Box plot of channel weights (shows distribution)
    plt.subplot(2, 3, 5)
    box_data = [all_weights[:, i] for i in range(len(channels))]
    plt.boxplot(box_data, labels=channels, vert=True, patch_artist=True)
    plt.xticks(rotation=45)
    plt.xlabel("Channel")
    plt.ylabel("Attention Weight Distribution")
    plt.title("Channel Importance Distribution Across Samples")
    plt.grid(True, alpha=0.3, axis='y')

    # 6. Temporal pattern visualization (if model has temporal attention)
    plt.subplot(2, 3, 6)
    if hasattr(model, 'temp_weights') and model.temp_weights is not None:
        # Visualize temporal attention pattern
        temp_weights = model.temp_weights.mean(dim=1).cpu().numpy()  # Average across heads
        plt.imshow(temp_weights[:10, :100], aspect='auto',
                   cmap='viridis')  # Show first 10 samples, first 100 timepoints
        plt.colorbar(label='Attention Weight')
        plt.xlabel("Time Step")
        plt.ylabel("Sample")
        plt.title("Temporal Attention Pattern (first 10 samples)")
    else:
        # Alternative: Show channel weights heatmap across samples
        plt.imshow(all_weights[:50, :].T, aspect='auto', cmap='viridis')  # First 50 samples
        plt.colorbar(label='Attention Weight')
        plt.xlabel("Sample")
        plt.ylabel("Channel")
        plt.title("Channel Attention Across Samples")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "temporal_channel_attention_analysis.png"), dpi=300)
    plt.close()

    return avg_weights


def analyze_channel_performance(channels, channel_weights, model=None, val_loader=None):
    """
    Enhanced analysis with temporal pattern information
    """
    print("\n" + "=" * 70)
    print("TEMPORAL CHANNEL ATTENTION ANALYSIS")
    print("=" * 70)

    # Sort channels by importance
    sorted_indices = np.argsort(channel_weights)[::-1]
    sorted_channels = [channels[i] for i in sorted_indices]
    sorted_weights = channel_weights[sorted_indices]

    # Print ranking
    print(f"\n{'Rank':<6} {'Channel':<10} {'Weight':<12} {'Importance':<15} {'Temporal Pattern'}")
    print("-" * 70)

    for rank, (ch, w) in enumerate(zip(sorted_channels, sorted_weights), 1):
        if rank <= 3:
            importance = "CRITICAL"
            pattern = "🎯 Rich temporal dynamics"
        elif rank <= 8:
            importance = "IMPORTANT"
            pattern = "📊 Moderate temporal info"
        elif rank <= 12:
            importance = "MODERATE"
            pattern = "📈 Some temporal patterns"
        else:
            importance = "LOW"
            pattern = "📉 Mostly baseline"

        print(f"{rank:<6} {ch:<10} {w:<12.4f} {importance:<15} {pattern}")

    # Analyze different channel subsets
    print(f"\n\nCHANNEL SUBSET ANALYSIS (Temporally-Aware):")
    print("-" * 70)

    # Calculate cumulative importance
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    n_80 = np.argmax(cumulative >= 0.8) + 1
    n_90 = np.argmax(cumulative >= 0.9) + 1

    subsets = {
        "Top 5 channels": sorted_channels[:5],
        f"Top {n_80} channels (80% power)": sorted_channels[:n_80],
        f"Top {n_90} channels (90% power)": sorted_channels[:n_90],
        "Top 10 channels": sorted_channels[:10],
        "Above average (> mean)": [ch for ch, w in zip(channels, channel_weights)
                                   if w > np.mean(channel_weights)],
        "Top 50%": sorted_channels[:len(channels) // 2],
    }

    for name, subset in subsets.items():
        reduction = 100 * (1 - len(subset) / len(channels))
        subset_weights = [channel_weights[channels.index(ch)] for ch in subset]
        avg_weight = np.mean(subset_weights)
        total_importance = np.sum(subset_weights)

        print(f"\n{name}:")
        print(f"  • Channels: {subset}")
        print(f"  • Count: {len(subset)}/{len(channels)} ({reduction:.1f}% reduction)")
        print(f"  • Avg weight: {avg_weight:.4f}")
        print(f"  • Total importance: {total_importance:.4f} ({total_importance / np.sum(channel_weights) * 100:.1f}%)")

    return sorted_channels, sorted_weights


def save_channel_analysis_report(channels, weights, val_accuracy, model=None, save_path="."):
    """
    Save detailed temporally-aware channel analysis to file
    """
    report_path = os.path.join(save_path, "temporal_channel_analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL CHANNEL ATTENTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"MODEL: TemporalChannelAttentionLSTM\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Total Channels: {len(channels)}\n")
        f.write(f"Mean Attention Weight: {np.mean(weights):.4f}\n")
        f.write(f"Std Attention Weight: {np.std(weights):.4f}\n")
        f.write(f"Max Weight: {np.max(weights):.4f}\n")
        f.write(f"Min Weight: {np.min(weights):.4f}\n")
        f.write(f"Range: {np.max(weights) - np.min(weights):.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("CHANNEL IMPORTANCE RANKING (Temporally-Aware)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Channel':<10} {'Weight':<12} {'% of Max':<12} {'Cumulative %':<12}\n")
        f.write("-" * 70 + "\n")

        sorted_indices = np.argsort(weights)[::-1]
        max_weight = np.max(weights)
        cumulative_sum = 0
        total_sum = np.sum(weights)

        for rank, idx in enumerate(sorted_indices, 1):
            channel = channels[idx]
            weight = weights[idx]
            percent_of_max = 100 * weight / max_weight
            cumulative_sum += weight
            cumulative_percent = 100 * cumulative_sum / total_sum

            f.write(f"{rank:<6} {channel:<10} {weight:<12.4f} {percent_of_max:<12.1f}% {cumulative_percent:<12.1f}%\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS (Based on Temporal Pattern Analysis)\n")
        f.write("=" * 80 + "\n")

        # Calculate cumulative importance
        sorted_weights = weights[sorted_indices]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)

        n_80 = np.argmax(cumulative >= 0.8) + 1
        n_90 = np.argmax(cumulative >= 0.9) + 1
        n_95 = np.argmax(cumulative >= 0.95) + 1

        f.write(f"\n📊 OPTIMAL CHANNEL SUBSETS:\n")
        f.write(f"   • 80% of discriminative power: Top {n_80} channels\n")
        f.write(f"   • 90% of discriminative power: Top {n_90} channels\n")
        f.write(f"   • 95% of discriminative power: Top {n_95} channels\n\n")

        f.write(f"\n🎯 TOP 5 MOST IMPORTANT CHANNELS (Critical for temporal patterns):\n")
        for i in range(min(5, len(channels))):
            idx = sorted_indices[i]
            f.write(f"   {i + 1}. Channel {channels[idx]} (weight: {weights[idx]:.4f})\n")

        f.write(f"\n⚠️ BOTTOM 5 LEAST IMPORTANT CHANNELS (Candidates for removal):\n")
        for i in range(1, min(6, len(channels) + 1)):
            idx = sorted_indices[-i]
            f.write(f"   {i}. Channel {channels[idx]} (weight: {weights[idx]:.4f})\n")

        f.write(f"\n\n💡 PRACTICAL DEPLOYMENT RECOMMENDATIONS:\n")
        f.write(f"   • Maximum accuracy setup: Use all {len(channels)} channels\n")
        f.write(f"   • Balanced setup: Use top {n_90} channels (90% power, {len(channels) - n_90} fewer sensors)\n")
        f.write(f"   • Minimal setup: Use top {n_80} channels (80% power, {len(channels) - n_80} fewer sensors)\n")
        f.write(f"   • Ultra-minimal setup: Use top 5 channels\n")

    print(f"\n✅ Detailed temporal channel analysis saved to: {report_path}")


# -----------------------
# TRAIN / EVAL FUNCTIONS (Unchanged)
# -----------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out, attention_weights = model(X)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += ((out > 0.5).float() == y).float().mean().item()

    return total_loss / len(loader), total_acc / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    preds, targets = [], []
    all_attention_weights = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out, attention_weights = model(X)

            loss = criterion(out, y)
            total_loss += loss.item()
            total_acc += ((out > 0.5).float() == y).float().mean().item()

            preds.extend((out > 0.5).float().cpu().numpy())
            targets.extend(y.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())

    return total_loss / len(loader), total_acc / len(loader), np.array(preds), np.array(targets)


# -----------------------
# MAIN FUNCTION (Updated to use temporal model)
# -----------------------
def main():
    print("=" * 70)
    print("TEMPORAL CHANNEL ATTENTION ANALYSIS FOR GAIT ANALYSIS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Using channels: {CHANNELS}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Attention type: Temporally-aware (using CNN features)")

    # Create output directory
    output_dir = "temporal_channel_attention_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_PATH}")

    # Find all .txt files
    def find_all_txt_files(root_dir):
        txt_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    full_path = os.path.join(root, file)
                    txt_files.append(full_path)
        return txt_files

    all_files = find_all_txt_files(RAW_PATH)
    print(f"Found {len(all_files)} .txt files total")

    # Get subject ID from filename
    def get_subject_id(file_path):
        return os.path.basename(file_path)[:6]  # GaCo01 / GaPt02

    # Get subjects
    subjects = np.array([get_subject_id(f) for f in all_files])

    # Unique subjects
    unique_subjects = np.unique(subjects)

    # Extract class label from subject ID
    subject_labels = np.array([1 if s[2:4] == "Pt" else 0 for s in unique_subjects])

    # Split by subjects
    train_subj, val_subj = train_test_split(
        unique_subjects,
        test_size=0.20,
        random_state=42,
        stratify=subject_labels
    )

    # Assign files to splits
    train_files = [f for f in all_files if get_subject_id(f) in train_subj]
    val_files = [f for f in all_files if get_subject_id(f) in val_subj]

    print(f"\nTraining set: {len(train_files)} files from {len(train_subj)} subjects")
    print(f"Validation set: {len(val_files)} files from {len(val_subj)} subjects")

    # Create datasets
    train_ds = GaitDataset(train_files, SEQ_LEN, CHANNELS)
    val_ds = GaitDataset(val_files, SEQ_LEN, CHANNELS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Initialize TEMPORALLY-AWARE model
    input_dim = len(CHANNELS)
    model = TemporalChannelAttentionLSTM(
        input_dim,
        HIDDEN_DIM,
        NUM_LAYERS,
        seq_len=SEQ_LEN
    ).to(DEVICE)

    # If you want to try the more sophisticated full attention model instead:
    # model = FullTemporalChannelAttention(input_dim, HIDDEN_DIM, NUM_LAYERS, seq_len=SEQ_LEN).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING TEMPORAL CHANNEL ATTENTION LSTM")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience, patience_counter = 15, 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, targets = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1:03d}: Train Loss={tr_loss:.4f}, Acc={tr_acc:.3f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"💾 Saved best model (Loss: {val_loss:.4f}, Acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model for analysis
    print("\n" + "=" * 60)
    print("ANALYZING TEMPORAL CHANNEL ATTENTION")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    model.eval()

    # Final evaluation
    final_val_loss, final_val_acc, final_preds, final_targets = eval_epoch(model, val_loader, criterion)
    print(f"\nFinal Validation - Loss: {final_val_loss:.4f}, Acc: {final_val_acc:.3f}")

    # Generate and save classification report
    report = classification_report(final_targets, final_preds,
                                   target_names=["Control", "Parkinson"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    # Generate confusion matrix
    cm = confusion_matrix(final_targets, final_preds)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Control", "Parkinson"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (Acc: {final_val_acc:.3f})")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # -----------------------
    # TEMPORAL CHANNEL ATTENTION ANALYSIS
    # -----------------------
    print("\n" + "=" * 60)
    print("COMPUTING TEMPORALLY-AWARE CHANNEL IMPORTANCE")
    print("=" * 60)

    # Get channel attention weights (now based on temporal patterns!)
    channel_weights = visualize_channel_attention(model, val_loader, CHANNELS, output_dir)

    # Print channel importance ranking
    print("\nCHANNEL IMPORTANCE RANKING (Based on Temporal Patterns):")
    print("-" * 70)

    sorted_indices = np.argsort(channel_weights)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        channel = CHANNELS[idx]
        weight = channel_weights[idx]
        stars = "★" * min(5, int(weight * 10))
        print(f"Rank {rank:2d}: Channel {channel:2d} → Weight: {weight:.4f} {stars}")

    # Analyze channel performance with temporal context
    sorted_channels, sorted_weights = analyze_channel_performance(
        CHANNELS, channel_weights, model, val_loader
    )

    # Save detailed analysis report
    save_channel_analysis_report(CHANNELS, channel_weights, final_val_acc, model, output_dir)

    # -----------------------
    # VISUALIZE TRAINING HISTORY
    # -----------------------
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    plt.plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History - Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    plt.close()

    # -----------------------
    # CREATE SUMMARY
    # -----------------------
    print("\n" + "=" * 70)
    print("SUMMARY OF TEMPORAL CHANNEL ATTENTION FINDINGS")
    print("=" * 70)

    # Find most and least important channels
    most_important_idx = np.argmax(channel_weights)
    least_important_idx = np.argmin(channel_weights)

    print(f"\n🎯 MOST IMPORTANT CHANNEL (Temporally):")
    print(f"   Channel {CHANNELS[most_important_idx]} with weight {channel_weights[most_important_idx]:.4f}")
    print(f"   This channel's temporal patterns are highly discriminative")

    print(f"\n⚠️ LEAST IMPORTANT CHANNEL (Temporally):")
    print(f"   Channel {CHANNELS[least_important_idx]} with weight {channel_weights[least_important_idx]:.4f}")
    print(f"   This channel's temporal patterns add little discriminative value")

    print(f"\n📊 CHANNEL IMPORTANCE STATISTICS:")
    print(f"   Highest weight: {np.max(channel_weights):.4f}")
    print(f"   Lowest weight:  {np.min(channel_weights):.4f}")
    print(f"   Mean weight:    {np.mean(channel_weights):.4f}")
    print(f"   Std deviation:  {np.std(channel_weights):.4f}")
    print(f"   Ratio (high/low): {np.max(channel_weights) / np.min(channel_weights):.2f}x")

    # Calculate cumulative importance
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    n_80 = np.argmax(cumulative >= 0.8) + 1
    n_90 = np.argmax(cumulative >= 0.9) + 1
    top_channels_80 = sorted_channels[:n_80]
    top_channels_90 = sorted_channels[:n_90]

    print(f"\n💡 PRACTICAL RECOMMENDATIONS (Based on Temporal Patterns):")
    print(f"   1. For optimal performance: Use ALL {len(CHANNELS)} channels")
    print(f"   2. For efficient setup: Use top {n_80} channels (80% of temporal discriminative power)")
    print(f"      Top channels: {top_channels_80}")
    print(f"   3. For high efficiency: Use top {n_90} channels (90% of power)")
    print(f"      Top channels: {top_channels_90}")
    print(f"   4. For minimal setup: Use top 5 channels: {sorted_channels[:5]}")

    print(f"\n✅ Analysis complete! All results saved to: {output_dir}/")
    print(f"   - temporal_channel_attention_analysis.png (enhanced visualizations)")
    print(f"   - temporal_channel_analysis_report.txt (detailed analysis)")
    print(f"   - classification_report.csv (performance metrics)")
    print(f"   - confusion_matrix.png (confusion matrix)")
    print(f"   - training_history.png (training curves)")

    # Save final channel weights for future use
    weights_df = pd.DataFrame({
        'channel': CHANNELS,
        'attention_weight': channel_weights,
        'rank': [np.where(sorted_indices == i)[0][0] + 1 for i in range(len(CHANNELS))]
    })
    weights_df = weights_df.sort_values('rank')
    weights_df.to_csv(os.path.join(output_dir, "channel_weights_temporal.csv"), index=False)

    print(f"\n📊 Channel weights saved to: channel_weights_temporal.csv")
    print("\n" + "=" * 70)

    return model, channel_weights


if __name__ == "__main__":
    model, channel_weights = main()