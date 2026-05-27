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

# ✅ No feature vector concatenation - using only time series data
SELECTED_FEATURES = []  # Empty list - no static features

# All available channels
ALL_CHANNELS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
CHANNELS = ALL_CHANNELS  # Using all channels initially

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LR = 2.5e-4
SEQ_LEN = 2000
HIDDEN_DIM = 96
NUM_LAYERS = 3
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
# MODEL with CHANNEL ATTENTION (Simplified - no feature vector)
# -----------------------
class ChannelAttentionGRU(nn.Module):  # Changed class name from ChannelAttentionLSTM to ChannelAttentionGRU
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        # Channel attention layer - learns importance of each sensor channel
        self.channel_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # Compress information
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),  # Reconstruct channel importance
            nn.Sigmoid()  # Output between 0-1 for each channel
        )

        # GRU for sequence processing (substituted LSTM with GRU)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)  # Changed LSTM to GRU

        gru_out_dim = hidden_dim * 2  # Bidirectional

        # Simplified classifier (no feature vector concatenation)
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, channels)

        # 1. Compute average signal for each channel across time
        channel_avg = x.mean(dim=1)  # (batch, channels)

        # 2. Get attention weights for each channel
        channel_weights = self.channel_attention(channel_avg)  # (batch, channels)

        # 3. Reshape for broadcasting
        channel_weights = channel_weights.unsqueeze(1)  # (batch, 1, channels)

        # 4. Apply attention: amplify important channels, suppress less important ones
        x_weighted = x * channel_weights

        # 5. Process with GRU (substituted LSTM with GRU)
        gru_out, _ = self.gru(x_weighted)  # Changed from LSTM to GRU
        gru_features = gru_out[:, -1, :]  # Last timestep features

        # 6. Final classification (no feature vector concatenation)
        output = self.fc(gru_features)

        return torch.sigmoid(output).squeeze(), channel_weights.squeeze(1)


# -----------------------
# CHANNEL ANALYSIS FUNCTIONS
# -----------------------
def visualize_channel_attention(model, dataloader, channels, save_path="."):
    """
    Collect and visualize attention weights for each channel
    Returns: Average attention weights for each channel
    """
    model.eval()
    all_weights = []

    with torch.no_grad():
        for X, y in dataloader:  # Simplified: only X, y (no feat_vec)
            X, y = X.to(DEVICE), y.to(DEVICE)
            out, weights = model(X)  # Only pass X

            # weights shape: (batch_size, num_channels)
            all_weights.append(weights.cpu().numpy())

    # Combine all batch weights
    if all_weights:
        all_weights = np.vstack(all_weights)  # (total_samples, num_channels)
        avg_weights = np.mean(all_weights, axis=0)  # (num_channels,)
    else:
        avg_weights = np.zeros(len(channels))

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # 1. Bar chart of channel importance
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(channels)), avg_weights)
    plt.xticks(range(len(channels)), channels, rotation=45)
    plt.xlabel("Channel Number")
    plt.ylabel("Attention Weight")
    plt.title("Channel Importance (Learned by Attention)")
    plt.grid(True, alpha=0.3, axis='y')

    # Color bars by importance
    for i, bar in enumerate(bars):
        if avg_weights[i] > np.mean(avg_weights):
            bar.set_color('green')
        elif avg_weights[i] < np.mean(avg_weights):
            bar.set_color('red')
        else:
            bar.set_color('blue')

    # 2. Sorted channel importance
    plt.subplot(2, 2, 2)
    sorted_indices = np.argsort(avg_weights)[::-1]
    sorted_channels = [channels[i] for i in sorted_indices]
    sorted_weights = avg_weights[sorted_indices]

    bars = plt.bar(range(len(channels)), sorted_weights)
    plt.xticks(range(len(channels)), sorted_channels, rotation=45)
    plt.xlabel("Channel (Sorted)")
    plt.ylabel("Attention Weight")
    plt.title("Channels Sorted by Importance")

    # Color code: top 5 in green, bottom 5 in red
    for i in range(len(bars)):
        if i < 5:
            bars[i].set_color('darkgreen')
        elif i >= len(bars) - 5:
            bars[i].set_color('darkred')
        else:
            bars[i].set_color('steelblue')

    # 3. Cumulative importance
    plt.subplot(2, 2, 3)
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    plt.plot(range(1, len(cumulative) + 1), cumulative, 'o-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    plt.xlabel("Number of Channels")
    plt.ylabel("Cumulative Importance")
    plt.title("How Many Channels Needed?")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Find how many channels give 80% of importance
    n_80 = np.argmax(cumulative >= 0.8) + 1
    plt.axvline(x=n_80, color='g', linestyle='--', alpha=0.5,
                label=f'Top {n_80} channels = 80%')

    # 4. Channel importance distribution
    plt.subplot(2, 2, 4)
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

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "channel_attention_analysis.png"), dpi=300)
    plt.close()

    return avg_weights


def analyze_channel_performance(channels, channel_weights):
    """
    Analyze what happens when we remove low-importance channels
    """
    print("\n" + "=" * 60)
    print("CHANNEL PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Sort channels by importance
    sorted_indices = np.argsort(channel_weights)[::-1]
    sorted_channels = [channels[i] for i in sorted_indices]
    sorted_weights = channel_weights[sorted_indices]

    # Print ranking
    print(f"\n{'Rank':<6} {'Channel':<10} {'Weight':<12} {'Importance':<15}")
    print("-" * 50)
    for rank, (ch, w) in enumerate(zip(sorted_channels, sorted_weights), 1):
        importance = ""
        if rank <= 5:
            importance = "CRITICAL"
        elif rank <= 10:
            importance = "IMPORTANT"
        elif rank <= 15:
            importance = "MODERATE"
        else:
            importance = "LOW"

        print(f"{rank:<6} {ch:<10} {w:<12.4f} {importance:<15}")

    # Analyze different channel subsets
    print(f"\n\nCHANNEL SUBSET ANALYSIS:")
    print("-" * 50)

    subsets = {
        "Top 5 channels": sorted_channels[:5],
        "Top 8 channels": sorted_channels[:8],
        "Top 10 channels": sorted_channels[:10],
        "Top 12 channels": sorted_channels[:12],
        "Above average (> mean)": [ch for ch, w in zip(channels, channel_weights)
                                   if w > np.mean(channel_weights)],
        "Top 50%": sorted_channels[:len(channels) // 2],
    }

    for name, subset in subsets.items():
        reduction = 100 * (1 - len(subset) / len(channels))
        avg_weight = np.mean([channel_weights[channels.index(ch)] for ch in subset])
        print(f"\n{name}:")
        print(f"  • Channels: {subset}")
        print(f"  • Count: {len(subset)}/{len(channels)} ({reduction:.1f}% reduction)")
        print(f"  • Avg weight: {avg_weight:.4f}")
        print(f"  • Total importance: {np.sum([channel_weights[channels.index(ch)] for ch in subset]):.4f}")

    return sorted_channels, sorted_weights


def save_channel_analysis_report(channels, weights, val_accuracy, save_path="."):
    """
    Save detailed channel analysis to file
    """
    report_path = os.path.join(save_path, "channel_analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CHANNEL ATTENTION ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"MODEL: ChannelAttentionGRU\n")  # Changed from ChannelAttentionLSTM to ChannelAttentionGRU
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Total Channels: {len(channels)}\n")
        f.write(f"Mean Attention Weight: {np.mean(weights):.4f}\n")
        f.write(f"Std Attention Weight: {np.std(weights):.4f}\n\n")

        f.write("CHANNEL IMPORTANCE RANKING:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Rank':<6} {'Channel':<10} {'Weight':<12} {'% of Max':<12}\n")
        f.write("-" * 50 + "\n")

        sorted_indices = np.argsort(weights)[::-1]
        max_weight = np.max(weights)

        for rank, idx in enumerate(sorted_indices, 1):
            channel = channels[idx]
            weight = weights[idx]
            percent = 100 * weight / max_weight
            f.write(f"{rank:<6} {channel:<10} {weight:<12.4f} {percent:<12.1f}%\n")

        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-" * 50 + "\n")

        # Calculate cumulative importance
        sorted_weights = weights[sorted_indices]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)

        n_80 = np.argmax(cumulative >= 0.8) + 1
        n_90 = np.argmax(cumulative >= 0.9) + 1

        f.write(f"1. For 80% of discriminative power: Use top {n_80} channels\n")
        f.write(f"2. For 90% of discriminative power: Use top {n_90} channels\n")
        f.write(f"3. Minimal setup (70% power): Use top {max(5, n_80 - 3)} channels\n\n")

        f.write("TOP CHANNELS TO FOCUS ON:\n")
        f.write("-" * 50 + "\n")
        for i in range(min(10, len(channels))):
            idx = sorted_indices[i]
            f.write(f"Channel {channels[idx]} (weight: {weights[idx]:.4f})\n")

        f.write("\nCHANNELS TO CONSIDER REMOVING:\n")
        f.write("-" * 50 + "\n")
        for i in range(1, min(6, len(channels))):
            idx = sorted_indices[-i]
            f.write(f"Channel {channels[idx]} (weight: {weights[idx]:.4f})\n")

    print(f"\n✅ Detailed channel analysis saved to: {report_path}")


# -----------------------
# TRAIN / EVAL FUNCTIONS (Simplified)
# -----------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0

    for X, y in loader:  # Simplified: only X, y
        X, y = X.to(DEVICE), y.to(DEVICE)
        out, attention_weights = model(X)  # Only pass X

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
        for X, y in loader:  # Simplified: only X, y
            X, y = X.to(DEVICE), y.to(DEVICE)
            out, attention_weights = model(X)  # Only pass X

            loss = criterion(out, y)
            total_loss += loss.item()
            total_acc += ((out > 0.5).float() == y).float().mean().item()

            preds.extend((out > 0.5).float().cpu().numpy())
            targets.extend(y.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())

    return total_loss / len(loader), total_acc / len(loader), np.array(preds), np.array(targets)


# -----------------------
# MAIN FUNCTION
# -----------------------
def main():
    print("Initializing Channel Attention Analysis...")
    print(f"Device: {DEVICE}")
    print(f"Using channels: {CHANNELS}")

    # Create output directory
    output_dir = "channel_attention_results"
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

    # Continue with original logic
    def get_subject_id(file_path):
        return os.path.basename(file_path)[:6]  # GaCo01 / GaPt02

    # Get subjects
    subjects = np.array([get_subject_id(f) for f in all_files])

    # Unique subjects
    unique_subjects = np.unique(subjects)

    # Extract class label from subject ID
    # subject format: GaCo01 → Co or Pt
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
    print("\n================ TRAIN/VAL SPLIT ================\n")

    print("TRAINING SUBJECTS:")
    for s in train_subj:
        print("  ", s)

    print("\nVALIDATION SUBJECTS:")
    for s in val_subj:
        print("  ", s)

    print("\nTRAINING FILES:")
    for f in train_files:
        print("  ", os.path.basename(f))

    print("\nVALIDATION FILES:")
    for f in val_files:
        print("  ", os.path.basename(f))

    print("\n=================================================\n")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")

    # Create datasets (simplified - no feature vector)
    train_ds = GaitDataset(train_files, SEQ_LEN, CHANNELS)
    val_ds = GaitDataset(val_files, SEQ_LEN, CHANNELS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Initialize model (simplified - no feature dimension)
    input_dim = len(CHANNELS)
    model = ChannelAttentionGRU(input_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)  # Changed to ChannelAttentionGRU

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING CHANNEL ATTENTION GRU")  # Changed from LSTM to GRU
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience, patience_counter = 10, 0
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
    print("ANALYZING CHANNEL ATTENTION")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    model.eval()

    # Final evaluation
    final_val_loss, final_val_acc, final_preds, final_targets = eval_epoch(model, val_loader, criterion)
    print(f"Final Validation - Loss: {final_val_loss:.4f}, Acc: {final_val_acc:.3f}")

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
    # CHANNEL ATTENTION ANALYSIS
    # -----------------------
    print("\n" + "=" * 60)
    print("COMPUTING CHANNEL IMPORTANCE")
    print("=" * 60)

    # Get channel attention weights
    channel_weights = visualize_channel_attention(model, val_loader, CHANNELS, output_dir)

    # Print channel importance ranking
    print("\nCHANNEL IMPORTANCE RANKING (from most to least important):")
    print("-" * 60)

    sorted_indices = np.argsort(channel_weights)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        channel = CHANNELS[idx]
        weight = channel_weights[idx]
        stars = "★" * min(5, int(weight * 10))  # Visual indicator
        print(f"Rank {rank:2d}: Channel {channel:2d} → Weight: {weight:.4f} {stars}")

    # Analyze channel performance
    sorted_channels, sorted_weights = analyze_channel_performance(CHANNELS, channel_weights)

    # Save detailed analysis report
    save_channel_analysis_report(CHANNELS, channel_weights, final_val_acc, output_dir)

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
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    # Find most and least important channels
    most_important_idx = np.argmax(channel_weights)
    least_important_idx = np.argmin(channel_weights)

    print(f"\n🎯 MOST IMPORTANT CHANNEL:")
    print(f"   Channel {CHANNELS[most_important_idx]} with weight {channel_weights[most_important_idx]:.4f}")
    print(
        f"   This channel contributes {100 * channel_weights[most_important_idx] / np.sum(channel_weights):.1f}% of total attention")

    print(f"\n⚠️ LEAST IMPORTANT CHANNEL:")
    print(f"   Channel {CHANNELS[least_important_idx]} with weight {channel_weights[least_important_idx]:.4f}")

    print(f"\n📊 CHANNEL IMPORTANCE RANGE:")
    print(f"   Highest weight: {np.max(channel_weights):.4f}")
    print(f"   Lowest weight:  {np.min(channel_weights):.4f}")
    print(f"   Ratio (high/low): {np.max(channel_weights) / np.min(channel_weights):.2f}x")

    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")

    # Calculate how many channels give 80% of importance
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    n_80 = np.argmax(cumulative >= 0.8) + 1
    top_channels = sorted_channels[:n_80]

    print(f"   1. For optimal performance: Use ALL {len(CHANNELS)} channels")
    print(f"   2. For efficient setup: Use top {n_80} channels (80% of discriminative power)")
    print(f"      Top channels: {top_channels}")
    print(f"   3. For minimal setup: Use top 5 channels: {sorted_channels[:5]}")

    print(f"\n✅ Analysis complete! All results saved to: {output_dir}/")
    print(f"   - channel_attention_analysis.png (visualizations)")
    print(f"   - channel_analysis_report.txt (detailed analysis)")
    print(f"   - classification_report.csv (performance metrics)")
    print(f"   - confusion_matrix.png (confusion matrix)")
    print(f"   - training_history.png (training curves)")

    # Save final channel weights for future use
    weights_df = pd.DataFrame({
        'channel': CHANNELS,
        'attention_weight': channel_weights,
        'rank': [np.where(sorted_indices == i)[0][0] + 1 for i in range(len(CHANNELS))]
    })
    weights_df.to_csv(os.path.join(output_dir, "channel_weights.csv"), index=False)

    return model, channel_weights


if __name__ == "__main__":
    model, channel_weights = main()