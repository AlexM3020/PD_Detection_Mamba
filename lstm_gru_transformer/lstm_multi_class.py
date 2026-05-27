import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
# -----------------------
# CONFIG
# -----------------------
# ✅ Paths adjusted for Windows: relative to THIS script's location
RAW_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset"
LABELS_CSV_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\dataset_labels.csv"

# ✅ Channel selection
CHANNELS = [3, 6, 7, 9, 11, 14, 15, 17, 18, 19]

# Hyperparameters
BATCH_SIZE = 30
EPOCHS = 200
LR = 5e-3
SEQ_LEN = 2000
HIDDEN_DIM = 64
NUM_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 42
NUM_CLASSES = 3  # 0, 1, 2


# -----------------------
# DATASET
# -----------------------
class GaitDataset(Dataset):
    def __init__(self, file_paths, labels_dict, seq_len=2000, channels=None):
        self.file_paths = file_paths
        self.labels_dict = labels_dict  # Dictionary mapping filename to label
        self.seq_len = seq_len
        self.channels = channels

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

        # Get label from dictionary instead of hardcoded logic
        label = self.labels_dict.get(fname, 0)  # Default to 0 if not found

        data = np.loadtxt(file_path)
        data = data[:, [i - 1 for i in self.channels]]
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = self.pad_or_truncate(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# -----------------------
# MODEL
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        lstm_out_dim = hidden_dim * 2

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out


# -----------------------
# TRAIN / EVAL LOOP
# -----------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in loader:
        X, y = batch
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        total_acc += (preds == y).float().mean().item()

    return total_loss / len(loader), total_acc / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)

            loss = criterion(out, y)
            total_loss += loss.item()

            batch_preds = torch.argmax(out, dim=1)
            total_acc += (batch_preds == y).float().mean().item()

            preds.extend(batch_preds.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return total_loss / len(loader), total_acc / len(loader), np.array(preds), np.array(targets)


# -----------------------
# VISUALIZATION UTILS
# -----------------------
def plot_results(preds, targets, save_path="results"):
    os.makedirs(save_path, exist_ok=True)

    # Class names for the 3 groups
    class_names = ["Control", "Group 1", "Group 2+3"]

    try:
        # Confusion Matrix
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Multi-class")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved confusion matrix to: {os.path.join(save_path, 'confusion_matrix.png')}")

    except Exception as e:
        print(f"❌ Error creating confusion matrix: {e}")

    try:
        # Classification Report
        report = classification_report(targets, preds, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        print("\nClassification Report:")
        print(report_df[["precision", "recall", "f1-score", "support"]].round(3))
        report_df.to_csv(os.path.join(save_path, "classification_report.csv"))
        print(f"✅ Saved classification report to: {os.path.join(save_path, 'classification_report.csv')}")

    except Exception as e:
        print(f"❌ Error saving classification report: {e}")


def plot_training_history(history, best_epoch, save_path="."):
    """Plot training and validation accuracy/loss curves"""
    try:
        # Accuracy curve
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
        plt.plot(history["val_acc"], label="Val Accuracy", linewidth=2)
        if best_epoch > 0:
            plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Loss curve
        plt.subplot(1, 2, 2)
        plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
        if best_epoch > 0:
            plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "results_multiclass/training_history.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved training_history.png")

    except Exception as e:
        print(f"❌ Error plotting training history: {e}")


# -----------------------
# MAIN
# -----------------------
def main():
    # Set matplotlib backend to avoid display issues
    import matplotlib
    matplotlib.use('Agg')

    print(f"Using device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_PATH}")

    # Load labels from CSV
    print("Loading labels from CSV...")
    labels_df = pd.read_csv(LABELS_CSV_PATH)

    # Create filename to label mapping
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    print("Label distribution in dataset:")
    label_counts = labels_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Class {label}: {count} samples")

    # Find all txt files
    def find_all_txt_files(root_dir):
        txt_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    full_path = os.path.join(root, file)
                    txt_files.append(full_path)
        return txt_files

    print("Scanning for data files...")
    all_files = find_all_txt_files(RAW_PATH)
    print(f"Found {len(all_files)} .txt files total")

    # Filter files that exist in our labels CSV and have valid data
    valid_files = []
    valid_labels = []

    print("Validating files...")
    for file_path in all_files:
        fname = os.path.basename(file_path)
        if fname in labels_dict:
            try:
                data = np.loadtxt(file_path)
                if data.ndim == 2 and data.shape[1] >= max(CHANNELS):
                    valid_files.append(file_path)
                    valid_labels.append(labels_dict[fname])
                else:
                    print(f"⚠️ Skipping {fname} - unexpected shape: {data.shape}")
            except Exception as e:
                print(f"❌ Skipping {fname} - read error: {e}")
        else:
            print(f"⚠️ Skipping {fname} - not found in labels CSV")

    print(f"✅ Valid data files with labels: {len(valid_files)}")

    if len(valid_files) == 0:
        raise RuntimeError("No valid data files found!")

    # Split data with stratification
    # Subject-based split with stratification
    print("Splitting data by subjects...")

    def get_subject_id(file_path):
        """Extract subject ID from filename (e.g., GaCo01 from GaCo01_01.txt)"""
        return os.path.basename(file_path)[:6]  # First 6 characters

    # Get subjects for all valid files
    subjects = np.array([get_subject_id(f) for f in valid_files])
    unique_subjects = np.unique(subjects)

    # Get label for each unique subject (use first file's label)
    subject_labels_dict = {}
    for subj in unique_subjects:
        # Find first file for this subject
        subj_files = [f for f in valid_files if get_subject_id(f) == subj]
        fname = os.path.basename(subj_files[0])
        subject_labels_dict[subj] = labels_dict[fname]

    subject_labels = np.array([subject_labels_dict[s] for s in unique_subjects])

    # Split by subjects with stratification
    train_subj, val_subj = train_test_split(
        unique_subjects,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=subject_labels
    )

    print("\n================ TRAIN/VAL SPLIT ================")
    print(f"\nTRAINING SUBJECTS ({len(train_subj)}):")
    for s in sorted(train_subj):
        print(f"  {s} (label: {subject_labels_dict[s]})")

    print(f"\nVALIDATION SUBJECTS ({len(val_subj)}):")
    for s in sorted(val_subj):
        print(f"  {s} (label: {subject_labels_dict[s]})")
    print("\n=================================================\n")

    # Assign files to splits based on subject
    train_files = [f for f in valid_files if get_subject_id(f) in train_subj]
    val_files = [f for f in valid_files if get_subject_id(f) in val_subj]

    print(f"Training set: {len(train_files)} files from {len(train_subj)} subjects")
    print(f"Validation set: {len(val_files)} files from {len(val_subj)} subjects")

    # Create labels dictionaries for datasets
    train_labels_dict = {os.path.basename(f): labels_dict[os.path.basename(f)] for f in train_files}
    val_labels_dict = {os.path.basename(f): labels_dict[os.path.basename(f)] for f in val_files}

    train_ds = GaitDataset(train_files, train_labels_dict, SEQ_LEN, CHANNELS)
    val_ds = GaitDataset(val_files, val_labels_dict, SEQ_LEN, CHANNELS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Initialize model
    input_dim = len(CHANNELS)

    print(f"Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Number of LSTM layers: {NUM_LAYERS}")

    model = LSTMClassifier(input_dim, HIDDEN_DIM, NUM_LAYERS, num_classes=NUM_CLASSES).to(DEVICE)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_preds = None
    best_targets = None
    best_epoch = 0
    patience, patience_counter = 20, 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, targets = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}: Train loss {tr_loss:.4f}, acc {tr_acc:.3f} | Val loss {val_loss:.4f}, acc {val_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model, "results_multiclass/lstm_multiclass_full.pt")
            best_epoch = epoch + 1
            best_preds = preds.copy()
            best_targets = targets.copy()
            print(f"💾 Saved best model! Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final evaluation with best model
    print(f"\n{'=' * 50}")
    print(f"Loading best model from epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"{'=' * 50}\n")

    # Load the saved best model
    if os.path.exists("results_multiclass/lstm_multiclass_full.pt"):
        model = torch.load("results_multiclass/lstm_multiclass_full.pt", weights_only=False)
        model.to(DEVICE)
        model.eval()

        # Re-evaluate on validation set with the best model
        print("Re-evaluating with best model...")
        final_val_loss, final_val_acc, final_preds, final_targets = eval_epoch(model, val_loader, criterion)
        print(f"Final Results - Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.3f}")

        # Update best_preds and best_targets with the re-evaluated results
        best_preds = final_preds
        best_targets = final_targets

    # Generate plots and reports
    print("\nGenerating results...")
    plot_results(best_preds, best_targets)
    plot_training_history(history, best_epoch)

    # Final file check
    import glob
    print("\nCreated files:")
    created_files = glob.glob("*.png") + glob.glob("*.pt") + glob.glob("results/*")
    if created_files:
        for file in created_files:
            print(f"  {file}")
        print(f"Total files created: {len(created_files)}")
    else:
        print("No output files were created!")

    print(f"\nModel saved as: lstm_multiclass_full.pt")
    print("Multi-class training complete!")


if __name__ == "__main__":
    main()