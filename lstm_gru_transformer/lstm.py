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
# ✅ Paths adjusted for Windows: relative to THIS script's location
RAW_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_oversampled"
FEATURE_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\merged_features.csv"

# ✅ Feature and channel selection
SELECTED_FEATURES = [

]

#CHANNELS = [2, 6, 7, 8, 9, 10, 14, 15, 16, 17]
#CHANNELS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
CHANNELS = [2,3,6,7,10,12,13,14,16,17]
#CHANNELS = [2,5,7,8,10,14,16,17,18,19]
# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-3
SEQ_LEN = 2000
HIDDEN_DIM = 64
NUM_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 42


# -----------------------
# DATASET
# -----------------------
class GaitDataset(Dataset):
    def __init__(self, file_paths, feature_df=None, selected_features=None, seq_len=2000, channels=None):
        self.file_paths = file_paths
        self.feature_df = feature_df
        self.selected_features = selected_features
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
        label = 0 if "Co" in fname else 1  # Co=control, Pt=parkinson
        data = np.loadtxt(file_path)
        data = data[:, [i - 1 for i in self.channels]]
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = self.pad_or_truncate(data)

        if self.feature_df is not None:
            subj_id = fname.split(".")[0]
            feat_row = self.feature_df.loc[self.feature_df['filename'] == subj_id, self.selected_features]
            feat_vec = feat_row.values.astype(np.float32)[0] if len(feat_row) > 0 else np.zeros(
                len(self.selected_features))
            return torch.tensor(data, dtype=torch.float32), torch.tensor(feat_vec, dtype=torch.float32), torch.tensor(
                label, dtype=torch.float32)
        else:
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# -----------------------
# MODEL
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, feature_dim=None, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        lstm_out_dim = hidden_dim * 2
        if feature_dim is not None:
            self.fc = nn.Sequential(
                nn.Linear(lstm_out_dim + feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

    def forward(self, x, feat_vec=None):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        if feat_vec is not None:
            out = torch.cat((out, feat_vec), dim=1)
        out = self.fc(out)
        return torch.sigmoid(out).squeeze()


# -----------------------
# TRAIN / EVAL LOOP
# -----------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in loader:
        if len(batch) == 3:
            X, feats, y = batch
            X, feats, y = X.to(DEVICE), feats.to(DEVICE), y.to(DEVICE)
            out = model(X, feats)
        else:
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += ((out > 0.5).float() == y).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X, feats, y = batch
                X, feats, y = X.to(DEVICE), feats.to(DEVICE), y.to(DEVICE)
                out = model(X, feats)
            else:
                X, y = batch
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            total_acc += ((out > 0.5).float() == y).float().mean().item()
            preds.extend((out > 0.5).float().cpu().numpy())
            targets.extend(y.cpu().numpy())
    return total_loss / len(loader), total_acc / len(loader), np.array(preds), np.array(targets)


# -----------------------
# VISUALIZATION UTILS
# -----------------------
def plot_results(preds, targets, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Control", "Parkinson"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

    report = classification_report(targets, preds, target_names=["Control", "Parkinson"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df[["precision", "recall", "f1-score", "support"]].round(3))
    report_df.to_csv(os.path.join(save_path, "classification_report.csv"))
    print(f"\nSaved confusion matrix and report in: {save_path}")


# -----------------------
# MAIN
# -----------------------
def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_PATH}")

    # FIXED: Comprehensive file discovery with debugging
    print(f"Scanning directory: {RAW_PATH}")

    # Find ALL .txt files (case-insensitive) in all subdirectories
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

    # Debug: Show file distribution
    control_files = [f for f in all_files if "Co" in os.path.basename(f)]
    patient_files = [f for f in all_files if "Pt" in os.path.basename(f)]
    other_files = [f for f in all_files if "Co" not in os.path.basename(f) and "Pt" not in os.path.basename(f)]

    print(f"Control files: {len(control_files)}")
    print(f"Patient files: {len(patient_files)}")
    print(f"Other files: {len(other_files)}")

    if len(all_files) == 0:
        raise RuntimeError(f"No .txt files found in {RAW_PATH}")

    # FIXED: Verify files are readable and have valid data
    valid_files = []
    for file_path in all_files:
        try:
            data = np.loadtxt(file_path)
            # Check if file has expected structure (time × sensors)
            if data.ndim == 2 and data.shape[1] >= 18:  # At least 18 columns (time + 17 sensors)
                valid_files.append(file_path)
            else:
                print(f"⚠️ Skipping {os.path.basename(file_path)} - unexpected shape: {data.shape}")
        except Exception as e:
            print(f"❌ Skipping {os.path.basename(file_path)} - read error: {e}")

    print(f"✅ Valid data files: {len(valid_files)}")

    if len(valid_files) == 0:
        raise RuntimeError("No valid data files found after verification!")

    all_files = valid_files

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

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    feature_df = pd.read_csv(FEATURE_PATH)
    if 'filename' not in feature_df.columns:
        feature_df['filename'] = feature_df.index.astype(str)

    train_ds = GaitDataset(train_files, feature_df, SELECTED_FEATURES, SEQ_LEN, CHANNELS)
    val_ds = GaitDataset(val_files, feature_df, SELECTED_FEATURES, SEQ_LEN, CHANNELS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    input_dim = len(CHANNELS)
    feature_dim = len(SELECTED_FEATURES)
    model = LSTMClassifier(input_dim, HIDDEN_DIM, NUM_LAYERS, feature_dim).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_val_acc = 0.0  # Track best accuracy too
    best_preds = None
    best_targets = None
    best_epoch = 0
    patience, patience_counter = 20, 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

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

        # Improved saving: track both loss and accuracy
        improvement = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement = True
        # if val_acc > best_val_acc:
        #   best_val_acc = val_acc
        #  improvement = True

        if improvement:
            patience_counter = 0
            # Saves both architecture and weights
            torch.save(model, "lstm_parkinson_full.pt")
            best_epoch = epoch + 1
            best_preds = preds.copy()
            best_targets = targets.copy()
            print(f"💾 Saved best model! Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ---- FIXED VISUALIZATION ----

    print(f"\n{'=' * 50}")
    print(f"Loading best model from epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'=' * 50}\n")

    # Load the saved best model
    model = torch.load("lstm_parkinson_full.pt", weights_only=False)
    model.to(DEVICE)
    model.eval()

    # Re-evaluate on validation set with the best model
    print("Re-evaluating with best model...")
    final_val_loss, final_val_acc, final_preds, final_targets = eval_epoch(model, val_loader, criterion)
    print(f"Final Results - Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.3f}")

    # Update best_preds and best_targets with the re-evaluated results
    best_preds = final_preds
    best_targets = final_targets
    try:
        # Re-import with Agg backend to prevent display issues
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def plot_results(preds, targets, save_path="results"):
            os.makedirs(save_path, exist_ok=True)

            try:
                # Calculate confusion matrix
                cm = confusion_matrix(targets, preds)

                # Create confusion matrix plot
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Control', 'Parkinson'])
                plt.yticks(tick_marks, ['Control', 'Parkinson'])

                # Add text annotations
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved confusion matrix to: {os.path.join(save_path, 'confusion_matrix.png')}")

            except Exception as e:
                print(f"❌ Error creating confusion matrix: {e}")
                return

            try:
                # Classification report
                report = classification_report(best_targets, best_preds, target_names=["Control", "Parkinson"],
                                               output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                print("\nClassification Report:")
                print(report_df[["precision", "recall", "f1-score", "support"]].round(3))
                report_df.to_csv(os.path.join(save_path, "classification_report.csv"))
                print(f"✅ Saved classification report to: {os.path.join(save_path, 'classification_report.csv')}")

            except Exception as e:
                print(f"❌ Error saving classification report: {e}")

        # Call the plotting function
        plot_results(best_preds, best_targets)

        # Plot accuracy and loss curves with best epoch marker
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
        plt.plot(history["val_acc"], label="Val Accuracy", linewidth=2)
        if best_epoch > 0:
            plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("accuracy_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved accuracy_curve.png")

        plt.figure(figsize=(8, 5))
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
        plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved loss_curve.png")

    except Exception as e:
        print(f"Visualization error: {e}")

    # Final file check
    import glob
    print("\ncreated files:")
    created_files = glob.glob("*.png") + glob.glob("*.pt") + glob.glob("results/*")
    if created_files:
        for file in created_files:
            print(f" {file}")
        print(f"Total files created: {len(created_files)}")
    else:
        print("No output files were created!")

    print(f"\nModel saved as: lstm_parkinson.pt")
    print("Training complete!")


if __name__ == "__main__":
    main()