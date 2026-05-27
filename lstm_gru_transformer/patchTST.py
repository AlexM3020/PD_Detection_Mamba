import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForClassification
from tqdm import tqdm
import json

# ========================
# Configuration
# ========================
RAW_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\raw\gaitpdb_unzipped\enlarged_dataset_oversampled"
OUTPUT_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\lstm_gru_transformer\patchtst_model"
CHANNELS = [2, 5, 6, 8, 10, 13, 14, 16, 17, 18]

# Training hyperparameters
BATCH_SIZE = 16  # Start smaller for stability, increase if GPU memory allows
EPOCHS = 100  # Transformers often need more epochs
LEARNING_RATE = 5e-4  # Good starting point for transformers
WARMUP_EPOCHS = 10  # Gradually increase learning rate
WEIGHT_DECAY = 0.01  # L2 regularization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PatchTST hyperparameters
# Since your files are ~2000 timesteps, use full length or slightly less
CONTEXT_LENGTH = 2000  # Use full sequence length from your data
NUM_INPUT_CHANNELS = len(CHANNELS)
NUM_CLASSES = 2

# Early stopping
PATIENCE = 15  # Stop if no improvement for 15 epochs

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Number of channels: {NUM_INPUT_CHANNELS}")
print(f"Selected channels: {CHANNELS}")

# ========================
# Data Discovery
# ========================
print(f"\nScanning directory: {RAW_PATH}")


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

# File distribution
control_files = [f for f in all_files if "Co" in os.path.basename(f)]
patient_files = [f for f in all_files if "Pt" in os.path.basename(f)]

print(f"Control files: {len(control_files)}")
print(f"Patient files: {len(patient_files)}")

# Validate files
valid_files = []
for file_path in all_files:
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 2 and data.shape[1] >= 18:
            valid_files.append(file_path)
        else:
            print(f"⚠️ Skipping {os.path.basename(file_path)} - unexpected shape: {data.shape}")
    except Exception as e:
        print(f"❌ Skipping {os.path.basename(file_path)} - read error: {e}")

print(f"✅ Valid data files: {len(valid_files)}")

if len(valid_files) == 0:
    raise RuntimeError("No valid data files found!")

all_files = valid_files


# ========================
# Train/Val Split (Subject-level)
# ========================
def get_subject_id(file_path):
    return os.path.basename(file_path)[:6]


subjects = np.array([get_subject_id(f) for f in all_files])
unique_subjects = np.unique(subjects)
subject_labels = np.array([1 if s[2:4] == "Pt" else 0 for s in unique_subjects])

train_subj, val_subj = train_test_split(
    unique_subjects,
    test_size=0.20,
    random_state=42,
    stratify=subject_labels
)

train_files = [f for f in all_files if get_subject_id(f) in train_subj]
val_files = [f for f in all_files if get_subject_id(f) in val_subj]

print(f"\nTrain subjects: {len(train_subj)}, Files: {len(train_files)}")
print(f"Val subjects: {len(val_subj)}, Files: {len(val_files)}")


# ========================
# Dataset Class
# ========================
class GaitDataset(Dataset):
    def __init__(self, file_paths, channels, context_length):
        self.file_paths = file_paths
        self.channels = channels
        self.context_length = context_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load data
        data = np.loadtxt(file_path)

        # Select channels (column indices)
        data = data[:, self.channels]

        # Pad or truncate to context_length
        if data.shape[0] < self.context_length:
            # Pad with zeros
            padding = np.zeros((self.context_length - data.shape[0], data.shape[1]))
            data = np.vstack([data, padding])
        else:
            # Truncate
            data = data[:self.context_length, :]

        # Transpose to (channels, time) for PatchTST
        data = data.T  # Shape: (num_channels, context_length)

        # Get label from filename
        label = 1 if "Pt" in os.path.basename(file_path) else 0

        return {
            'past_values': torch.FloatTensor(data),
            'labels': torch.LongTensor([label])
        }


# Create datasets
train_dataset = GaitDataset(train_files, CHANNELS, CONTEXT_LENGTH)
val_dataset = GaitDataset(val_files, CHANNELS, CONTEXT_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ========================
# Model Configuration
# ========================
config = PatchTSTConfig(
    num_input_channels=NUM_INPUT_CHANNELS,
    context_length=CONTEXT_LENGTH,

    # Patch parameters - CRITICAL for performance
    patch_length=256,  # Larger patches = fewer tokens, faster training
    stride=128,  # stride < patch_length creates overlap (smoother)

    # Model architecture
    num_hidden_layers=1,  # Start with 3 layers (2-4 is typical)
    d_model=16,  # Hidden dimension (64, 128, 256 are common)
    num_attention_heads=2,  # Should divide d_model evenly
    ffn_dim=32,  # Usually 2-4x d_model

    # Regularization - IMPORTANT to prevent overfitting
    dropout=0.1,  # General dropout
    head_dropout=0.1,  # Dropout on classification head
    attention_dropout=0.1,  # Dropout in attention mechanism
    path_dropout=0.1,  # Stochastic depth

    # Output
    pooling_type="mean",  # How to aggregate patches: "mean" or "max"
    num_labels=NUM_CLASSES,

    # Normalization
    norm_type="batchnorm",  # "batchnorm" or "layernorm"
)

model = PatchTSTForClassification(config)
model = model.to(DEVICE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Effective sequence length after patching: {(CONTEXT_LENGTH - config.patch_length) // config.stride + 1}")

# ========================
# Training Setup
# ========================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)  # Default Adam betas work well
)


# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
criterion = torch.nn.CrossEntropyLoss()

# Early stopping
best_val_acc = 0.0
patience_counter = 0

# Tracking metrics
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rate': []
}

# ========================
# Training Loop
# ========================
print("\n" + "=" * 50)
print("Starting Training")
print("=" * 50)

best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train] LR={current_lr:.2e}")
    for batch in pbar:
        past_values = batch['past_values'].to(DEVICE)
        labels = batch['labels'].squeeze().to(DEVICE)

        optimizer.zero_grad()
        past_values = past_values.permute(0, 2, 1)
        outputs = model(past_values=past_values)

        outputs = model(past_values=past_values)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping - helps with training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        for batch in pbar:
            past_values = batch['past_values'].to(DEVICE)
            labels = batch['labels'].squeeze().to(DEVICE)
            past_values = past_values.permute(0, 2, 1)
            outputs = model(past_values=past_values)

            outputs = model(past_values=past_values)
            logits = outputs.logits

            loss = criterion(logits, labels)
            val_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Update learning rate
    scheduler.step()

    # Store metrics
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rate'].append(current_lr)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"LR: {current_lr:.2e}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        print(f"✅ New best validation accuracy: {best_val_acc:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'config': config.to_dict()
        }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
        break

# ========================
# Save Final Model
# ========================
model.save_pretrained(OUTPUT_DIR)
config.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Model saved to {OUTPUT_DIR}")

# Save training history
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=4)

# ========================
# Final Evaluation
# ========================
print("\n" + "=" * 50)
print("Final Evaluation on Validation Set")
print("=" * 50)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        past_values = batch['past_values'].to(DEVICE)
        labels = batch['labels'].squeeze().to(DEVICE)
        past_values = past_values.permute(0, 2, 1)
        outputs = model(past_values=past_values)

        outputs = model(past_values=past_values)
        logits = outputs.logits

        _, predicted = torch.max(logits, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
class_names = ['Control', 'Parkinson']
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\nClassification Report:")
print(report_df)

# Save report
report_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# ========================
# Plot Training Curves
# ========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
axes[1].plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate curve
axes[2].plot(history['learning_rate'], marker='o', markersize=3, color='green')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate Schedule')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
print(f"\n✅ Training curves saved to {os.path.join(OUTPUT_DIR, 'training_curves.png')}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=class_names, yticklabels=class_names,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved to {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")

print(f"\n{'=' * 50}")
print(f"Training Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"{'=' * 50}")