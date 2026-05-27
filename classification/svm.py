# classification/svm_fixed.py
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\merged_features.csv"
OUTPUT_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\classification\classification_results_svm"


def load_and_prepare_data(csv_path, selected_features):
    """Load data with proper validation."""
    print("📥 Loading data...")
    df = pd.read_csv(csv_path)

    # Verify we have the expected dataset size
    print(f"Original dataset: {len(df)} samples")

    # Check if selected features exist
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]

    if missing_features:
        print(f"Missing features: {missing_features}")

    if not available_features:
        raise ValueError("No selected features found in the dataset!")

    print(f"Using {len(available_features)} features")

    # Extract features and labels
    X = df[available_features]
    y = df['group']

    # Verify class distribution
    class_counts = y.value_counts()
    print(f"Class distribution: {dict(class_counts)}")

    # Handle missing values
    X = X.fillna(X.mean())

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le.classes_


def run_proper_evaluation(X, y, class_names, feature_set_name):
    """Run proper evaluation without data leakage."""
    print(f"\nRunning PROPER evaluation for {feature_set_name}...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        'Linear SVM': SVC(kernel='linear', random_state=42),
        'RBF SVM': SVC(kernel='rbf', random_state=42)
    }

    # Use ONLY cross-validation for fair comparison (no separate test set)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

        # Also get predictions for confusion matrix
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in cv.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        # Store results
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
            'classification_report': classification_report(y_true_all, y_pred_all,
                                                           target_names=class_names, output_dict=True)
        }

        print(f"   Cross-val Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Full dataset CM shape: {results[name]['confusion_matrix'].shape}")

    return results, scaler


def plot_proper_results(results, feature_set_name, class_names):
    """Create proper visualizations."""
    # First figure: Original plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle(f'SVM Performance\n(5-Fold Cross-Validation)',
                  fontsize=16, fontweight='bold')

    # Plot 1: Model comparison
    model_names = list(results.keys())
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]

    bars = axes1[0, 0].bar(model_names, cv_means, yerr=cv_stds,
                           capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    axes1[0, 0].set_ylabel('Accuracy')
    axes1[0, 0].set_title('Cross-Validation Performance\n(5-Fold, Mean ± STD)')
    axes1[0, 0].set_ylim(0, 1)
    axes1[0, 0].grid(True, alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, cv_means):
        axes1[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.3f}', ha='center', va='bottom')

    # Plot 2: Confusion matrices (from full cross-validation)
    for idx, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        ax = axes1[idx, 1]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{name}\nConfusion Matrix\n(All CV folds)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Verify matrix sums to expected values
        total_samples = np.sum(cm)
        print(f"   {name} CM - Total samples: {total_samples}")

    # Plot 3: Cross-validation scores distribution
    all_cv_scores = [results[name]['cv_scores'] for name in model_names]
    box_plot = axes1[1, 0].boxplot(all_cv_scores, labels=model_names, widths=0.6)
    axes1[1, 0].set_ylabel('Accuracy')
    axes1[1, 0].set_title('Cross-Validation Scores Distribution\n(5 folds)')
    axes1[1, 0].grid(True, alpha=0.3)

    # Plot 4: Detailed metrics
    metrics_text = f"Dataset: {np.sum(results[model_names[0]]['confusion_matrix'])} samples\n"
    metrics_text += f"Classes: {class_names}\n"
    metrics_text += f"Best: {max(results.items(), key=lambda x: x[1]['cv_mean'])[0]}\n"
    metrics_text += f"Accuracy: {max([r['cv_mean'] for r in results.values()]):.3f}"

    axes1[1, 1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                     fontsize=12, transform=axes1[1, 1].transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes1[1, 1].set_title('Evaluation Summary')
    axes1[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'svm_proper_{feature_set_name.replace(" ", "_").lower()}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Second figure: Detailed metrics table only
    fig2, ax = plt.subplots(figsize=(14, 8))
    fig2.suptitle(f'SVM Detailed Performance Metrics\n{feature_set_name}',
                  fontsize=16, fontweight='bold')
    ax.axis('off')

    # Create detailed metrics table - COMPACT VERSION
    table_data = []
    columns = ['Model', 'Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

    for name, result in results.items():
        report = result['classification_report']
        accuracy = result['cv_mean']

        # Get metrics for both classes
        control_precision = report[class_names[0]]['precision']
        control_recall = report[class_names[0]]['recall']
        control_f1 = report[class_names[0]]['f1-score']

        patient_precision = report[class_names[1]]['precision']
        patient_recall = report[class_names[1]]['recall']
        patient_f1 = report[class_names[1]]['f1-score']

        # Add rows for both classes - NO EMPTY ROWS
        table_data.append([name, class_names[0], f"{accuracy:.3f}",
                           f"{control_precision:.3f}", f"{control_recall:.3f}", f"{control_f1:.3f}"])
        table_data.append([name, class_names[1], f"{accuracy:.3f}",
                           f"{patient_precision:.3f}", f"{patient_recall:.3f}", f"{patient_f1:.3f}"])

    # Create table with better sizing
    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    # Auto-adjust font size and cell dimensions
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)  # Wider and taller cells

    # Style the table headers
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Alternate row colors for better readability
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(columns)):
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'svm_detailed_metrics_{feature_set_name.replace(" ", "_").lower()}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function with better feature selection."""

    # === BETTER FEATURE SELECTION ===
    selected_features = [
        'L2__maximum', 'L2__absolute_maximum', 'R2__maximum', 'R2__absolute_maximum', 'R6__median', 'L6__median',
        'R5__median', 'L8__median', 'L2__standard_deviation'
    ]

    feature_set_name = "Manual + TSFresh Features"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load data
        X, y, class_names = load_and_prepare_data(DATA_PATH, selected_features)

        # Run proper evaluation
        results, scaler = run_proper_evaluation(X, y, class_names, feature_set_name)

        # Plot results - removed selected_features parameter since we don't need it anymore
        plot_proper_results(results, feature_set_name, class_names)

        # Print detailed summary with all metrics
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)

        for name, result in results.items():
            print(f"\n🎯 {name}:")
            print(f"   CV Accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")

            # Print detailed metrics
            report = result['classification_report']
            print(f"   {class_names[0]} Class:")
            print(f"     Precision: {report[class_names[0]]['precision']:.3f}")
            print(f"     Recall:    {report[class_names[0]]['recall']:.3f}")
            print(f"     F1-Score:  {report[class_names[0]]['f1-score']:.3f}")
            print(f"   {class_names[1]} Class:")
            print(f"     Precision: {report[class_names[1]]['precision']:.3f}")
            print(f"     Recall:    {report[class_names[1]]['recall']:.3f}")
            print(f"     F1-Score:  {report[class_names[1]]['f1-score']:.3f}")

            cm = result['confusion_matrix']
            print(f"   Confusion Matrix Total: {np.sum(cm)} samples")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()