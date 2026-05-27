# classification/random_forest_fixed.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed\merged_features.csv"
OUTPUT_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\classification\classification_results"


def load_and_prepare_data(csv_path, selected_features):
    """Load data with proper validation."""
    print("📥 Loading data...")
    df = pd.read_csv(csv_path)

    # Check if selected features exist
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]

    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")

    if not available_features:
        raise ValueError("❌ No selected features found in the dataset!")

    print(f"✅ Using {len(available_features)} features")

    # Extract features and labels
    X = df[available_features]
    y = df['group']

    # Verify class distribution
    class_counts = y.value_counts()
    print(f"🎯 Class distribution: {dict(class_counts)}")

    # Handle missing values
    X = X.fillna(X.mean())

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le.classes_


def run_random_forest_evaluation(X, y, class_names, feature_set_name):
    """Run Random Forest evaluation with cross-validation."""
    print(f"\n🌲 Running Random Forest evaluation for {feature_set_name}...")

    # Define Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1
    )

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation scores
    cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')

    # Get predictions for confusion matrix
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Store results
    results = {
        'model': rf_model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
        'classification_report': classification_report(y_true_all, y_pred_all,
                                                       target_names=class_names, output_dict=True)
    }

    print(f"   Cross-val Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return results


def plot_feature_importance(rf_model, feature_names, top_n=15):
    """Plot the most important features."""
    importance_scores = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(top_n)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features\n(Random Forest)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    return feature_importance


def plot_random_forest_results(results, feature_importance, class_names, feature_set_name):
    """Create comprehensive visualizations for Random Forest."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Random Forest Performance\n(5-Fold Cross-Validation)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Cross-validation performance
    cv_scores = results['cv_scores']
    axes[0, 0].boxplot([cv_scores], labels=['Random Forest'], widths=0.6)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'Cross-Validation Performance\nAccuracy: {results["cv_mean"]:.3f} ± {results["cv_std"]:.3f}')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)

    # Add individual fold scores as points
    for i, score in enumerate(cv_scores):
        axes[0, 0].scatter(1, score, color='red', zorder=3, s=50)

    # Plot 2: Confusion matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix\n(All CV folds)')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')

    # Plot 3: Feature importance (top 10)
    top_10_features = feature_importance.head(10)
    y_pos = np.arange(len(top_10_features))
    axes[1, 0].barh(y_pos, top_10_features['importance'])
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([f[:25] + '...' if len(f) > 25 else f
                                for f in top_10_features['feature']])
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].set_title('Top 10 Feature Importances')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Plot 4: Performance metrics summary - FIXED VERSION
    report = results['classification_report']
    metrics_text = f"Overall Accuracy: {results['cv_mean']:.3f}\n\n"

    # Use class names instead of numbers
    control_name = class_names[0]  # 'control'
    patient_name = class_names[1]  # 'patient'

    metrics_text += f"{control_name} Class:\n"
    metrics_text += f"  Precision: {report[control_name]['precision']:.3f}\n"
    metrics_text += f"  Recall:    {report[control_name]['recall']:.3f}\n"
    metrics_text += f"  F1-Score:  {report[control_name]['f1-score']:.3f}\n\n"
    metrics_text += f"{patient_name} Class:\n"
    metrics_text += f"  Precision: {report[patient_name]['precision']:.3f}\n"
    metrics_text += f"  Recall:    {report[patient_name]['recall']:.3f}\n"
    metrics_text += f"  F1-Score:  {report[patient_name]['f1-score']:.3f}"

    axes[1, 1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                    fontsize=11, transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Detailed Performance Metrics')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'random_forest_{feature_set_name.replace(" ", "_").lower()}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function for Random Forest classification."""

    # === SELECT YOUR FEATURES ===
    selected_features = [
        'L2__maximum', 'L2__absolute_maximum', 'R2__maximum', 'R2__absolute_maximum', 'R6__median', 'L6__median',
        'R5__median', 'L8__median', 'L2__standard_deviation'
    ]

    feature_set_name = "Selected Features"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load data
        X, y, class_names = load_and_prepare_data(DATA_PATH, selected_features)

        # Run Random Forest evaluation
        results = run_random_forest_evaluation(X, y, class_names, feature_set_name)

        # Get feature importance
        feature_importance = plot_feature_importance(results['model'], selected_features)

        # Plot results
        plot_random_forest_results(results, feature_importance, class_names, feature_set_name)

        # Print detailed summary
        print("\n" + "=" * 60)
        print("🌲 RANDOM FOREST RESULTS SUMMARY")
        print("=" * 60)
        print(f"📊 Cross-val Accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"🎯 Best fold: {max(results['cv_scores']):.3f}")
        print(f"📉 Worst fold: {min(results['cv_scores']):.3f}")

        cm = results['confusion_matrix']
        print(f"\n📋 Confusion Matrix:")
        print(f"   Controls: {cm[0, 0]} correct, {cm[0, 1]} misclassified")
        print(f"   Patients: {cm[1, 1]} correct, {cm[1, 0]} misclassified")

        print(f"\n🔝 Top 5 Most Important Features:")
        top_5 = feature_importance.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

        # Print classification report
        print(f"\n📈 Detailed Classification Report:")
        report = results['classification_report']
        for class_name in class_names:
            print(f"   {class_name}:")
            print(f"     Precision: {report[class_name]['precision']:.3f}")
            print(f"     Recall:    {report[class_name]['recall']:.3f}")
            print(f"     F1-Score:  {report[class_name]['f1-score']:.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()