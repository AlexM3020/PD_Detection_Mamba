# improved_tsne_visualization.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# === CONFIG ===
PROCESSED_DIR = r"C:\Users\Alex\OneDrive\Desktop\mamba\data\processed"
FEATURES_CSV = os.path.join(PROCESSED_DIR, "merged_features.csv")


def analyze_feature_importance(features, labels, feature_names, top_n=15):
    """
    Analyze which features are most discriminative.
    Returns both statistical and information-theoretic rankings.
    This function identifies which features best distinguish Parkinson's patients from controls using three statistical methods:
    Mutual Information (MI): Measures nonlinear relationships (most important for complex biological data)
    ANOVA F-statistic: Measures linear discriminative power
    Cohen's d: Measures effect size (mean difference relative to variability)
    """
    # Convert labels to numeric
    label_map = {'control': 0, 'patient': 1}
    y_numeric = labels.map(label_map)

    # 1. Mutual Information (captures nonlinear relationships)
    mi_scores = mutual_info_classif(features, y_numeric, random_state=42)
    mi_ranking = pd.DataFrame({
        'feature': feature_names,
        'MI_score': mi_scores
    }).sort_values('MI_score', ascending=False)

    # 2. ANOVA F-statistic (linear discriminative power)
    f_scores, p_values = f_classif(features, y_numeric)
    f_ranking = pd.DataFrame({
        'feature': feature_names,
        'F_score': f_scores,
        'p_value': p_values
    }).sort_values('F_score', ascending=False)

    # 3. Effect size (Cohen's d) for top features
    control_mask = labels == 'control'
    patient_mask = labels == 'patient'

    effect_sizes = []
    for i, fname in enumerate(feature_names):
        control_vals = features[:, i][control_mask]
        patient_vals = features[:, i][patient_mask]

        mean_diff = np.mean(patient_vals) - np.mean(control_vals)
        pooled_std = np.sqrt((np.var(control_vals) + np.var(patient_vals)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        effect_sizes.append(abs(cohens_d))

    effect_ranking = pd.DataFrame({
        'feature': feature_names,
        'cohens_d': effect_sizes
    }).sort_values('cohens_d', ascending=False)

    # Combine rankings with weighted scoring
    combined = mi_ranking.merge(f_ranking, on='feature').merge(effect_ranking, on='feature')

    # Define weights for each metric (adjust these as needed)
    mi_weight = 0  # Mutual Information weight
    f_weight = 0.5  # F-score weight
    cohen_weight = 0.5  # Cohen's d weight

    # Normalize scores to 0-1 range for fair comparison
    combined['MI_norm'] = (combined['MI_score'] - combined['MI_score'].min()) / (
            combined['MI_score'].max() - combined['MI_score'].min())
    combined['F_norm'] = (combined['F_score'] - combined['F_score'].min()) / (
            combined['F_score'].max() - combined['F_score'].min())
    combined['cohens_d_norm'] = (combined['cohens_d'] - combined['cohens_d'].min()) / (
            combined['cohens_d'].max() - combined['cohens_d'].min())

    # Calculate weighted combined score
    combined['combined_score'] = (
            combined['MI_norm'] * mi_weight +
            combined['F_norm'] * f_weight +
            combined['cohens_d_norm'] * cohen_weight
    )

    # Also keep the rank-based method as backup
    combined['combined_rank'] = (
            combined['MI_score'].rank(ascending=False) +
            combined['F_score'].rank(ascending=False) +
            combined['cohens_d'].rank(ascending=False)
    )

    # Sort by combined score (primary) and rank (secondary)
    combined = combined.sort_values(['combined_score', 'combined_rank'], ascending=[False, True])

    print("\n" + "=" * 80)
    print("TOP DISCRIMINATIVE FEATURES FOR PD DETECTION")
    print("=" * 80)
    print(f"Feature Weights: MI={mi_weight}, F-score={f_weight}, Cohen's d={cohen_weight}")
    cohens_d_label = "Cohen's d"
    print(f"\n{'Feature':<35} {'Combined':<8} {'MI Score':<10} {'F-Score':<10} {cohens_d_label:<10} {'p-value':<10}")
    print("-" * 80)

    for idx, row in combined.head(top_n).iterrows():
        print(f"{row['feature']:<35} {row['combined_score']:>7.3f}  {row['MI_score']:>9.4f}  "
              f"{row['F_score']:>9.2f}  {row['cohens_d']:>9.3f}  {row['p_value']:>9.2e}")

    return combined


def plot_feature_distributions(df, top_features, n_features=9):
    """Plot distributions of top discriminative features."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    print("\nFEATURE DISTRIBUTION DIAGNOSTICS:")
    print("=" * 80)

    for idx, feature in enumerate(top_features[:n_features]):
        ax = axes[idx]

        # Split by group
        control_data = df[df['group'] == 'control'][feature].dropna()
        patient_data = df[df['group'] == 'patient'][feature].dropna()

        # Diagnostics
        print(f"\n{feature}:")
        print(f"  Control: n={len(control_data)}, mean={control_data.mean():.4f}, std={control_data.std():.4f}")
        print(f"           range=[{control_data.min():.4f}, {control_data.max():.4f}]")
        print(f"  Patient: n={len(patient_data)}, mean={patient_data.mean():.4f}, std={patient_data.std():.4f}")
        print(f"           range=[{patient_data.min():.4f}, {patient_data.max():.4f}]")
        print(f"  NaN count: Control={df[df['group'] == 'control'][feature].isna().sum()}, "
              f"Patient={df[df['group'] == 'patient'][feature].isna().sum()}")

        # Check if data is valid
        if len(control_data) == 0 or len(patient_data) == 0:
            ax.text(0.5, 0.5, f'NO DATA\n{feature}',
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax.transAxes)
            ax.set_title(f'{feature} (NO DATA)', fontsize=10, fontweight='bold', color='red')
            continue

        # Check for zero variance
        if control_data.std() == 0 and patient_data.std() == 0:
            ax.text(0.5, 0.5, f'ZERO VARIANCE\n{feature}\nValue: {control_data.mean():.4f}',
                    ha='center', va='center', fontsize=10, color='orange',
                    transform=ax.transAxes)
            ax.set_title(f'{feature} (ZERO VAR)', fontsize=10, fontweight='bold', color='orange')
            continue

        # Determine appropriate bins
        all_data = pd.concat([control_data, patient_data])
        data_range = all_data.max() - all_data.min()

        if data_range > 0:
            # Use automatic binning with range specification
            bins = np.linspace(all_data.min(), all_data.max(), 21)

            # Plot histograms
            if len(control_data) > 0:
                ax.hist(control_data, alpha=0.6, bins=bins, label=f'Control (n={len(control_data)})',
                        color='blue', density=True)
                ax.axvline(control_data.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'Control μ={control_data.mean():.3f}')

            if len(patient_data) > 0:
                ax.hist(patient_data, alpha=0.6, bins=bins, label=f'PD (n={len(patient_data)})',
                        color='red', density=True)
                ax.axvline(patient_data.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'PD μ={patient_data.mean():.3f}')

            ax.set_title(feature, fontsize=9, fontweight='bold')
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, f'CONSTANT VALUE\n{feature}\n{all_data.mean():.4f}',
                    ha='center', va='center', fontsize=10, color='gray',
                    transform=ax.transAxes)
            ax.set_title(f'{feature} (CONSTANT)', fontsize=10, fontweight='bold', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved feature distributions to: {PROCESSED_DIR}/feature_distributions.png")
    print("=" * 80)
    return fig


def visualize_tsne_comprehensive(df, X_scaled, feature_names):
    """Create comprehensive t-SNE visualizations with multiple perspectives."""

    labels = df["group"]
    sites = df["site"]
    label_map = {'control': 0, 'patient': 1}
    y_numeric = labels.map(label_map)

    # Analyze feature importance
    feature_importance = analyze_feature_importance(X_scaled, labels, feature_names)

    # Get top features for focused analysis
    top_feature_names = feature_importance.head(20)['feature'].tolist()
    top_feature_indices = [feature_names.index(f) for f in top_feature_names if f in feature_names]
    X_top_features = X_scaled[:, top_feature_indices]

    print(f"\nUsing top {len(top_feature_indices)} features for dimensionality reduction")

    # === PCA for variance analysis ===
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Find number of components for 95% variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

    print(f"PCA Analysis:")
    print(f"   • Components for 95% variance: {n_components_95}/{X_scaled.shape[1]}")


    # Use PCA for dimensionality reduction
    pca = PCA(n_components=min(50, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Also try LDA (supervised dimensionality reduction)
    try:
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(X_scaled, y_numeric)
    except:
        print("  LDA failed (possibly due to singular matrix), skipping...")
        X_lda = None

    # === t-SNE configurations ===
    perplexity = min(30, len(X_pca) - 1)  # Use only perplexity 30

    fig = plt.figure(figsize=(24, 16))
    plot_idx = 1

    # t-SNE on PCA-reduced data with perplexity 30 only
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=2000,
        learning_rate='auto',
        init='pca',
        metric='euclidean'
    )
    X_tsne = tsne.fit_transform(X_pca)

    # Plot 1: By group
    ax = plt.subplot(3, 4, plot_idx)
    for group, color in zip(['control', 'patient'], ['blue', 'red']):
        mask = labels == group
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=color, label=group, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_title(f't-SNE (perplexity={perplexity}): PD vs Control', fontweight='bold')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # === t-SNE on different numbers of top features ===
    feature_counts = [10, 9, 8, 7, 6, 5]

    for n_features in feature_counts:
        # Ensure we're using the top N by combined score
        top_n_features = top_feature_names[:n_features]  # Already sorted by combined_score
        top_indices = [feature_names.index(f) for f in top_n_features if f in feature_names]
        X_top = X_scaled[:, top_indices]

        print(f"   • Top {n_features}: {', '.join([f.split('_')[-1] for f in top_n_features[:3]])}...")

        tsne_top = TSNE(
            n_components=2,
            perplexity=min(30, len(X_top) - 1),
            random_state=42,
            n_iter=3000,
            learning_rate='auto',
            init='random'
        )
        X_tsne_top = tsne_top.fit_transform(X_top)

        ax = plt.subplot(3, 4, plot_idx)
        for group, color in zip(['control', 'patient'], ['blue', 'red']):
            mask = labels == group
            ax.scatter(X_tsne_top[mask, 0], X_tsne_top[mask, 1],
                       c=color, label=group, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax.set_title(f't-SNE: Top {n_features} Features', fontweight='bold', fontsize=11)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # === PCA Visualization (2 components) ===
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    ax = plt.subplot(3, 4, plot_idx)
    for group, color in zip(['control', 'patient'], ['blue', 'red']):
        mask = labels == group
        ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                   c=color, label=group, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_title(
        f'PCA: PC1 vs PC2 ({pca_2d.explained_variance_ratio_[0]:.1%} + {pca_2d.explained_variance_ratio_[1]:.1%} var)',
        fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # === PCA Scree Plot ===
    ax = plt.subplot(3, 4, plot_idx)
    ax.plot(range(1, min(21, len(pca_full.explained_variance_ratio_) + 1)),
            pca_full.explained_variance_ratio_[:20], 'bo-', linewidth=2, markersize=8)
    ax.set_title('PCA Scree Plot', fontweight='bold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained Ratio')
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # === Cumulative Variance ===
    ax = plt.subplot(3, 4, plot_idx)
    ax.plot(range(1, min(51, len(cumsum_var) + 1)),
            cumsum_var[:50], 'ro-', linewidth=2, markersize=6)
    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
    ax.axvline(x=n_components_95, color='green', linestyle='--', linewidth=2)
    ax.set_title('Cumulative Variance', fontweight='bold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # === Feature Importance Bar Plot ===
    ax = plt.subplot(3, 4, plot_idx)
    top_10 = feature_importance.head(10)

    y_pos = np.arange(len(top_10))
    ax.barh(y_pos, top_10['combined_score'], color='blue', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:30] for f in top_10['feature']], fontsize=8)
    ax.set_xlabel('Combined Score')
    ax.set_title('Top 10 Features', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'comprehensive_tsne_analysis.png'),
                dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive visualization to: {PROCESSED_DIR}/comprehensive_tsne_analysis.png")

    # === CREATE FOCUSED t-SNE COMPARISON FIGURE ===
    fig_focused = plt.figure(figsize=(25, 15))

    # Ensure we're using combined ranking
    top_feature_names_combined = feature_importance.sort_values('combined_score', ascending=False)['feature'].tolist()

    plot_idx = 1

    # Define the feature counts to plot
    feature_counts = [20, 9, 8, 7, 6, 5, 4, 16]

    # Plot top features from 10 down to 3
    for n_features in feature_counts:
        ax = plt.subplot(3, 3, plot_idx)

        # Get top N features by combined ranking
        top_n_features = top_feature_names_combined[:n_features]
        top_indices = [feature_names.index(f) for f in top_n_features if f in feature_names]

        if top_indices:  # Only plot if features are found
            X_top = X_scaled[:, top_indices]

            tsne_top = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=3000,
                            learning_rate='auto', init='random')
            X_tsne_top = tsne_top.fit_transform(X_top)

            for group, color in zip(['control', 'patient'], ['blue', 'red']):
                mask = labels == group
                ax.scatter(X_tsne_top[mask, 0], X_tsne_top[mask, 1],
                           c=color, label=group, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

            ax.set_title(f'Top {n_features} Features', fontweight='bold', fontsize=11)
            ax.set_xlabel('t-SNE Component 1', fontsize=10)
            ax.set_ylabel('t-SNE Component 2', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No features found\nfor top {n_features}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Top {n_features} Features\n(Not Found)', fontweight='bold', fontsize=11)

        plot_idx += 1

    # Custom features plot
    custom_features = [
        'L2__maximum',
        'L2__absolute_maximum',
        'L_impulse_cv',
        'L_swing_ratio'

    ]

    ax_custom = plt.subplot(3, 3, plot_idx)
    custom_indices = [feature_names.index(f) for f in custom_features if f in feature_names]

    if custom_indices:  # Only plot if features are found
        X_custom = X_scaled[:, custom_indices]

        tsne_custom = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=3000,
                           learning_rate='auto', init='random')
        X_tsne_custom = tsne_custom.fit_transform(X_custom)

        for group, color in zip(['control', 'patient'], ['blue', 'red']):
            mask = labels == group
            ax_custom.scatter(X_tsne_custom[mask, 0], X_tsne_custom[mask, 1],
                              c=color, label=group, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

        ax_custom.set_title(f'Custom {len(custom_features)} Features', fontweight='bold', fontsize=11)
        ax_custom.set_xlabel('t-SNE Component 1', fontsize=10)
        ax_custom.set_ylabel('t-SNE Component 2', fontsize=10)
        ax_custom.legend(fontsize=9)
        ax_custom.grid(True, alpha=0.3)
    else:
        ax_custom.text(0.5, 0.5, 'No custom features found',
                       ha='center', va='center', transform=ax_custom.transAxes, fontsize=12)
        ax_custom.set_title('Custom Features\n(Not Found)', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'tsne_focused_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"Saved focused t-SNE comparison to: {PROCESSED_DIR}/tsne_focused_comparison.png")

    # Print which features were used
    print(f"\nFeatures used in focused t-SNE comparison:")

    # Print top features for each configuration
    for n_features in feature_counts:
        features_list = top_feature_names_combined[:n_features]
        print(f"\nTop {n_features} features:")
        for i, fname in enumerate(features_list, 1):
            # Safe lookup - check if feature exists
            if fname in feature_importance['feature'].values:
                score = feature_importance[feature_importance['feature'] == fname]['combined_score'].values[0]
                print(f"   {i:2d}. {fname:<35} - Combined Score: {score:.3f}")
            else:
                print(f"   {i:2d}. {fname:<35} - NOT FOUND")

    # Print custom features
    print(f"\nCustom {len(custom_features)} features:")
    for i, fname in enumerate(custom_features, 1):
        # Safe lookup - check if feature exists
        if fname in feature_importance['feature'].values:
            score = feature_importance[feature_importance['feature'] == fname]['combined_score'].values[0]
            print(f"   {i:2d}. {fname:<35} - Combined Score: {score:.3f}")
        else:
            print(f"   {i:2d}. {fname:<35} - NOT FOUND")

    return fig, fig_focused, feature_importance


def main():
    # Load data
    df = pd.read_csv(FEATURES_CSV)

    print("=" * 80)
    print("🧬 GAIT FEATURE ANALYSIS FOR PARKINSON'S DISEASE DETECTION")
    print("=" * 80)

    # FIXED: Remove ALL non-numeric columns including filename, subject_id, etc.
    # First, identify numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Keep only numeric columns for feature processing
    features = df[numeric_columns].copy()

    print(f"Using {len(numeric_columns)} numeric features for analysis")
    print(f"Removed {len(df.columns) - len(numeric_columns)} non-numeric columns")

    # Check for missing values
    missing_pct = (features.isnull().sum() / len(features) * 100).sort_values(ascending=False)
    if missing_pct.max() > 0:
        print(f"\nMissing values detected:")
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"   - {col}: {pct:.1f}% missing")

        # Handle missing values more carefully
        print(f"\nHandling missing values...")

        # Remove features with too many missing values (>50%)
        high_missing = missing_pct[missing_pct > 50].index
        if len(high_missing) > 0:
            print(f"   Removing {len(high_missing)} features with >50% missing values:")
            for col in high_missing:
                print(f"     - {col}")
            features = features.drop(columns=high_missing)

        # Fill remaining missing values with median
        features = features.fillna(features.median())

    # Remove features with zero variance (now safe since we only have numeric columns)
    zero_var_cols = features.columns[features.var() == 0]
    if len(zero_var_cols) > 0:
        print(f"\nRemoving {len(zero_var_cols)} zero-variance features")
        features = features.drop(columns=zero_var_cols)

    print(f"\n📊 Dataset Statistics:")
    print(f"   • Samples: {len(df)}")
    print(f"   • Features: {features.shape[1]}")
    print(f"   • Class distribution:")
    for group, count in df['group'].value_counts().items():
        print(f"     - {group}: {count} ({count / len(df) * 100:.1f}%)")
    print(f"   • Sites: {df['site'].nunique()} ({', '.join(df['site'].unique())})")

    # Use RobustScaler (better for outliers in medical data)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features)
    feature_names = features.columns.tolist()

    # Check for outliers
    outlier_counts = np.sum(np.abs(X_scaled) > 3, axis=0)
    if np.max(outlier_counts) > 0:
        print(f"\n⚠Extreme outliers detected (>3 std):")
        outlier_features = [(feature_names[i], count) for i, count in enumerate(outlier_counts) if count > 0]
        outlier_features.sort(key=lambda x: x[1], reverse=True)
        for feature, count in outlier_features[:10]:  # Show top 10 only
            print(f"     - {feature}: {count} outliers")

    # Create comprehensive visualizations
    fig_tsne, fig_focused, feature_importance = visualize_tsne_comprehensive(df, X_scaled, feature_names)

    # Plot feature distributions for top features
    top_features = feature_importance.head(9)['feature'].tolist()
    fig_dist = plot_feature_distributions(df, top_features)

    # Save feature importance
    feature_importance.to_csv(
        os.path.join(PROCESSED_DIR, 'feature_importance_ranking.csv'),
        index=False
    )
    print(f"\nSaved feature importance ranking to: {PROCESSED_DIR}/feature_importance_ranking.csv")

    plt.show()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()