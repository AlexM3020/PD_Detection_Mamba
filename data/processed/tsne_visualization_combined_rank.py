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
FEATURES_CSV = os.path.join(PROCESSED_DIR, "improved_gait_features.csv")


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

    # Combine rankings
    combined = mi_ranking.merge(f_ranking, on='feature').merge(effect_ranking, on='feature')
    combined['combined_rank'] = (
            combined['MI_score'].rank(ascending=False) +
            combined['F_score'].rank(ascending=False) +
            combined['cohens_d'].rank(ascending=False)
    )
    combined = combined.sort_values('combined_rank')

    print("\n" + "=" * 80)
    print("🎯 TOP DISCRIMINATIVE FEATURES FOR PD DETECTION")
    print("=" * 80)
    cohens_d_label = "Cohen's d"
    print(f"\n{'Feature':<35} {'MI Score':<12} {'F-Score':<12} {cohens_d_label:<12} {'p-value'}")
    print("-" * 80)

    for idx, row in combined.head(top_n).iterrows():
        print(f"{row['feature']:<35} {row['MI_score']:>10.4f}  {row['F_score']:>10.2f}  "
              f"{row['cohens_d']:>10.3f}  {row['p_value']:>10.2e}")

    return combined


def plot_feature_distributions(df, top_features, n_features=9):
    """Plot distributions of top discriminative features."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    print("\n🔍 FEATURE DISTRIBUTION DIAGNOSTICS:")
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
    print(f"\n📊 Saved feature distributions to: {PROCESSED_DIR}/feature_distributions.png")
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

    print(f"\n🔬 Using top {len(top_feature_indices)} features for dimensionality reduction")

    # === PCA for variance analysis ===
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Find number of components for 95% variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

    print(f"📉 PCA Analysis:")
    print(f"   • Components for 95% variance: {n_components_95}/{X_scaled.shape[1]}")
    print(f"   • Top 5 components explain: {cumsum_var[4]:.1%} variance")

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=min(50, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Also try LDA (supervised dimensionality reduction)
    try:
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(X_scaled, y_numeric)
    except:
        print("⚠️  LDA failed (possibly due to singular matrix), skipping...")
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
    # Focus on range from top 10 to top 5
    feature_counts = [10, 9, 8, 7, 6, 5]  # All from 10 to 5

    for n_features in feature_counts:
        top_indices = [feature_names.index(f) for f in top_feature_names[:n_features] if f in feature_names]
        X_top = X_scaled[:, top_indices]

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
    ax.set_title('PCA Scree Plot (Variance Explained)', fontweight='bold')
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
    ax.set_title('Cumulative Variance Explained', fontweight='bold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # === Feature Importance Bar Plot ===
    ax = plt.subplot(3, 4, plot_idx)
    top_10 = feature_importance.head(10)
    # CHANGED: Use combined rank score and all blue color
    colors_bar = ['steelblue' for _ in top_10['feature']]

    y_pos = np.arange(len(top_10))
    # CHANGED: Use combined rank score instead of MI score
    ax.barh(y_pos, top_10['combined_rank'], color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:30] for f in top_10['feature']], fontsize=8)
    ax.set_xlabel('Combined Rank Score (sum of MI + F-score + Cohen\'s d ranks)')  # CHANGED: Explicit label
    ax.set_title('Top 10 Discriminative Features\n(Combined Rank)', fontweight='bold')  # CHANGED: Updated title
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'comprehensive_tsne_analysis.png'),
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comprehensive visualization to: {PROCESSED_DIR}/comprehensive_tsne_analysis.png")

    # === CREATE FOCUSED t-SNE COMPARISON FIGURE ===
    fig_focused = plt.figure(figsize=(25, 12))

    # All features (as demonstration)
    ax1 = plt.subplot(2, 5, 1)
    tsne_all = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=3000,
                    learning_rate='auto', init='pca')
    X_tsne_all = tsne_all.fit_transform(X_pca)
    for group, color in zip(['control', 'patient'], ['blue', 'red']):
        mask = labels == group
        ax1.scatter(X_tsne_all[mask, 0], X_tsne_all[mask, 1],
                    c=color, label=group, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    ax1.set_title(f'All {X_scaled.shape[1]} Features\n(PCA pre-processed)',
                  fontweight='bold', fontsize=11)
    ax1.set_xlabel('t-SNE Component 1', fontsize=10)
    ax1.set_ylabel('t-SNE Component 2', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Top 10 to Top 5 features
    feature_counts_focused = [10, 9, 8, 7, 6, 5]

    for idx, n_features in enumerate(feature_counts_focused, 2):
        top_indices = [feature_names.index(f) for f in top_feature_names[:n_features] if f in feature_names]
        X_top = X_scaled[:, top_indices]

        tsne_top = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=3000,
                        learning_rate='auto', init='random')
        X_tsne_top = tsne_top.fit_transform(X_top)

        ax = plt.subplot(2, 5, idx)
        for group, color in zip(['control', 'patient'], ['blue', 'red']):
            mask = labels == group
            ax.scatter(X_tsne_top[mask, 0], X_tsne_top[mask, 1],
                       c=color, label=group, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

        # CHANGED: Remove the "BEST" green text
        ax.set_title(f'Top {n_features} Features', fontweight='bold', fontsize=11)

        ax.set_xlabel('t-SNE Component 1', fontsize=10)
        ax.set_ylabel('t-SNE Component 2', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Feature importance (last plot)
    ax8 = plt.subplot(2, 5, 8)
    top_10_importance = feature_importance.head(10)
    # CHANGED: Use combined rank score and all blue color
    colors_bar = ['steelblue' for _ in top_10_importance['feature']]

    y_pos = np.arange(len(top_10_importance))
    # CHANGED: Use combined rank score instead of MI score
    ax8.barh(y_pos, top_10_importance['combined_rank'], color=colors_bar, alpha=0.8, edgecolor='black')
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([f[:25] for f in top_10_importance['feature']], fontsize=8)
    ax8.set_xlabel('Combined Rank Score (sum of MI + F-score + Cohen\'s d ranks)',
                   fontsize=10)  # CHANGED: Explicit label
    ax8.set_title('Top 10 Features by Combined Rank',  # CHANGED: Updated title
                  fontweight='bold', fontsize=11)
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'tsne_focused_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"📊 Saved focused t-SNE comparison to: {PROCESSED_DIR}/tsne_focused_comparison.png")

    # Print which features were used
    print(f"\n🎯 Features used in each t-SNE:")
    for n_features in [10, 9, 8, 7, 6, 5]:
        print(f"\nTop {n_features} features:")
        for i, fname in enumerate(top_feature_names[:n_features], 1):
            print(f"   {i}. {fname}")

    return fig, fig_focused, feature_importance


def main():
    # Load data
    df = pd.read_csv(FEATURES_CSV)

    print("=" * 80)
    print("🧬 GAIT FEATURE ANALYSIS FOR PARKINSON'S DISEASE DETECTION")
    print("=" * 80)

    # FIXED: Remove ALL metadata columns including subject_id
    metadata_columns = ["filename", "group", "site", "subject_id"]
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    features = df[feature_columns].copy()

    # Check for missing values
    missing_pct = (features.isnull().sum() / len(features) * 100).sort_values(ascending=False)
    if missing_pct.max() > 0:
        print(f"\n⚠️  Missing values detected:")
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"   - {col}: {pct:.1f}% missing")

        # FIXED: Handle missing values more carefully
        print(f"\n🔄 Handling missing values...")

        # Remove features with too many missing values (>50%)
        high_missing = missing_pct[missing_pct > 50].index
        if len(high_missing) > 0:
            print(f"   Removing {len(high_missing)} features with >50% missing values:")
            for col in high_missing:
                print(f"     - {col}")
            features = features.drop(columns=high_missing)

        # Fill remaining missing values with median (only numeric columns)
        numeric_features = features.select_dtypes(include=[np.number]).columns
        features[numeric_features] = features[numeric_features].fillna(features[numeric_features].median())

    # Remove features with zero variance
    zero_var_cols = features.columns[features.var() == 0]
    if len(zero_var_cols) > 0:
        print(f"\n⚠️  Removing {len(zero_var_cols)} zero-variance features")
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
        print(f"\n⚠️  Extreme outliers detected (>3 std):")
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
    print(f"\n💾 Saved feature importance ranking to: {PROCESSED_DIR}/feature_importance_ranking.csv")

    plt.show()

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()