"""
Enhanced Breast Cancer Diagnosis System
Phase 3: Exploratory Data Analysis (EDA)
Comprehensive Statistical and Visual Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df = pd.read_csv('breast_cancer_enhanced_dataset.csv')

print("=" * 100)
print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 100)

# ============================================================================
# 1. DATASET SHAPE AND STRUCTURE
# ============================================================================
print("\n1. DATASET SHAPE AND STRUCTURE")
print("─" * 100)
print(f"Shape: {df.shape[0]} records × {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nData Types:")
print(df.dtypes)

# ============================================================================
# 2. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n2. CLASS DISTRIBUTION (TARGET VARIABLE)")
print("─" * 100)

class_dist = df['diagnosis'].value_counts()
class_pct = (df['diagnosis'].value_counts(normalize=True) * 100).round(2)

print(f"Benign (B):    {class_dist['B']:5d} records ({class_pct['B']:6.2f}%)")
print(f"Malignant (M): {class_dist['M']:5d} records ({class_pct['M']:6.2f}%)")
print(f"\nClass Imbalance Ratio: {class_dist['B']/class_dist['M']:.2f}:1")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
class_dist.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Diagnosis')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Benign', 'Malignant'], rotation=0)

# Pie chart
colors = ['#2ecc71', '#e74c3c']
axes[1].pie(class_dist, labels=['Benign', 'Malignant'], autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('03_class_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_class_distribution.png")
plt.close()

# ============================================================================
# 3. STATISTICAL SUMMARY
# ============================================================================
print("\n3. STATISTICAL SUMMARY")
print("─" * 100)

stats_summary = df.describe().T
print(stats_summary.round(4))

print("\nKey Observations:")
print(f"  • Minimum variance features: {stats_summary['std'].nsmallest(3).index.tolist()}")
print(f"  • Maximum variance features: {stats_summary['std'].nlargest(3).index.tolist()}")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("\n4. CORRELATION MATRIX ANALYSIS")
print("─" * 100)

# Encode target for correlation
df_encoded = df.copy()
df_encoded['diagnosis'] = (df_encoded['diagnosis'] == 'M').astype(int)

# Calculate correlation with target
target_corr = df_encoded.corr()['diagnosis'].drop('diagnosis').sort_values(ascending=False)

print("\nTop 10 Features Correlated with Malignancy:")
print(target_corr.head(10).round(4))

print("\nBottom 5 Features Correlated with Malignancy:")
print(target_corr.tail(5).round(4))

# Correlation heatmap
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
corr_matrix = df_encoded[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('03_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_correlation_matrix.png")
plt.close()

# Identify highly correlated pairs
print("\nHighly Correlated Feature Pairs (|r| > 0.85):")
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if corr_pairs:
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print(corr_pairs_df.to_string(index=False))
else:
    print("  No pairs with |r| > 0.85")

# ============================================================================
# 5. FEATURE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n5. FEATURE DISTRIBUTION ANALYSIS")
print("─" * 100)

numeric_features = [col for col in df.columns if col not in ['id', 'diagnosis']]

# Distribution plots
fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for idx, col in enumerate(numeric_features[:15]):
    ax = axes[idx]
    
    # Histogram with KDE
    df[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
    ax2 = ax.twinx()
    df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2, label='KDE')
    
    ax.set_title(f'Distribution: {col}', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

# Remove extra subplots
for idx in range(len(numeric_features[:15]), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('03_feature_distributions_1.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_feature_distributions_1.png")
plt.close()

# Distribution plots for remaining features
if len(numeric_features) > 15:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_features[15:]):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_title(f'Distribution: {col}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_feature_distributions_2.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_feature_distributions_2.png")
    plt.close()

# ============================================================================
# 6. BOXPLOTS BY DIAGNOSIS
# ============================================================================
print("\n6. BOXPLOT ANALYSIS BY DIAGNOSIS")
print("─" * 100)

fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for idx, col in enumerate(numeric_features[:15]):
    ax = axes[idx]
    df.boxplot(column=col, by='diagnosis', ax=ax)
    ax.set_title(f'{col} by Diagnosis', fontweight='bold')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Value')
    plt.sca(ax)
    plt.xticks([1, 2], ['Benign', 'Malignant'])

plt.suptitle('')  # Remove default title
for idx in range(len(numeric_features[:15]), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('03_boxplots_1.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_boxplots_1.png")
plt.close()

# ============================================================================
# 7. PAIRPLOT
# ============================================================================
print("\n7. PAIRPLOT ANALYSIS (First 4 features)")
print("─" * 100)

# Create pairplot with first 4 numeric features
pairplot_features = numeric_features[:4] + ['diagnosis']
pairplot_data = df[pairplot_features].copy()

plt.figure(figsize=(12, 10))
pairplot = sns.pairplot(pairplot_data, hue='diagnosis', diag_kind='kde', 
                        plot_kws={'alpha': 0.6}, palette=['#2ecc71', '#e74c3c'])
pairplot.fig.suptitle('Pairplot: First 4 Features vs Diagnosis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('03_pairplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_pairplot.png")
plt.close()

# ============================================================================
# 8. TARGET ANALYSIS - FEATURE MEANS BY DIAGNOSIS
# ============================================================================
print("\n8. MEAN FEATURE VALUES BY DIAGNOSIS")
print("─" * 100)

feature_by_diagnosis = df.groupby('diagnosis')[numeric_features].mean()
print(feature_by_diagnosis.round(4))

print("\nDifference (Malignant - Benign):")
difference = feature_by_diagnosis.loc['M'] - feature_by_diagnosis.loc['B']
print(difference.sort_values(ascending=False).round(4))

# ============================================================================
# 9. CLINICAL PATTERN DISCOVERY
# ============================================================================
print("\n9. CLINICAL PATTERN DISCOVERY")
print("─" * 100)

print("\nKey Findings:")

# Size characteristics
print("\n▪ TUMOR SIZE CHARACTERISTICS:")
for diag in ['B', 'M']:
    subset = df[df['diagnosis'] == diag]
    label = 'Benign' if diag == 'B' else 'Malignant'
    print(f"\n  {label} Tumors:")
    print(f"    - Avg Radius: {subset['radius_mean'].mean():.2f} mm")
    print(f"    - Avg Area:   {subset['area_mean'].mean():.0f} mm²")
    print(f"    - Avg Perimeter: {subset['perimeter_mean'].mean():.2f} mm")

# Morphology
print("\n▪ MORPHOLOGICAL CHARACTERISTICS:")
for diag in ['B', 'M']:
    subset = df[df['diagnosis'] == diag]
    label = 'Benign' if diag == 'B' else 'Malignant'
    print(f"\n  {label} Tumors:")
    print(f"    - Concavity:  {subset['concavity_mean'].mean():.4f}")
    print(f"    - Concave Points: {subset['concave_points_mean'].mean():.4f}")
    print(f"    - Shape Irregularity: {subset['shape_irregularity'].mean():.4f}")

# Texture
print("\n▪ TEXTURE CHARACTERISTICS:")
for diag in ['B', 'M']:
    subset = df[df['diagnosis'] == diag]
    label = 'Benign' if diag == 'B' else 'Malignant'
    print(f"\n  {label} Tumors:")
    print(f"    - Avg Texture: {subset['texture_mean'].mean():.2f}")
    print(f"    - Smoothness: {subset['smoothness_mean'].mean():.4f}")
    print(f"    - Compactness: {subset['compactness_mean'].mean():.4f}")

# ============================================================================
# 10. STATISTICAL TESTS
# ============================================================================
print("\n10. STATISTICAL SIGNIFICANCE TESTS")
print("─" * 100)

print("\nT-Tests (Benign vs Malignant) - Features with p < 0.001:")
print("\nFeature                        | t-statistic | p-value      | Significant")
print("─" * 75)

significant_features = []
for col in numeric_features:
    benign = df[df['diagnosis'] == 'B'][col]
    malignant = df[df['diagnosis'] == 'M'][col]
    
    t_stat, p_value = stats.ttest_ind(benign, malignant)
    
    if p_value < 0.001:
        significant_features.append((col, t_stat, p_value))
        sig_marker = "✓ YES"
    else:
        sig_marker = "  NO"
    
    print(f"{col:30s} | {t_stat:11.4f} | {p_value:.2e} | {sig_marker}")

print(f"\nTotal Significant Features (p < 0.001): {len(significant_features)}")

print("\n" + "=" * 100)
print("END PHASE 3 - EDA COMPLETE")
print("=" * 100)
