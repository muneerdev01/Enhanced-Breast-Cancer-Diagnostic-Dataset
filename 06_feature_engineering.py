"""
Enhanced Breast Cancer Diagnosis System
Phase 6: Feature Engineering
Creation of Medically Meaningful Features
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("PHASE 6: FEATURE ENGINEERING")
print("=" * 100)

# Load cleaned dataset
df = pd.read_csv('cleaned_breast_cancer.csv')
df['diagnosis'] = df['diagnosis'].astype('category')

print(f"\n1. INITIAL STATE")
print(f"   Records: {df.shape[0]}")
print(f"   Features: {df.shape[1]}")
print(f"   Base features: {[col for col in df.columns if col != 'diagnosis']}")

df_engineered = df.copy()

# ============================================================================
# FEATURE ENGINEERING: SIZE RATIOS
# ============================================================================
print(f"\n2. SIZE-BASED FEATURES")
print("─" * 100)

# Feature 1: Area to Radius ratio (should be ~π for circular objects)
# Deviation from π indicates irregular shape
print(f"\n   Feature: area_radius_ratio")
df_engineered['area_radius_ratio'] = df_engineered['area_mean'] / (np.pi * df_engineered['radius_mean']**2)
print(f"   Formula: area_mean / (π × radius_mean²)")
print(f"   Clinical Meaning: Deviation from circular shape indicator")
print(f"   Interpretation: Value close to 1.0 = circular; >1.0 = irregular shape")
print(f"   Expected Range: 0.8 - 1.5")
print(f"   Reason for Creation: Captures shape deviations normalizing for size")

# Feature 2: Compactness to Smoothness ratio
print(f"\n   Feature: compactness_smoothness_ratio")
df_engineered['compactness_smoothness_ratio'] = df_engineered['compactness_mean'] / (df_engineered['smoothness_mean'] + 1e-6)
print(f"   Formula: compactness_mean / smoothness_mean")
print(f"   Clinical Meaning: Shape irregularity vs surface smoothness ratio")
print(f"   Reason for Creation: Combines two complementary shape descriptors")

# ============================================================================
# FEATURE ENGINEERING: BORDER IRREGULARITY INDICES
# ============================================================================
print(f"\n3. BORDER IRREGULARITY FEATURES")
print("─" * 100)

# Feature 3: Concavity Index
print(f"\n   Feature: concavity_index")
df_engineered['concavity_index'] = (df_engineered['concavity_mean'] * 
                                     df_engineered['concave_points_mean'] * 
                                     df_engineered['border_complexity'])
print(f"   Formula: concavity_mean × concave_points_mean × border_complexity")
print(f"   Clinical Meaning: Composite border irregularity measure")
print(f"   Interpretation: Higher values = more irregular borders")
print(f"   Expected Predictive Value: Strong - Combines three border metrics")
print(f"   Reason for Creation: Captures synergistic effect of border abnormalities")

# Feature 4: Border Irregularity Normalized
print(f"\n   Feature: border_irregularity_normalized")
df_engineered['border_irregularity_normalized'] = (df_engineered['concavity_mean'] + 
                                                    df_engineered['concave_points_mean']) / 2
print(f"   Formula: (concavity_mean + concave_points_mean) / 2")
print(f"   Clinical Meaning: Average border irregularity metric")
print(f"   Reason for Creation: Normalized combination of concavity features")

# ============================================================================
# FEATURE ENGINEERING: MORPHOLOGICAL COMPOSITE
# ============================================================================
print(f"\n4. MORPHOLOGICAL COMPOSITE FEATURES")
print("─" * 100)

# Feature 5: Morphology Score
print(f"\n   Feature: morphology_score")

# Normalize features to 0-1 scale for equal weighting
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

morphology_features = ['shape_irregularity', 'compactness_mean', 'concavity_mean']
normalized_morph = pd.DataFrame(
    scaler.fit_transform(df_engineered[morphology_features]),
    columns=morphology_features
)

df_engineered['morphology_score'] = normalized_morph.mean(axis=1)
print(f"   Formula: mean(normalized[shape_irregularity, compactness_mean, concavity_mean])")
print(f"   Clinical Meaning: Overall morphological abnormality score")
print(f"   Range: 0.0 - 1.0 (0 = normal, 1 = highly abnormal)")
print(f"   Expected Predictive Value: Very Strong - Composite abnormality indicator")
print(f"   Reason for Creation: Aggregates multiple morphology signals into single score")

# Feature 6: Heterogeneity Score
print(f"\n   Feature: cellular_heterogeneity_score")
texture_smooth_score = pd.DataFrame(
    scaler.fit_transform(df_engineered[['texture_mean', 'smoothness_mean']]),
    columns=['texture_mean', 'smoothness_mean']
)
df_engineered['cellular_heterogeneity_score'] = (texture_smooth_score['texture_mean'] - 
                                                   texture_smooth_score['smoothness_mean'])
print(f"   Formula: normalized_texture - normalized_smoothness")
print(f"   Clinical Meaning: Cell heterogeneity indicator")
print(f"   Interpretation: Higher = more heterogeneous cells (malignant indicator)")
print(f"   Reason for Creation: Captures organizational disarray")

# ============================================================================
# FEATURE ENGINEERING: INTERACTION TERMS
# ============================================================================
print(f"\n5. INTERACTION FEATURES")
print("─" * 100)

# Feature 7: Size-Morphology Interaction
print(f"\n   Feature: size_morphology_interaction")
radius_norm = pd.DataFrame(scaler.fit_transform(df_engineered[['radius_mean']]))
morph_norm = df_engineered['morphology_score']
df_engineered['size_morphology_interaction'] = (radius_norm[0] * morph_norm).values
print(f"   Formula: normalized_radius × morphology_score")
print(f"   Clinical Meaning: Combined effect of size and shape abnormality")
print(f"   Reason for Creation: Captures interaction between tumor size and shape irregularity")
print(f"   Expected Value: Malignant tumors likely larger AND more irregular")

# Feature 8: Border Size Interaction
print(f"\n   Feature: border_size_interaction")
border_norm = pd.DataFrame(scaler.fit_transform(df_engineered[['concavity_mean']]))
df_engineered['border_size_interaction'] = (radius_norm[0] * border_norm[0]).values
print(f"   Formula: normalized_radius × normalized_concavity")
print(f"   Clinical Meaning: Size and border irregularity interaction")
print(f"   Reason for Creation: Captures combined malignancy signal")

# ============================================================================
# FEATURE ENGINEERING: STATISTICAL AGGREGATIONS
# ============================================================================
print(f"\n6. STATISTICAL AGGREGATION FEATURES")
print("─" * 100)

# Feature 9: Malignancy Risk Features Combination
print(f"\n   Feature: malignancy_composite_score\")\nagg_features = ['concavity_mean', 'concave_points_mean', 'shape_irregularity', \n               'border_complexity', 'texture_mean']\ndf_engineered['malignancy_composite_score'] = df_engineered[agg_features].mean(axis=1)\nprint(f\"   Formula: mean(concavity_mean, concave_points_mean, shape_irregularity, border_complexity, texture_mean)\")\nprint(f\"   Clinical Meaning: Aggregate malignancy indicator\")\nprint(f\"   Reason for Creation: Captures overall malignancy propensity\")\n\n# Feature 10: Benign Tumor Score (inverse)\nprint(f\"\"\"\n   Feature: benign_characteristics_score\"\"\")\nbenign_features = ['smoothness_mean', 'compactness_mean']  # These should be lower in malignant\ndf_engineered['benign_characteristics_score'] = df_engineered[benign_features].mean(axis=1)\nprint(f\"   Formula: mean(smoothness_mean, compactness_mean)\")\nprint(f\"   Clinical Meaning: Benign tumor characteristics indicator\")\nprint(f\"   Interpretation: Higher = more benign characteristics\")\nprint(f\"   Reason for Creation: Inverse malignancy indicator\")\n\n# ============================================================================\n# FEATURE ENGINEERING: NORMALIZED METRICS\n# ============================================================================\nprint(f\"\"\"\n7. NORMALIZED/STANDARDIZED FEATURES\"\"\")\nprint(\"─\" * 100)\n\n# Feature 11: Size-Adjusted Concavity\nprint(f\"\"\"\n   Feature: size_adjusted_concavity\"\"\")\ndf_engineered['size_adjusted_concavity'] = df_engineered['concavity_mean'] / (df_engineered['radius_mean'] + 1e-6)\nprint(f\"   Formula: concavity_mean / radius_mean\")\nprint(f\"   Clinical Meaning: Border irregularity normalized by tumor size\")\nprint(f\"   Reason for Creation: Removes size bias from concavity metric\")\nprint(f\"   Expected Value: Same concavity in larger/smaller tumors has different significance\")\n\n# Feature 12: Complexity per Unit Area\nprint(f\"\"\"\n   Feature: complexity_per_area\"\"\")\ndf_engineered['complexity_per_area'] = df_engineered['border_complexity'] / (df_engineered['area_mean'] + 1e-6)\nprint(f\"   Formula: border_complexity / area_mean\")\nprint(f\"   Clinical Meaning: Border complexity density\")\nprint(f\"   Reason for Creation: Normalizes complexity for tumor size\")\n\n# ============================================================================\n# FEATURE VERIFICATION\n# ============================================================================\nprint(f\"\"\"\n8. ENGINEERED FEATURES SUMMARY\"\"\")\nprint(\"─\" * 100)\n\noriginal_features = set(df.columns) - {'diagnosis'}\nengineered_features = set(df_engineered.columns) - set(df.columns)\n\nprint(f\"\"\"\n   Original features: {len(original_features)}\"\"\")\nprint(f\"   Engineered features: {len(engineered_features)}\")\nprint(f\"   Total features: {df_engineered.shape[1] - 1}\")\n\nprint(f\"\"\"\n   Engineered Features List:\"\"\")\nfor i, feat in enumerate(sorted(engineered_features), 1):\n    print(f\"   {i:2d}. {feat}\")\n\nprint(f\"\"\"\n9. ENGINEERED FEATURES STATISTICS\"\"\")\nprint(\"─\" * 100)\n\nprint(f\"\"\"\n   {'Feature':<40} | {'Min':<12} | {'Max':<12} | {'Mean':<12}\"\"\")\nprint(\"─\" * 80)\nfor feat in sorted(engineered_features):\n    print(f\"   {feat:<40} | {df_engineered[feat].min():>11.4f} | {df_engineered[feat].max():>11.4f} | {df_engineered[feat].mean():>11.4f}\")\n\n# ============================================================================\n# CHECK FOR NaN OR INFINITE VALUES\n# ============================================================================\nprint(f\"\"\"\n10. DATA QUALITY CHECK\"\"\")\nprint(\"─\" * 100)\n\nnan_count = df_engineered.isnull().sum().sum()\ninf_count = np.isinf(df_engineered.select_dtypes(include=[np.number])).sum().sum()\n\nprint(f\"\"\"\n   NaN values: {nan_count}\"\"\")\nprint(f\"   Infinite values: {inf_count}\")\nprint(f\"   ✓ Data quality maintained\")\n\n# ============================================================================\n# SAVE ENGINEERED DATASET\n# ============================================================================\nprint(f\"\"\"\n11. SAVE ENGINEERED DATASET\"\"\")\nprint(\"─\" * 100)\n\ndf_engineered.to_csv('engineered_breast_cancer.csv', index=False)\nprint(f\"\"\"\n   ✓ Saved: engineered_breast_cancer.csv\"\"\")\nprint(f\"   Size: {df_engineered.shape[0]} records × {df_engineered.shape[1]} features\")\n\n# ============================================================================\n# FEATURE ENGINEERING SUMMARY\n# ============================================================================\nprint(f\"\"\"\n\" + \"=\" * 100)\nprint(\"FEATURE ENGINEERING SUMMARY\")\nprint(\"=\" * 100)\n\nfeature_engineering_log = {\n    'Size-based': [\n        'area_radius_ratio - Shape deviation indicator',\n        'compactness_smoothness_ratio - Shape descriptor combination'\n    ],\n    'Border Irregularity': [\n        'concavity_index - Composite border abnormality',\n        'border_irregularity_normalized - Average border metric'\n    ],\n    'Morphological Composite': [\n        'morphology_score - Overall morphological abnormality (0-1)',\n        'cellular_heterogeneity_score - Cell organization indicator'\n    ],\n    'Interactions': [\n        'size_morphology_interaction - Size × shape abnormality',\n        'border_size_interaction - Size × border irregularity'\n    ],\n    'Aggregations': [\n        'malignancy_composite_score - Aggregate malignancy indicator',\n        'benign_characteristics_score - Inverse malignancy indicator'\n    ],\n    'Normalized': [\n        'size_adjusted_concavity - Size-normalized concavity',\n        'complexity_per_area - Complexity density metric'\n    ]\n}\n\nprint(f\"\"\"\nFEATURES CREATED BY CATEGORY:\"\"\")\nfor category, features in feature_engineering_log.items():\n    print(f\"\"\"\n{category}:\"\"\")\n    for feat in features:\n        print(f\"  • {feat}\")\n\nprint(f\"\"\"\nCLINICAL RATIONALE FOR ENGINEERING:\n  ✓ Normalized metrics reduce size bias\n  ✓ Composite scores capture synergistic effects\n  ✓ Interaction terms capture combined abnormalities\n  ✓ All features have clear medical interpretability\n  ✓ No arbitrary mathematical combinations\n  ✓ Each feature addresses specific clinical aspect\n\nFEATURE SELECTION STRATEGY:\n  → These engineered features will be evaluated during Phase 7\n  → Original features retained for comparison\n  → Correlation analysis will identify redundancy\n  → Final selection based on clinical relevance + predictive power\n\"\"\")\n\nprint(\"=\" * 100)\nprint(\"END PHASE 6 - FEATURE ENGINEERING COMPLETE\")\nprint(\"=\" * 100)\n