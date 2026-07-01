"""
Enhanced Breast Cancer Diagnosis System
Phase 4: Medical Data Validation
Healthcare Perspective Review
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('breast_cancer_enhanced_dataset.csv')

print("=" * 100)
print("PHASE 4: MEDICAL DATA VALIDATION")
print("=" * 100)

print("\nReview of all variables from clinical/healthcare perspective\n")

# ============================================================================
# FEATURE CATEGORIZATION
# ============================================================================
print("1. FEATURE CATEGORIZATION FROM CLINICAL PERSPECTIVE")
print("=" * 100)

# Define feature categories with clinical assessment
feature_assessment = {
    'PRIMARY MORPHOLOGICAL FEATURES (Strong Clinical Indicators)': {
        'radius_mean': {
            'clinical_value': '★★★★★',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Direct measure of tumor size; larger malignant tumors are typical',
            'retention': '✓ RETAIN'
        },
        'perimeter_mean': {
            'clinical_value': '★★★★☆',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Size metric; highly correlated with radius',
            'retention': '⚠ REVIEW - Multicollinearity risk'
        },
        'area_mean': {
            'clinical_value': '★★★★☆',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Normalized size metric; correlated with radius',
            'retention': '⚠ REVIEW - Multicollinearity risk'
        },
    },
    'BORDER IRREGULARITY FEATURES (Very Strong Indicators)': {
        'concavity_mean': {
            'clinical_value': '★★★★★',
            'assessment': 'VERY STRONG PREDICTOR',
            'reason': 'Malignant tumors typically show irregular, concave borders',
            'retention': '✓ RETAIN - Critical feature'
        },
        'concave_points_mean': {
            'clinical_value': '★★★★★',
            'assessment': 'VERY STRONG PREDICTOR',
            'reason': 'Number of concave regions; key malignancy indicator',
            'retention': '✓ RETAIN - Critical feature'
        },
        'border_complexity': {
            'clinical_value': '★★★★☆',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Fractal dimension indicates border irregularity',
            'retention': '✓ RETAIN'
        },
    },
    'SHAPE AND TEXTURE FEATURES (Important Indicators)': {
        'shape_irregularity': {
            'clinical_value': '★★★★☆',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Nuclear pleomorphism; malignant cells show irregular shapes',
            'retention': '✓ RETAIN'
        },
        'texture_mean': {
            'clinical_value': '★★★★☆',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Cell heterogeneity; indicates disorganized tissue',
            'retention': '✓ RETAIN'
        },
        'smoothness_mean': {
            'clinical_value': '★★★☆☆',
            'assessment': 'MODERATE PREDICTOR',
            'reason': 'Surface regularity; clinical relevance moderate',
            'retention': '✓ RETAIN'
        },
        'compactness_mean': {
            'clinical_value': '★★★★☆',
            'assessment': 'STRONG PREDICTOR',
            'reason': 'Measures shape compactness; abnormal shapes suggest malignancy',
            'retention': '✓ RETAIN'
        },
    },
    'DERIVED/INTERACTION FEATURES (Questionable)': {
        'tumor_aggressiveness': {
            'clinical_value': '⚠ UNCLEAR',
            'assessment': 'POTENTIAL LEAKAGE',
            'reason': 'Pre-calculated score; may contain information from diagnosis',
            'retention': '✗ REMOVE - Leakage risk'
        },
        'radius_texture_interaction': {
            'clinical_value': '★★☆☆☆',
            'assessment': 'WEAK/ENGINEERED',
            'reason': 'Mathematical interaction; limited independent clinical meaning',
            'retention': '⚠ REVIEW - May be redundant'
        },
        'radius_concavity_interaction': {
            'clinical_value': '★★☆☆☆',
            'assessment': 'WEAK/ENGINEERED',
            'reason': 'Mathematical interaction; may capture combined effect',
            'retention': '⚠ REVIEW - May be redundant'
        },
        'concavity_density': {
            'clinical_value': '★★☆☆☆',
            'assessment': 'WEAK/DERIVED',
            'reason': 'Normalized metric; derived from other features',
            'retention': '⚠ REVIEW - May be redundant'
        },
    },
    'IDENTIFIER COLUMN': {
        'id': {
            'clinical_value': '✗ NONE',
            'assessment': 'METADATA',
            'reason': 'Patient identifier; should never be used for modeling',
            'retention': '✗ REMOVE - Always'
        },
    }
}

# Print detailed assessment
for category, features in feature_assessment.items():
    print(f"\n{category}")
    print("─" * 100)
    
    for feature, details in features.items():
        print(f"\n  Feature: {feature}")
        print(f"  ├─ Clinical Value: {details['clinical_value']}")
        print(f"  ├─ Assessment: {details['assessment']}")
        print(f"  ├─ Reason: {details['reason']}")
        print(f"  └─ Recommendation: {details['retention']}")

# ============================================================================
# MULTICOLLINEARITY ANALYSIS
# ============================================================================
print("\n\n2. MULTICOLLINEARITY ANALYSIS (Redundant Measurements)")
print("=" * 100)

df_encoded = df.copy()
df_encoded['diagnosis'] = (df_encoded['diagnosis'] == 'M').astype(int)
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.drop('id')
corr_matrix = df_encoded[numeric_cols].corr()

print("\nHighly Correlated Feature Pairs (|r| > 0.90):")
print("─" * 100)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.90 and 'diagnosis' not in corr_matrix.columns[i]:
            pair = {
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j],
                'Issue': 'MULTICOLLINEARITY'
            }
            high_corr_pairs.append(pair)

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    for idx, row in high_corr_df.iterrows():
        print(f"\n  {row['Feature 1']} ←→ {row['Feature 2']}")
        print(f"  Correlation: {row['Correlation']:.4f}")
        print(f"  ├─ These measure similar characteristics")
        print(f"  └─ Consider: Keep only the clinically most important one")
else:
    print("  No pairs with |r| > 0.90 found")

print("\nModerate Correlation Pairs (0.85 < |r| < 0.90):")
print("─" * 100)

moderate_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if 0.85 < abs(corr_matrix.iloc[i, j]) <= 0.90 and 'diagnosis' not in corr_matrix.columns[i]:
            moderate_corr.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if moderate_corr:
    moderate_df = pd.DataFrame(moderate_corr).sort_values('Correlation', key=abs, ascending=False)
    print(moderate_df.to_string(index=False))
else:
    print("  No pairs with 0.85 < |r| < 0.90 found")

# ============================================================================
# WEAK CLINICAL INDICATORS
# ============================================================================
print("\n\n3. WEAK CLINICAL INDICATORS ANALYSIS")
print("=" * 100)

print("\nFeatures with Weak Correlation to Diagnosis (|r| < 0.30):")
print("─" * 100)

diagnosis_corr = corr_matrix['diagnosis'].drop('diagnosis').sort_values(key=abs)

weak_indicators = diagnosis_corr[abs(diagnosis_corr) < 0.30]
if len(weak_indicators) > 0:
    for feat, corr in weak_indicators.items():
        print(f"\n  {feat}: r = {corr:.4f}")
        print(f"  ├─ Very weak correlation with diagnosis")
        print(f"  ├─ Limited predictive value")
        print(f"  └─ Consider: May be redundant or noisy")
else:
    print("  All features show reasonable correlation (|r| > 0.30)")

# ============================================================================
# NOISE VARIABLES ASSESSMENT
# ============================================================================
print("\n\n4. NOISE VARIABLES ASSESSMENT")
print("=" * 100)

print("\nCriteria for Noise Identification:")
print("  • Very low variance (high redundancy)")
print("  • Very low correlation with target (no predictive power)")
print("  • Highly skewed distributions (unreliable estimates)")

numeric_features = [col for col in df.columns if col not in ['id', 'diagnosis']]

noise_score = {}
for feat in numeric_features:
    # Variance score (lower is more redundant)
    variance_score = df[feat].var()
    
    # Correlation with target
    benign = df[df['diagnosis'] == 'B'][feat]
    malignant = df[df['diagnosis'] == 'M'][feat]
    target_corr = abs(corr_matrix.loc[feat, 'diagnosis'])
    
    # Skewness (higher = more skewed = less reliable)
    skewness = abs(df[feat].skew())
    
    # Calculate noise score (lower = less noisy)
    noise_score[feat] = {
        'variance': variance_score,
        'target_corr': target_corr,
        'skewness': skewness,
        'noise_level': 'LOW' if target_corr > 0.30 else 'MODERATE' if target_corr > 0.20 else 'HIGH'
    }

print("\nNoise Assessment by Feature:")
print("─" * 100)
print(f"{'Feature':<35} | {'Target Corr':<12} | {'Noise Level':<12}")
print("─" * 100)

for feat in sorted(noise_score.keys(), key=lambda x: noise_score[x]['target_corr']):
    print(f"{feat:<35} | {noise_score[feat]['target_corr']:>11.4f} | {noise_score[feat]['noise_level']:<12}")

# ============================================================================
# FEATURE REDUNDANCY
# ============================================================================
print("\n\n5. REDUNDANT MEASUREMENTS IDENTIFICATION")
print("=" * 100)

print("\nSize Metrics (Likely Redundant):")
print("  • radius_mean: ✓ PRIMARY - Keep")
print("  • perimeter_mean: ⚠ DERIVED - Perimeter ≈ 2π × radius")
print("  • area_mean: ⚠ DERIVED - Area ≈ π × radius²")
print("\n  Clinical Note: These three measure the same underlying dimension.")
print("  Recommendation: Consider keeping only radius_mean OR one dimensionless metric")

print("\nBorder Irregularity Metrics:")
print("  • concavity_mean: ✓ PRIMARY - Keep")
print("  • concave_points_mean: ✓ PRIMARY - Keep (counts concave regions)")
print("  • border_complexity: ⚠ DERIVED - Related to concavity")
print("\n  Clinical Note: All three capture border irregularity but from different angles.")
print("  Recommendation: All three may be complementary; retain all for initial model")

print("\nShape Features:")
print("  • shape_irregularity: ✓ PRIMARY - Keep")
print("  • compactness_mean: Partially related (shape compactness)")
print("\n  Clinical Note: Both capture morphological abnormalities.")
print("  Recommendation: Retain both; check for redundancy during feature selection")

# ============================================================================
# ENGINEERED FEATURES ASSESSMENT
# ============================================================================
print("\n\n6. ENGINEERED FEATURES - CLINICAL VALIDITY")
print("=" * 100)

engineered = {
    'tumor_aggressiveness': {
        'type': 'Composite Score',
        'issue': 'CRITICAL LEAKAGE RISK',
        'reason': 'Pre-calculated aggressiveness score likely derived from diagnosis or related clinical assessments',
        'action': '✗ REMOVE IMMEDIATELY'
    },
    'radius_texture_interaction': {
        'type': 'Interaction Term',
        'issue': 'Limited Clinical Meaning',
        'reason': 'Mathematical product of radius and texture; not a recognized clinical metric',
        'action': '⚠ CONSIDER REMOVING (Redundant with raw features)'
    },
    'radius_concavity_interaction': {
        'type': 'Interaction Term',
        'issue': 'Limited Clinical Meaning',
        'reason': 'Mathematical product; not a standard clinical measurement',
        'action': '⚠ CONSIDER REMOVING (May capture combined effect; verify usefulness)'
    },
    'concavity_density': {
        'type': 'Normalized Ratio',
        'issue': 'Potential Leakage',
        'reason': 'Derived metric; concavity per area may encode target information',
        'action': '⚠ REVIEW - May be correlated with diagnosis through derivation'
    }
}

for feat, assessment in engineered.items():
    print(f"\n{feat}")
    print(f"  Type: {assessment['type']}")
    print(f"  Issue: {assessment['issue']}")
    print(f"  Reason: {assessment['reason']}")
    print(f"  Action: {assessment['action']}")

# ============================================================================
# RECOMMENDATIONS SUMMARY
# ============================================================================
print("\n\n7. CLINICAL VALIDATION SUMMARY & RECOMMENDATIONS")
print("=" * 100)

print("""
FEATURE RETENTION RECOMMENDATIONS:

✓ DEFINITELY RETAIN (Clinical Gold Standard):
  • radius_mean - Direct tumor size measure
  • concavity_mean - Key malignancy indicator
  • concave_points_mean - Border irregularity indicator
  • texture_mean - Cell heterogeneity indicator
  • shape_irregularity - Nuclear pleomorphism indicator
  • compactness_mean - Shape abnormality indicator
  • smoothness_mean - Surface regularity indicator
  • border_complexity - Border irregularity (fractal dimension)

⚠ REVIEW FOR REDUNDANCY:
  • perimeter_mean - Highly correlated with radius (derived)
  • area_mean - Highly correlated with radius (derived)
  → Recommendation: Consider keeping only radius for size dimension

⚠ QUESTIONABLE VALIDITY:
  • radius_texture_interaction - Engineered feature; limited clinical meaning
  • radius_concavity_interaction - Engineered feature; verify usefulness
  • concavity_density - Derived metric; check for correlation structure

✗ REMOVE IMMEDIATELY (Leakage Risk):
  • tumor_aggressiveness - Pre-calculated risk score (likely contains diagnosis info)
  • id - Patient identifier (never use for modeling)

✗ ALWAYS REMOVE (Metadata):
  • id - For data leakage prevention

CLINICAL FEATURE ENGINEERING OPPORTUNITIES:
  1. Size to Shape Ratio: radius / shape_irregularity
  2. Concavity Index: concave_points / perimeter
  3. Morphology Score: average of normalized shape features
  4. Border Irregularity Score: mean of concavity and border_complexity

TOTAL FEATURES RECOMMENDED FOR MODELING: 8-10 core features
  (After removing redundant size metrics and questionable derived features)
""")

print("=" * 100)
print("END PHASE 4 - MEDICAL VALIDATION COMPLETE")
print("=" * 100)
