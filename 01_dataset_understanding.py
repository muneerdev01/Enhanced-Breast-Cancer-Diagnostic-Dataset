"""
PHASE 1: DATASET UNDERSTANDING
Enhanced Breast Cancer Diagnosis System
Healthcare Data Science Workflow
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('breast_cancer_enhanced_dataset.csv')

print("="*80)
print("PHASE 1: DATASET UNDERSTANDING & COLUMN ANALYSIS")
print("="*80)

# Basic dataset info
print("\n1. DATASET DIMENSIONS")
print(f"   Shape: {df.shape[0]} records × {df.shape[1]} features")
print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n2. COLUMN-BY-COLUMN MEDICAL ANALYSIS")
print("-"*80)

# Define medical feature metadata
feature_metadata = {
    'id': {
        'type': 'Identifier',
        'medical_meaning': 'Unique patient identifier',
        'data_type': 'Integer',
        'expected_range': 'Any unique value',
        'clinical_significance': 'Patient tracking only - should not be used in model',
        'usage': 'Identifier - remove for ML'
    },
    'diagnosis': {
        'type': 'Target Variable',
        'medical_meaning': 'Cancer diagnosis (B=Benign, M=Malignant)',
        'data_type': 'Categorical (Binary)',
        'expected_range': 'B or M only',
        'clinical_significance': 'Primary outcome variable for classification',
        'usage': 'Target variable'
    },
    'radius_mean': {
        'type': 'Morphometric Feature',
        'medical_meaning': 'Mean radius of cell nuclei in the tumor',
        'data_type': 'Float',
        'expected_range': '6.0 - 28.0 mm',
        'clinical_significance': 'Larger nuclei often indicate malignancy',
        'usage': 'Predictive feature'
    },
    'texture_mean': {
        'type': 'Morphometric Feature',
        'medical_meaning': 'Mean texture (std dev of gray-scale values)',
        'data_type': 'Float',
        'expected_range': '9.0 - 39.0',
        'clinical_significance': 'Texture variation indicates cellular irregularity',
        'usage': 'Predictive feature'
    },
    'perimeter_mean': {
        'type': 'Morphometric Feature',
        'medical_meaning': 'Mean perimeter of cell nuclei',
        'data_type': 'Float',
        'expected_range': '43.0 - 188.5 mm',
        'clinical_significance': 'Related to nucleus size; larger perimeter suggests abnormality',
        'usage': 'Predictive feature'
    },
    'area_mean': {
        'type': 'Morphometric Feature',
        'medical_meaning': 'Mean area of cell nuclei',
        'data_type': 'Float',
        'expected_range': '143.0 - 2500.0 mm²',
        'clinical_significance': 'Cell area is key indicator of pathological changes',
        'usage': 'Predictive feature'
    },
    'smoothness_mean': {
        'type': 'Texture Feature',
        'medical_meaning': 'Mean local variation in radius lengths',
        'data_type': 'Float',
        'expected_range': '0.06 - 0.16',
        'clinical_significance': 'Smoothness indicates regular nuclear boundary',
        'usage': 'Predictive feature'
    },
    'compactness_mean': {
        'type': 'Morphometric Feature',
        'medical_meaning': 'Mean (perimeter²/area - 1.0)',
        'data_type': 'Float',
        'expected_range': '0.02 - 0.35',
        'clinical_significance': 'Measures how compact nucleus is; malignant cells less compact',
        'usage': 'Predictive feature'
    },
    'concavity_mean': {
        'type': 'Shape Feature',
        'medical_meaning': 'Mean severity of concave portions in nucleus outline',
        'data_type': 'Float',
        'expected_range': '0.0 - 0.43',
        'clinical_significance': 'Concavity indicates irregular nuclear boundaries (high in malignant)',
        'usage': 'Predictive feature'
    },
    'concave points_mean': {
        'type': 'Shape Feature',
        'medical_meaning': 'Mean number of concave points in nucleus outline',
        'data_type': 'Float',
        'expected_range': '0.0 - 0.29',
        'clinical_significance': 'Irregular nuclear boundaries suggest malignancy',
        'usage': 'Predictive feature'
    },
    'shape_irregularity': {
        'type': 'Morphometric Feature',
        'medical_meaning': 'Deviation from circular shape (normalized radius variation)',
        'data_type': 'Float',
        'expected_range': '0.0 - 0.60',
        'clinical_significance': 'Regular shapes suggest benign; irregular suggest malignant',
        'usage': 'Predictive feature'
    },
    'border_complexity': {
        'type': 'Texture Feature',
        'medical_meaning': 'Fractal dimension-like measure of nucleus border',
        'data_type': 'Float',
        'expected_range': '0.0 - 0.03',
        'clinical_significance': 'Complex borders associated with malignancy',
        'usage': 'Predictive feature'
    },
    'tumor_aggressiveness': {
        'type': 'Derived Metric',
        'medical_meaning': 'Combined measure of nuclear abnormality',
        'data_type': 'Float',
        'expected_range': '0.0 - 0.30',
        'clinical_significance': 'Proxy for mitotic activity; higher in aggressive tumors',
        'usage': 'Potential leakage - derived from other features'
    },
    'radius_texture_interaction': {
        'type': 'Interaction Feature',
        'medical_meaning': 'Product of radius × texture (radius_mean × texture_mean)',
        'data_type': 'Float',
        'expected_range': '100.0 - 900.0',
        'clinical_significance': 'Combined metric of size and texture irregularity',
        'usage': 'Potential leakage - derived feature'
    },
    'radius_concavity_interaction': {
        'type': 'Interaction Feature',
        'medical_meaning': 'Product of radius × concavity',
        'data_type': 'Float',
        'expected_range': '0.0 - 5.0',
        'clinical_significance': 'Combined measure of size and shape irregularity',
        'usage': 'Potential leakage - derived feature'
    },
    'concavity_density': {
        'type': 'Density Metric',
        'medical_meaning': 'Concavity normalized by area (concavity/area)',
        'data_type': 'Float',
        'expected_range': '0.0 - 0.0005',
        'clinical_significance': 'Concavity intensity per unit area',
        'usage': 'Potential leakage - derived feature'
    },
    'malignancy_risk_score': {
        'type': 'Composite Score',
        'medical_meaning': 'Pre-computed malignancy probability/risk metric',
        'data_type': 'Float',
        'expected_range': '15.0 - 55.0',
        'clinical_significance': 'HIGH LEAKAGE RISK - Contains diagnosis information',
        'usage': 'MUST REMOVE - Direct target leakage'
    }
}

# Create and display analysis table
analysis_data = []
for col in df.columns:
    if col in feature_metadata:
        meta = feature_metadata[col]
        analysis_data.append({
            'Column': col,
            'Type': meta['type'],
            'Medical Meaning': meta['medical_meaning'],
            'Data Type': meta['data_type'],
            'Range': meta['expected_range'],
            'Clinical Significance': meta['clinical_significance'],
            'Usage': meta['usage']
        })

analysis_df = pd.DataFrame(analysis_data)

print("\nFEATURE ANALYSIS TABLE:")
for idx, row in analysis_df.iterrows():
    print(f"\n{idx+1}. {row['Column'].upper()}")
    print(f"   ├─ Type: {row['Type']}")
    print(f"   ├─ Medical Meaning: {row['Medical Meaning']}")
    print(f"   ├─ Data Type: {row['Data Type']}")
    print(f"   ├─ Expected Range: {row['Range']}")
    print(f"   ├─ Clinical Significance: {row['Clinical Significance']}")
    print(f"   └─ Usage: {row['Usage']}")

print("\n" + "="*80)
print("3. VARIABLE CLASSIFICATION SUMMARY")
print("="*80)

print("\n✓ TARGET VARIABLE:")
print("   - diagnosis (B/M binary classification)")

print("\n✓ PREDICTIVE FEATURES (Primary morphometric measurements):")
primary_predictive = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'shape_irregularity', 'border_complexity'
]
for feat in primary_predictive:
    print(f"   - {feat}")

print("\n⚠ IDENTIFIER COLUMNS (Remove from model):")
print("   - id (patient identifier)")

print("\n⚠ POTENTIAL LEAKAGE COLUMNS (High risk - review carefully):")
leakage_features = [
    'tumor_aggressiveness', 'radius_texture_interaction',
    'radius_concavity_interaction', 'concavity_density', 'malignancy_risk_score'
]
for feat in leakage_features:
    print(f"   - {feat}")

print("\n" + "="*80)
print("4. CURRENT DATA SAMPLE")
print("="*80)
print(df.head())

print("\n" + "="*80)
print("5. DATA TYPE OVERVIEW")
print("="*80)
print(df.dtypes)

print("\n" + "="*80)
print("PHASE 1 COMPLETE: Dataset Understanding Documented")
print("="*80)

# Save analysis for next phase
df.to_csv('00_raw_dataset.csv', index=False)
print("\n✓ Raw dataset saved to '00_raw_dataset.csv'")
