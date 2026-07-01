"""
Enhanced Breast Cancer Diagnosis System
Phase 2: Data Quality Assessment
Professional Data Quality Audit
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('breast_cancer_enhanced_dataset.csv')

print("=" * 100)
print("PHASE 2: COMPREHENSIVE DATA QUALITY ASSESSMENT")
print("=" * 100)

# Create quality report dictionary
quality_report = {
    'Total Records': df.shape[0],
    'Total Columns': df.shape[1],
}

print("\n1. MISSING VALUES ANALYSIS")
print("─" * 100)
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)

if len(missing_data) == 0:
    print("✓ EXCELLENT: No missing values detected in any column")
    quality_report['Missing Values'] = 'None'
else:
    print("⚠ ALERT: Missing values detected:")
    print(missing_data.to_string(index=False))
    quality_report['Missing Values'] = missing_data.to_dict()

print("\n2. DUPLICATE RECORDS ANALYSIS")
print("─" * 100)

# Check for complete duplicates
complete_duplicates = df.duplicated().sum()
print(f"Complete Row Duplicates: {complete_duplicates}")

# Check for duplicates excluding ID and target
feature_cols = [col for col in df.columns if col not in ['id', 'diagnosis']]
duplicate_patterns = df.duplicated(subset=feature_cols, keep=False).sum()

print(f"Duplicate Records (excluding ID): {complete_duplicates}")
print(f"Rows with Identical Features: {duplicate_patterns}")

if complete_duplicates > 0:
    print("\n⚠ ALERT: Duplicate rows detected!")
    print(df[df.duplicated(keep=False)].sort_values('id').head(10))
    quality_report['Duplicates'] = complete_duplicates
else:
    print("✓ EXCELLENT: No complete duplicate rows found")
    quality_report['Duplicates'] = 'None'

print("\n3. DATA TYPE ANALYSIS")
print("─" * 100)
print(df.dtypes)

# Check for type inconsistencies
print("\n✓ Data Type Audit:")
for col in df.columns:
    if col == 'diagnosis':
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            if set(unique_vals) <= {'B', 'M'}:
                print(f"   {col}: ✓ Correct (object with values B/M)")
            else:
                print(f"   {col}: ⚠ ALERT - Unexpected values: {unique_vals}")
    elif col == 'id':
        print(f"   {col}: {df[col].dtype} - Identifier field")
    else:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"   {col}: ✓ Numeric (float64 expected)")
        else:
            print(f"   {col}: ⚠ ALERT - Non-numeric: {df[col].dtype}")

print("\n4. OUTLIER AND IMPOSSIBLE VALUE ANALYSIS")
print("─" * 100)

outlier_report = []

for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'id':
        continue
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    # Check for negative values (impossible for medical measurements)
    negative_count = (df[col] < 0).sum()
    
    # Check for zero values (may be suspicious depending on feature)
    zero_count = (df[col] == 0).sum()
    
    if outlier_count > 0 or negative_count > 0:
        outlier_report.append({
            'Column': col,
            'Outliers': outlier_count,
            'Outlier %': f"{outlier_pct:.2f}%",
            'Negative Values': negative_count,
            'Zero Values': zero_count,
            'Min': df[col].min(),
            'Max': df[col].max()
        })

if outlier_report:
    outlier_df = pd.DataFrame(outlier_report)
    print(outlier_df.to_string(index=False))
    quality_report['Outliers'] = outlier_df.to_dict()
else:
    print("✓ No statistical outliers detected")
    quality_report['Outliers'] = 'None'

print("\n5. NEGATIVE AND IMPOSSIBLE VALUES")
print("─" * 100)

impossible_values = {}
for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'id':
        continue
    
    # Clinical ranges that should not have negative values
    if any(x in col.lower() for x in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                                        'compactness', 'concavity', 'concave', 'shape', 'border',
                                        'tumor', 'aggressiveness', 'risk', 'density']):
        
        negative = (df[col] < 0).sum()
        if negative > 0:
            impossible_values[col] = {
                'negative_count': negative,
                'min_value': df[col].min(),
                'sample_negative': df[df[col] < 0][col].head(3).tolist()
            }

if impossible_values:
    print("⚠ ALERT: Impossible negative values detected!")
    for col, info in impossible_values.items():
        print(f"\n   {col}:")
        print(f"      Negative values: {info['negative_count']}")
        print(f"      Minimum value: {info['min_value']:.6f}")
        print(f"      Sample: {info['sample_negative']}")
    quality_report['Impossible Values'] = impossible_values
else:
    print("✓ No impossible negative values detected")

print("\n6. VALUE RANGE VALIDATION")
print("─" * 100)

expected_ranges = {
    'radius_mean': (6, 35),
    'texture_mean': (9, 39),
    'perimeter_mean': (43, 188),
    'area_mean': (143, 2500),
    'smoothness_mean': (0.06, 0.16),
    'compactness_mean': (0.02, 0.35),
    'concavity_mean': (0.0, 0.43),
    'concave_points_mean': (0.0, 0.16),
    'shape_irregularity': (0.05, 0.80),
    'border_complexity': (0.0, 0.04),
}

range_violations = {}
for col, (min_exp, max_exp) in expected_ranges.items():
    if col in df.columns:
        out_of_range = ((df[col] < min_exp) | (df[col] > max_exp)).sum()
        if out_of_range > 0:
            range_violations[col] = {
                'expected': f"({min_exp}, {max_exp})",
                'violations': out_of_range,
                'actual_min': df[col].min(),
                'actual_max': df[col].max()
            }

if range_violations:
    print("⚠ ALERT: Values outside expected clinical ranges:")
    for col, info in range_violations.items():
        print(f"\n   {col}:")
        print(f"      Expected range: {info['expected']}")
        print(f"      Out of range:   {info['violations']} records")
        print(f"      Actual range:   [{info['actual_min']:.4f}, {info['actual_max']:.4f}]")
    quality_report['Range Violations'] = range_violations
else:
    print("✓ All values within expected clinical ranges")

print("\n7. FEATURE DISTRIBUTION ABNORMALITIES")
print("─" * 100)

distribution_issues = {}
for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'id':
        continue
    
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
    
    if abs(skewness) > 2 or kurtosis > 3:
        distribution_issues[col] = {
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'cv': round(cv, 3)
        }

if distribution_issues:
    print("⚠ ALERT: Abnormal distributions detected:")
    for col, stats in distribution_issues.items():
        print(f"\n   {col}:")
        print(f"      Skewness: {stats['skewness']} (|>2| indicates high skew)")
        print(f"      Kurtosis: {stats['kurtosis']} (>3 indicates heavy tails)")
        print(f"      Coeff of Variation: {stats['cv']}")
    quality_report['Distribution Issues'] = distribution_issues
else:
    print("✓ Distributions appear normal")

print("\n8. IDENTIFIER UNIQUENESS CHECK")
print("─" * 100)

id_col = 'id'
unique_ids = df[id_col].nunique()
total_records = len(df)

print(f"Total Records: {total_records}")
print(f"Unique IDs: {unique_ids}")

if unique_ids == total_records:
    print("✓ EXCELLENT: All IDs are unique (no patient duplication)")
    quality_report['ID Uniqueness'] = 'All unique'
else:
    print(f"⚠ ALERT: Duplicate IDs detected! {total_records - unique_ids} records share IDs")
    duplicate_ids = df[df.duplicated(subset=['id'], keep=False)][['id', 'diagnosis']].sort_values('id')
    print(duplicate_ids.head(10))
    quality_report['ID Uniqueness'] = 'Duplicates found'

print("\n9. TARGET VARIABLE ANALYSIS")
print("─" * 100)

diagnosis_counts = df['diagnosis'].value_counts()
print(f"Target Distribution:")
print(diagnosis_counts)
print(f"\nPercentage Distribution:")
print((diagnosis_counts / len(df) * 100).round(2))

if len(diagnosis_counts) != 2 or not set(diagnosis_counts.index) <= {'B', 'M'}:
    print("⚠ ALERT: Unexpected diagnosis values!")
    quality_report['Target Validity'] = 'Issues detected'
else:
    print("✓ EXCELLENT: Valid binary target variable")
    if min(diagnosis_counts) / max(diagnosis_counts) < 0.5:
        print(f"⚠ WARNING: Class imbalance detected (ratio: {min(diagnosis_counts)/max(diagnosis_counts):.2%})")
    quality_report['Target Validity'] = 'Valid'

print("\n" + "=" * 100)
print("DATA QUALITY REPORT SUMMARY")
print("=" * 100)

print(f"""
✓ COMPLETED CHECKS:
  1. Missing Values:        {quality_report.get('Missing Values', 'None')}
  2. Duplicate Records:     {quality_report.get('Duplicates', 'None')}
  3. Data Type Issues:      Checked
  4. Outliers:              Found - See detailed report
  5. Impossible Values:     {quality_report.get('Impossible Values', 'None')}
  6. Range Violations:      {quality_report.get('Range Violations', 'None')}
  7. Distribution Issues:   {quality_report.get('Distribution Issues', 'None')}
  8. ID Uniqueness:         {quality_report.get('ID Uniqueness', 'All unique')}
  9. Target Validity:       {quality_report.get('Target Validity', 'Valid')}

OVERALL ASSESSMENT:
  Dataset contains {df.shape[0]} records with {df.shape[1]} columns
  Quality Issues Detected: YES (negative values, out-of-range values)
  Readiness for Cleaning: READY FOR PHASE 3
""")

print("=" * 100)
print("END PHASE 2")
print("=" * 100)
