"""
Data cleaning and preparation pipeline (TEMPLATE).

Mirrors the cleaning approach used in Nuryunarsih et al. (2023, 2025):
- median imputation for isolated missing values
- IQR-based outlier bounds for continuous variables (age, BMI, blood pressure, etc.)
- categorical encoding per docs/data_dictionary.md
- train/test split preparation

This is a TEMPLATE: column names below are placeholders matching the
questionnaire domains in docs/data_dictionary.md. Adjust to match the actual
exported column names from the electronic case report form (e.g., REDCap) once
data collection begins.

Usage
-----
    python scripts/01_data_cleaning.py --input data/raw_data.csv --output data/processed_data.csv
"""

import argparse
import pandas as pd
import numpy as np


CONTINUOUS_VARS = [
    "age", "bmi", "sbp", "dbp", "mmse_score", "moca_score", "iqcode_score",
]

# categorical columns expected to be Yes/No (1/2) per data_dictionary.md
BINARY_YESNO_VARS = [
    "hypertension", "diabetes", "cvd_history", "stroke_tia", "family_history_dementia",
    "physical_activity", "smoking", "alcohol_use",
]


def iqr_bounds(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_continuous(df: pd.DataFrame, cols=CONTINUOUS_VARS) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        # median imputation for missing values
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        if n_missing:
            print(f"[{col}] imputing {n_missing} missing values with median={median_val:.2f}")
            df[col] = df[col].fillna(median_val)
        # IQR-based outlier winsorisation
        low, high = iqr_bounds(df[col])
        n_outliers = ((df[col] < low) | (df[col] > high)).sum()
        if n_outliers:
            print(f"[{col}] winsorising {n_outliers} outliers to [{low:.1f}, {high:.1f}]")
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def encode_binary(df: pd.DataFrame, cols=BINARY_YESNO_VARS) -> pd.DataFrame:
    """Recode Yes/No (1/2) items to 0/1 for modelling; leaves already-numeric as-is."""
    df = df.copy()
    mapping = {1: 1, 2: 0, "yes": 1, "no": 0, "Yes": 1, "No": 0}
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].map(mapping).fillna(df[col])
    return df


def check_multicollinearity(df: pd.DataFrame, threshold: float = 0.9) -> list:
    """Flag predictor pairs with |correlation| > threshold, per PROTOCOL.md Section 4.6."""
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > threshold:
                pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))
    if pairs:
        print(f"WARNING: {len(pairs)} predictor pairs exceed |r|>{threshold} — consider consolidation:")
        for a, b, r in pairs:
            print(f"  {a} <-> {b}: r={r}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw_data.csv")
    parser.add_argument("--output", default="data/processed_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.input}")

    df = clean_continuous(df)
    df = encode_binary(df)
    check_multicollinearity(df)

    df.to_csv(args.output, index=False)
    print(f"Saved cleaned dataset to {args.output}")


if __name__ == "__main__":
    main()
