"""
Aim 2 — Risk-factor association profiling via apriori / market-basket analysis
(TEMPLATE).

Follows the methodology of Nuryunarsih et al. (2024, Journal of Medical
Artificial Intelligence): binary-encoded attribute matrix, apriori algorithm
(mlxtend), minimum support >=60%, association rules with support/confidence/lift.

Usage
-----
    python scripts/03_aim2_apriori_association.py --input data/processed_data.csv \
        --outdir results/aim2 --min_support 0.6
"""

import argparse
import os

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all candidate attribute columns to boolean (True = attribute present),
    per the Yes/No (1/2) coding scheme in docs/data_dictionary.md.
    Assumes columns are already numerically coded upstream (01_data_cleaning.py).
    Continuous variables (age, BMI, blood pressure, cognitive scores) should be
    pre-binned into categorical flags before calling this function -- see
    docs/data_dictionary.md for the coding scheme used in the source HTN papers.
    """
    return df.astype(bool)


def run_apriori(df_bool: pd.DataFrame, min_support: float = 0.6, min_confidence: float = 0.7):
    frequent_itemsets = apriori(df_bool, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )
    rules = rules.sort_values(["confidence", "support"], ascending=False)
    return frequent_itemsets, rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed_data.csv")
    parser.add_argument("--outdir", default="results/aim2")
    parser.add_argument("--min_support", type=float, default=0.6,
                         help="Minimum support threshold (default 0.6, per Nuryunarsih et al. 2024)")
    parser.add_argument("--min_confidence", type=float, default=0.7)
    parser.add_argument("--outcome_col", default="cognitive_impairment",
                         help="Column to always include as a candidate consequent, "
                              "e.g. to replicate 'Diagnosis' association rules")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    # Drop obviously continuous columns not yet binned -- adapt to actual schema
    candidate_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
    print(f"Using {len(candidate_cols)} binary-coded columns for apriori analysis")

    df_bool = binarize(df[candidate_cols])

    frequent_itemsets, rules = run_apriori(
        df_bool, min_support=args.min_support, min_confidence=args.min_confidence
    )

    freq_path = os.path.join(args.outdir, "frequent_itemsets.csv")
    rules_path = os.path.join(args.outdir, "association_rules.csv")
    frequent_itemsets.to_csv(freq_path, index=False)
    rules.to_csv(rules_path, index=False)

    print(f"\nFound {len(frequent_itemsets)} frequent itemsets (min_support={args.min_support})")
    print(f"Found {len(rules)} association rules (min_confidence={args.min_confidence})")
    print(f"Saved: {freq_path}")
    print(f"Saved: {rules_path}")

    # Rules where outcome column is the consequent -- mirrors Table 5 in
    # Nuryunarsih et al. (2024)
    if args.outcome_col in df.columns:
        outcome_rules = rules[
            rules["consequents"].apply(lambda x: args.outcome_col in x)
        ]
        outcome_path = os.path.join(args.outdir, f"rules_to_{args.outcome_col}.csv")
        outcome_rules.to_csv(outcome_path, index=False)
        print(f"Saved {len(outcome_rules)} rules with consequent='{args.outcome_col}' to {outcome_path}")


if __name__ == "__main__":
    main()
