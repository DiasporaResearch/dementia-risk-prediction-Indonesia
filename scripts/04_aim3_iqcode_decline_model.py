"""
Aim 3 — Retrospective decline-trajectory profiling using IQCODE (TEMPLATE).

Models informant-rated 10-year cognitive decline (IQCODE score) against the
same multi-domain predictor set used in Aim 1, via both a linear/logistic
regression baseline and a tree-based model (Random Forest) for comparison,
per PROTOCOL.md Section 4.8.

IQCODE scoring note: each item is rated 1 (much improved) to 5 (much worse),
with 3 = no change; mean item score >=3.3-3.6 is commonly used as a screening
cutoff for likely cognitive decline (confirm exact cutoff against the specific
IQCODE version used -- short 16-item vs long 26-item form).

Usage
-----
    python scripts/04_aim3_iqcode_decline_model.py --input data/processed_data.csv \
        --outdir results/aim3
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

IQCODE_DECLINE_CUTOFF = 3.3  # placeholder -- confirm against IQCODE form used


def continuous_decline_models(X_train, X_test, y_train, y_test):
    results = {}

    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results["linear_regression"] = {
        "r2": r2_score(y_test, pred_lr),
        "mae": mean_absolute_error(y_test, pred_lr),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_lr))),
        "coefficients": dict(zip(X_train.columns, lr.coef_.tolist())),
    }

    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["random_forest"] = {
        "r2": r2_score(y_test, pred_rf),
        "mae": mean_absolute_error(y_test, pred_rf),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_rf))),
        "feature_importance": dict(zip(X_train.columns, rf.feature_importances_.tolist())),
    }
    return results


def binary_decline_models(X_train, X_test, y_train, y_test):
    results = {}

    logit = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    proba_logit = logit.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = {
        "auc": roc_auc_score(y_test, proba_logit),
        "coefficients": dict(zip(X_train.columns, logit.coef_[0].tolist())),
    }

    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "auc": roc_auc_score(y_test, proba_rf),
        "feature_importance": dict(zip(X_train.columns, rf.feature_importances_.tolist())),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed_data.csv")
    parser.add_argument("--iqcode_col", default="iqcode_score")
    parser.add_argument("--outdir", default="results/aim3")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    predictor_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in (args.iqcode_col, "cognitive_impairment")
    ]
    X = df[predictor_cols]

    # --- Continuous IQCODE score model ---
    y_cont = df[args.iqcode_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y_cont, test_size=0.20, random_state=42)
    cont_results = continuous_decline_models(X_train, X_test, y_train, y_test)

    # --- Binary decline-vs-no-decline model (cutoff-based) ---
    y_bin = (df[args.iqcode_col] >= IQCODE_DECLINE_CUTOFF).astype(int)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=0.20, stratify=y_bin, random_state=42
    )
    bin_results = binary_decline_models(X_train_b, X_test_b, y_train_b, y_test_b)

    all_results = {
        "continuous_iqcode_score_models": cont_results,
        "binary_decline_cutoff": IQCODE_DECLINE_CUTOFF,
        "binary_decline_models": bin_results,
    }

    out_path = os.path.join(args.outdir, "aim3_iqcode_models.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Continuous model R^2: linear={cont_results['linear_regression']['r2']:.3f}, "
          f"RF={cont_results['random_forest']['r2']:.3f}")
    print(f"Binary decline model AUC: logistic={bin_results['logistic_regression']['auc']:.3f}, "
          f"RF={bin_results['random_forest']['auc']:.3f}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
