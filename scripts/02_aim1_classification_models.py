"""
Aim 1 — Current cognitive impairment status prediction (TEMPLATE).

Decision Tree / Random Forest / XGBoost classification pipeline, following the
methodology of Nuryunarsih et al. (2025): 80/20 train-test split, 5-fold
cross-validation, grid-search hyperparameter tuning, class-imbalance weighting,
and reporting of accuracy, sensitivity, specificity, precision, F1-score, AUC,
and calibration (Brier score, calibration slope/intercept) per PROTOCOL.md
Section 4.8.

Also computes the three benchmark scores (BDSI, BDRM, ANU-ADRI) for comparison
-- see `compute_benchmark_scores()`, which requires mapping local questionnaire
items to each score's published item set (TEMPLATE -- verify against original
publications before use).

Usage
-----
    python scripts/02_aim1_classification_models.py --input data/processed_data.csv \
        --outcome cognitive_impairment --outdir results/aim1
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBClassifier


PARAM_GRIDS = {
    "decision_tree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini"],
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5, 10],
    },
    "xgboost": {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
    },
}

MODELS = {
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
    "xgboost": XGBClassifier(random_state=42, eval_metric="logloss"),
}


def evaluate_model(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def calibration_summary(y_true, y_proba, n_bins=10):
    """Returns calibration curve points; slope/intercept via simple linear fit
    of observed vs predicted bin means (see PROTOCOL.md Section 4.8)."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    if len(prob_pred) >= 2:
        slope, intercept = np.polyfit(prob_pred, prob_true, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "bin_predicted": prob_pred.tolist(),
        "bin_observed": prob_true.tolist(),
    }


def fit_and_evaluate(name, model, param_grid, X_train, y_train, X_test, y_test, outdir):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba)
    metrics.update(calibration_summary(y_test, y_proba))
    metrics["best_params"] = grid.best_params_

    print(f"\n[{name}] best params: {grid.best_params_}")
    print(f"[{name}] accuracy={metrics['accuracy']:.3f} sensitivity={metrics['sensitivity']:.3f} "
          f"specificity={metrics['specificity']:.3f} F1={metrics['f1_score']:.3f} "
          f"AUC={metrics['auc']:.3f} Brier={metrics['brier_score']:.3f}")

    if name == "decision_tree":
        dot_path = os.path.join(outdir, "decision_tree.dot")
        export_graphviz(
            best_model, out_file=dot_path, feature_names=X_train.columns,
            class_names=["No impairment", "Impairment"], filled=True, rounded=True,
        )
        print(f"[{name}] decision tree exported to {dot_path} "
              f"(render with: dot -Tpng {dot_path} -o decision_tree.png)")

    # feature importance for tree-based models
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
        importances.sort_values(ascending=False).to_csv(
            os.path.join(outdir, f"{name}_feature_importance.csv")
        )

    return metrics


def compute_benchmark_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    TEMPLATE — compute BDSI, BDRM, ANU-ADRI scores from local questionnaire items,
    per PROTOCOL.md Section 4.8 benchmark comparison.

    IMPORTANT: item weights/cutoffs below are placeholders. Before use, verify
    against the original publications:
      - BDSI: Barnes et al. (2014), Alzheimer's & Dementia, 10(6), 656-665.
      - BDRM: Licher et al. (2019), American Journal of Psychiatry, 176(7), 543-551.
      - ANU-ADRI: Anstey et al. (2014), PLoS ONE, 9(1), e86141.
    """
    scores = pd.DataFrame(index=df.index)
    # Placeholder scoring logic -- replace with validated item weights.
    scores["bdsi_score"] = np.nan
    scores["bdrm_score"] = np.nan
    scores["anu_adri_score"] = np.nan
    print("NOTE: compute_benchmark_scores() is a placeholder. Populate with "
          "validated item weights before use (see docstring).")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed_data.csv")
    parser.add_argument("--outcome", default="cognitive_impairment")
    parser.add_argument("--outdir", default="results/aim1")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    y = df[args.outcome]
    X = df.drop(columns=[args.outcome]).select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
          f"Train prevalence: {y_train.mean():.2%}")

    all_metrics = {}
    for name, model in MODELS.items():
        all_metrics[name] = fit_and_evaluate(
            name, model, PARAM_GRIDS[name], X_train, y_train, X_test, y_test, args.outdir
        )

    with open(os.path.join(args.outdir, "aim1_model_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nSaved all metrics to {args.outdir}/aim1_model_metrics.json")


if __name__ == "__main__":
    main()
