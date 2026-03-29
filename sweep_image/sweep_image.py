"""
Classifier Sweep — Grid search over models, classifiers, and feature selection.
================================================================================
Runs 5-fold stratified CV for every combination and appends results to a CSV.
Skips combinations already present in the CSV.

All pipelines apply StandardScaler, then feature selection: PCA, ANOVA (SelectKBest), or ANOVA then PCA.

Usage:
    python sweep_image.py
    python sweep_image.py --models ResNet50 InceptionV3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Add parent directory (python/) to sys.path so `from pipeline import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import RANDOM_STATE, compute_metrics, load_or_extract, run_cv

BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "sweep_results.csv"

# ──────────────────────────────────────────────────────────────
# Feature selection configurations: (label, fs_mode, anova_k, n_pca)
# ──────────────────────────────────────────────────────────────
FS_CONFIGS = []

# PCA only
for n_pca in [5, 10, 15, 20, 30, 50]:
    FS_CONFIGS.append((f"PCA{n_pca}", "pca", None, n_pca))

# ANOVA only
for k in [10, 20, 50, 100, 200]:
    FS_CONFIGS.append((f"ANOVA{k}", "anova", k, None))

# ANOVA then PCA
for k in [50, 100, 200]:
    for n_pca in [5, 10, 15, 20, 30]:
        FS_CONFIGS.append((f"ANOVA{k}_PCA{n_pca}", "anova_pca", k, n_pca))

# ──────────────────────────────────────────────────────────────
# Classifier grid
# ──────────────────────────────────────────────────────────────
CLASSIFIERS = {
    # --- SVM RBF ---
    "SVM_C0.01": lambda: SVC(kernel="rbf", C=0.01, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C0.05": lambda: SVC(kernel="rbf", C=0.05, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C0.1":  lambda: SVC(kernel="rbf", C=0.1, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C0.5":  lambda: SVC(kernel="rbf", C=0.5, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C1":    lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C10":   lambda: SVC(kernel="rbf", C=10.0, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    # --- SVM linear / poly ---
    "SVM_linear_C0.1": lambda: SVC(kernel="linear", C=0.1,
                                    class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_linear_C1":   lambda: SVC(kernel="linear", C=1.0,
                                    class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_poly2_C0.1":  lambda: SVC(kernel="poly", degree=2, C=0.1, gamma="scale",
                                    class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_poly3_C0.1":  lambda: SVC(kernel="poly", degree=3, C=0.1, gamma="scale",
                                    class_weight="balanced", random_state=RANDOM_STATE),
    # --- Calibrated SVM (probabilistic) ---
    "SVM_C0.1_calibrated": lambda: CalibratedClassifierCV(
        SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced",
            random_state=RANDOM_STATE),
        cv=3, method="sigmoid", n_jobs=7),
    # --- Bagging SVM ---
    "BagSVM_C0.1": lambda: BaggingClassifier(
        estimator=SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced",
                      random_state=RANDOM_STATE),
        n_estimators=10, random_state=RANDOM_STATE, n_jobs=7),
    # --- Logistic Regression ---
    "LogReg_C0.1": lambda: LogisticRegression(C=0.1, max_iter=5000,
                                               class_weight="balanced", random_state=RANDOM_STATE,
                                               n_jobs=7),
    "LogReg_C1":   lambda: LogisticRegression(C=1.0, max_iter=5000,
                                               class_weight="balanced", random_state=RANDOM_STATE,
                                               n_jobs=7),
    "LogReg_C10":  lambda: LogisticRegression(C=10.0, max_iter=5000,
                                               class_weight="balanced", random_state=RANDOM_STATE,
                                               n_jobs=7),
    "LogReg_l1_C0.1": lambda: LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                                   max_iter=5000, class_weight="balanced",
                                                   random_state=RANDOM_STATE, n_jobs=7),
    "LogReg_l1_C1":   lambda: LogisticRegression(penalty="l1", solver="saga", C=1.0,
                                                   max_iter=5000, class_weight="balanced",
                                                   random_state=RANDOM_STATE, n_jobs=7),
    "LogReg_elasticnet_C0.1": lambda: LogisticRegression(penalty="elasticnet", solver="saga",
                                                          l1_ratio=0.5, C=0.1, max_iter=5000,
                                                          class_weight="balanced",
                                                          random_state=RANDOM_STATE, n_jobs=7),
    # --- Naive Bayes ---
    "GaussianNB": lambda: GaussianNB(),
    # --- Discriminant Analysis ---
    "LDA":     lambda: LinearDiscriminantAnalysis(),
    "QDA":     lambda: QuadraticDiscriminantAnalysis(),
    # --- Ridge ---
    "Ridge_0.1": lambda: RidgeClassifier(alpha=0.1, class_weight="balanced"),
    "Ridge":     lambda: RidgeClassifier(alpha=1.0, class_weight="balanced",
                                          random_state=RANDOM_STATE),
    "Ridge_10":  lambda: RidgeClassifier(alpha=10.0, class_weight="balanced"),
    # --- SGD ---
    "SGD_hinge": lambda: SGDClassifier(loss="hinge", class_weight="balanced",
                                        max_iter=5000, random_state=RANDOM_STATE),
    "SGD_log":   lambda: SGDClassifier(loss="log_loss", class_weight="balanced",
                                        max_iter=5000, random_state=RANDOM_STATE),
    "SGD_modified_huber": lambda: SGDClassifier(loss="modified_huber", class_weight="balanced",
                                                 max_iter=5000, random_state=RANDOM_STATE),
    "SGD_perceptron": lambda: SGDClassifier(loss="perceptron", class_weight="balanced",
                                             max_iter=5000, random_state=RANDOM_STATE),
    # --- Passive Aggressive ---
    "PassiveAggressive": lambda: PassiveAggressiveClassifier(C=0.1, class_weight="balanced",
                                                              max_iter=5000,
                                                              random_state=RANDOM_STATE),
    # --- Ensemble ---
    "RF_100":  lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=7),
    "RF_500":  lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=7),
    "ExtraTrees_100": lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                                     random_state=RANDOM_STATE, n_jobs=7),
    "GBM_100": lambda: GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GBM_200": lambda: GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "AdaBoost_100": lambda: AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
    # --- MLP ---
    "MLP_32":         lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000,
                                             alpha=0.01, random_state=RANDOM_STATE),
    "MLP_64":         lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000,
                                             random_state=RANDOM_STATE),
    "MLP_128":        lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=3000,
                                             random_state=RANDOM_STATE),
    "MLP_64_reg":     lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000,
                                             alpha=0.1, random_state=RANDOM_STATE),
    "MLP_128_64":     lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000,
                                             random_state=RANDOM_STATE),
    "MLP_128_64_reg": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000,
                                             alpha=0.1, random_state=RANDOM_STATE),
    "MLP_32_16":      lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000,
                                             alpha=0.01, random_state=RANDOM_STATE),
    "MLP_64_tanh":    lambda: MLPClassifier(hidden_layer_sizes=(64,), activation="tanh",
                                             max_iter=3000, alpha=0.01, random_state=RANDOM_STATE),
    # --- Other ---
    "NearestCentroid": lambda: NearestCentroid(),
}

MODELS = ["ResNet50", "DenseNet121", "InceptionV3"]


def load_existing_results():
    """Load existing CSV or return empty DataFrame."""
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def is_done(existing, model, classifier, fs_config):
    """Check if this combination is already in the results."""
    if existing.empty:
        return False
    mask = (
        (existing["model"] == model)
        & (existing["classifier"] == classifier)
        & (existing["fs_config"] == fs_config)
    )
    return mask.any()


def compute_auc(y, decision_vals, proba):
    """Compute AUC from probabilities (preferred) or decision values (fallback)."""
    if proba is not None and not np.all(np.isnan(proba[:, 1])):
        return roc_auc_score(y, proba[:, 1])
    if decision_vals is not None and not np.all(np.isnan(decision_vals)):
        return roc_auc_score(y, decision_vals)
    return np.nan


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep classifiers, models, feature selection.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                        help="Models to sweep (default: all)")
    parser.add_argument("--classifiers", nargs="+", choices=list(CLASSIFIERS.keys()),
                        default=None, help="Classifiers to sweep (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    clf_names = args.classifiers or list(CLASSIFIERS.keys())
    combos = [
        (model, clf_name, fs_label, fs_mode, anova_k, n_pca)
        for model in args.models
        for clf_name in clf_names
        for fs_label, fs_mode, anova_k, n_pca in FS_CONFIGS
    ]

    existing = load_existing_results()
    todo = [(m, c, fl, fm, ak, np_)
            for m, c, fl, fm, ak, np_ in combos
            if not is_done(existing, m, c, fl)]
    print(f"Total combinations: {len(combos)}, already done: {len(combos) - len(todo)}, "
          f"to run: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        raise SystemExit(0)

    # Pre-load features per model (cached)
    features = {}
    models_needed = sorted(set(m for m, *_ in todo))
    for model in models_needed:
        features[model] = load_or_extract(model)

    rows = []
    for i, (model, clf_name, fs_label, fs_mode, anova_k, n_pca) in enumerate(todo, 1):
        X, y, filenames = features[model]
        clf = CLASSIFIERS[clf_name]()
        print(f"[{i}/{len(todo)}] {model} | {clf_name} | {fs_label} ... ", end="", flush=True)

        try:
            y_pred, decision_vals, proba = run_cv(X, y, n_pca, clf, fs_mode, anova_k)
            metrics = compute_metrics(y, y_pred)
            auc = compute_auc(y, decision_vals, proba)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        row = {
            "model": model,
            "classifier": clf_name,
            "fs_config": fs_label,
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "auc": auc,
            "TP": int(metrics["cm"][1, 1]),
            "TN": int(metrics["cm"][0, 0]),
            "FP": int(metrics["cm"][0, 1]),
            "FN": int(metrics["cm"][1, 0]),
        }
        rows.append(row)
        auc_str = f"auc={auc:.3f}" if not np.isnan(auc) else "auc=N/A"
        print(f"acc={metrics['accuracy']:.3f}  sens={metrics['sensitivity']:.3f}  "
              f"spec={metrics['specificity']:.3f}  f1={metrics['f1']:.3f}  {auc_str}")

        # Append after each run so progress is saved even if interrupted
        df_new = pd.DataFrame([row])
        df_new.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)

    # Reload full results and sort for readability
    df_all = pd.read_csv(RESULTS_CSV)
    df_all = df_all.sort_values(["accuracy", "f1", "auc", "model", "classifier", "fs_config"],
                                ascending=[False, False, False, True, True, True])
    df_all.to_csv(RESULTS_CSV, index=False, float_format="%.4f")

    print(f"\nDone. Results saved to {RESULTS_CSV}")
    print(f"\nTop 10:")
    print(df_all.head(10).to_string(index=False))
