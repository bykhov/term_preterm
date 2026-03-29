"""
Clinical Data Classifier Sweep
===============================
Grid search over classifiers using 10 clinical tabular features (80 samples)
from clinical_data.csv.

Pipeline: StandardScaler → Classifier (no PCA, no variance filter).
5-fold stratified CV, incremental CSV save.

Usage:
    python clinical_sweep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR.parent / "Data" / "clinical_data.csv"
RESULTS_CSV = BASE_DIR / "clinical_sweep_results.csv"

FEATURE_COLS = [
    "Age", "Pregnancy number", "Past births", "Previous preterm births",
    "Previous cesarean section", "Cervical length (cm)", "Smoker",
    "Pre-pregnancy diseases", "Pregnancy diseases", "Treatment to stop labor",
]

# ──────────────────────────────────────────────────────────────
# Classifier grid
# ──────────────────────────────────────────────────────────────
CLASSIFIERS = {
    # --- SVM RBF ---
    "SVM_C0.1": lambda: SVC(kernel="rbf", C=0.1, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C0.5": lambda: SVC(kernel="rbf", C=0.5, gamma="scale",
                              class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C1": lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                            class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_C10": lambda: SVC(kernel="rbf", C=10.0, gamma="scale",
                             class_weight="balanced", random_state=RANDOM_STATE),
    # --- SVM linear / poly ---
    "SVM_linear_C0.1": lambda: SVC(kernel="linear", C=0.1,
                                    class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_linear_C1": lambda: SVC(kernel="linear", C=1.0,
                                  class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_poly2_C0.1": lambda: SVC(kernel="poly", degree=2, C=0.1, gamma="scale",
                                    class_weight="balanced", random_state=RANDOM_STATE),
    "SVM_poly3_C0.1": lambda: SVC(kernel="poly", degree=3, C=0.1, gamma="scale",
                                    class_weight="balanced", random_state=RANDOM_STATE),
    # --- Logistic Regression ---
    "LogReg_C0.1": lambda: LogisticRegression(C=0.1, max_iter=5000,
                                               class_weight="balanced",
                                               random_state=RANDOM_STATE),
    "LogReg_C1": lambda: LogisticRegression(C=1.0, max_iter=5000,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE),
    "LogReg_C10": lambda: LogisticRegression(C=10.0, max_iter=5000,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE),
    "LogReg_l1_C0.1": lambda: LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                                   max_iter=5000, class_weight="balanced",
                                                   random_state=RANDOM_STATE),
    "LogReg_l1_C1": lambda: LogisticRegression(penalty="l1", solver="saga", C=1.0,
                                                 max_iter=5000, class_weight="balanced",
                                                 random_state=RANDOM_STATE),
    "LogReg_elasticnet_C0.1": lambda: LogisticRegression(penalty="elasticnet", solver="saga",
                                                          l1_ratio=0.5, C=0.1, max_iter=5000,
                                                          class_weight="balanced",
                                                          random_state=RANDOM_STATE),
    # --- Naive Bayes ---
    "GaussianNB": lambda: GaussianNB(),
    # --- Discriminant Analysis ---
    "LDA": lambda: LinearDiscriminantAnalysis(),
    "QDA": lambda: QuadraticDiscriminantAnalysis(),
    # --- Ridge ---
    "Ridge_0.1": lambda: RidgeClassifier(alpha=0.1, class_weight="balanced"),
    "Ridge": lambda: RidgeClassifier(alpha=1.0, class_weight="balanced",
                                      random_state=RANDOM_STATE),
    "Ridge_10": lambda: RidgeClassifier(alpha=10.0, class_weight="balanced"),
    # --- SGD ---
    "SGD_hinge": lambda: SGDClassifier(loss="hinge", class_weight="balanced",
                                        max_iter=5000, random_state=RANDOM_STATE),
    "SGD_log": lambda: SGDClassifier(loss="log_loss", class_weight="balanced",
                                      max_iter=5000, random_state=RANDOM_STATE),
    "SGD_modified_huber": lambda: SGDClassifier(loss="modified_huber",
                                                 class_weight="balanced",
                                                 max_iter=5000, random_state=RANDOM_STATE),
    "SGD_perceptron": lambda: SGDClassifier(loss="perceptron", class_weight="balanced",
                                             max_iter=5000, random_state=RANDOM_STATE),
    # --- Passive Aggressive ---
    "PassiveAggressive": lambda: PassiveAggressiveClassifier(C=0.1, class_weight="balanced",
                                                              max_iter=5000,
                                                              random_state=RANDOM_STATE),
    # --- Tree-based / Ensemble ---
    "DecisionTree": lambda: DecisionTreeClassifier(class_weight="balanced",
                                                    random_state=RANDOM_STATE),
    "RF_100": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                              random_state=RANDOM_STATE),
    "RF_500": lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                              random_state=RANDOM_STATE),
    "ExtraTrees_100": lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                                     random_state=RANDOM_STATE),
    "GBM_100": lambda: GradientBoostingClassifier(n_estimators=100,
                                                    random_state=RANDOM_STATE),
    "GBM_200": lambda: GradientBoostingClassifier(n_estimators=200,
                                                    random_state=RANDOM_STATE),
    "AdaBoost_100": lambda: AdaBoostClassifier(n_estimators=100,
                                                random_state=RANDOM_STATE),
    # --- KNN ---
    "KNN_3": lambda: KNeighborsClassifier(n_neighbors=3),
    "KNN_5": lambda: KNeighborsClassifier(n_neighbors=5),
    "KNN_7": lambda: KNeighborsClassifier(n_neighbors=7),
    # --- Other ---
    "NearestCentroid": lambda: NearestCentroid(),
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_clinical_data():
    """Load clinical_data.csv and return X, y, filenames."""
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values.astype(int)
    filenames = df["filename"].tolist()
    return X, y, filenames


def build_pipeline(classifier):
    """Build sklearn Pipeline: StandardScaler → classifier."""
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


def run_cv(X, y, classifier):
    """5-fold stratified CV.

    Returns y_pred, decision_vals, proba.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros(len(y), dtype=int)
    decision_vals = np.full(len(y), np.nan)
    proba = np.full((len(y), 2), np.nan)
    has_proba = False

    for train_idx, test_idx in skf.split(X, y):
        pipe = build_pipeline(classifier)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X[test_idx])

        X_test_transformed = pipe[:-1].transform(X[test_idx])
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "decision_function"):
            decision_vals[test_idx] = clf.decision_function(X_test_transformed)
        if hasattr(clf, "predict_proba"):
            has_proba = True
            proba[test_idx] = clf.predict_proba(X_test_transformed)

    if not has_proba:
        proba = None

    return y_pred, decision_vals, proba


def compute_metrics(y, y_pred):
    """Compute classification metrics."""
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = f1_score(y, y_pred)
    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "f1": f1, "cm": cm,
    }


def compute_auc(y, decision_vals, proba):
    """Compute AUC from probabilities (preferred) or decision values (fallback)."""
    if proba is not None and not np.all(np.isnan(proba[:, 1])):
        return roc_auc_score(y, proba[:, 1])
    if decision_vals is not None and not np.all(np.isnan(decision_vals)):
        return roc_auc_score(y, decision_vals)
    return np.nan


def load_existing_results():
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def is_done(existing, classifier):
    if existing.empty:
        return False
    return (existing["classifier"] == classifier).any()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, filenames = load_clinical_data()
    print(f"Clinical data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Term: {np.sum(y == 0)}, Preterm: {np.sum(y == 1)}")

    clf_names = list(CLASSIFIERS.keys())
    existing = load_existing_results()
    todo = [c for c in clf_names if not is_done(existing, c)]
    print(f"Total classifiers: {len(clf_names)}, already done: {len(clf_names) - len(todo)}, "
          f"to run: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        raise SystemExit(0)

    rows = []
    for i, clf_name in enumerate(todo, 1):
        clf = CLASSIFIERS[clf_name]()
        print(f"[{i}/{len(todo)}] {clf_name} ... ", end="", flush=True)

        try:
            y_pred, decision_vals, proba = run_cv(X, y, clf)
            metrics = compute_metrics(y, y_pred)
            auc = compute_auc(y, decision_vals, proba)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        row = {
            "classifier": clf_name,
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

        # Append after each run so progress is saved
        df_new = pd.DataFrame([row])
        df_new.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)

    # Reload and sort
    df_all = pd.read_csv(RESULTS_CSV)
    df_all = df_all.sort_values(["accuracy", "f1", "auc", "classifier"],
                                ascending=[False, False, False, True])
    df_all.to_csv(RESULTS_CSV, index=False, float_format="%.4f")

    print(f"\nDone. Results saved to {RESULTS_CSV}")
    print(f"\nTop 10:")
    print(df_all.head(10).to_string(index=False))
