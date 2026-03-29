"""
Early Fusion Sweep — CNN Features + Clinical Features
======================================================
Concatenates dimensionality-reduced CNN features with clinical tabular
features and trains classifiers on the joint feature space.

Grid: 2 CNN models × 12 classifiers × 9 feature selection configs = 216 combos.
Models and classifiers selected from top performers in image and clinical sweeps.

Per fold: variance filter CNN → [ANOVA] → [PCA] → concat clinical → StandardScaler → classify.
5-fold stratified CV, incremental CSV save.

Usage:
    conda run -n torch python early_fusion.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add parent directories to sys.path
_PARENT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PARENT)
sys.path.insert(0, str(Path(_PARENT) / "sweep_clinical"))

from pipeline import RANDOM_STATE, load_or_extract
from clinical_sweep import FEATURE_COLS, compute_metrics, load_clinical_data

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "early_fusion_results.csv"

# ──────────────────────────────────────────────────────────────
# CNN models (top 2 from image sweep)
# ──────────────────────────────────────────────────────────────
MODELS = ["ResNet50", "DenseNet121"]

# ──────────────────────────────────────────────────────────────
# Classifiers (12, from top performers in image + clinical sweeps)
# ──────────────────────────────────────────────────────────────
CLASSIFIERS = {
    # Discriminant analysis — top in both sweeps
    "LDA": lambda: LinearDiscriminantAnalysis(),
    # Logistic Regression — top clinical performers
    "LogReg_C0.1": lambda: LogisticRegression(
        C=0.1, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    "LogReg_C1": lambda: LogisticRegression(
        C=1.0, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    "LogReg_C10": lambda: LogisticRegression(
        C=10.0, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    # SVM RBF (calibrated for predict_proba) — top image performers
    "SVM_C0.1": lambda: CalibratedClassifierCV(
        SVC(kernel="rbf", C=0.1, gamma="scale",
            class_weight="balanced", random_state=RANDOM_STATE),
        cv=3, method="sigmoid"),
    "SVM_C1": lambda: CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", random_state=RANDOM_STATE),
        cv=3, method="sigmoid"),
    # SVM linear (calibrated) — top clinical performers
    "SVM_linear_C0.1": lambda: CalibratedClassifierCV(
        SVC(kernel="linear", C=0.1,
            class_weight="balanced", random_state=RANDOM_STATE),
        cv=3, method="sigmoid"),
    "SVM_linear_C1": lambda: CalibratedClassifierCV(
        SVC(kernel="linear", C=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
        cv=3, method="sigmoid"),
    # Ensembles — top image performers
    "GBM_100": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
    "GBM_200": lambda: GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE),
    "RF_100": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
    "AdaBoost_100": lambda: AdaBoostClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
}

# ──────────────────────────────────────────────────────────────
# Feature selection configs: (label, mode, anova_k, n_pca)
# mode: "pca", "anova", "anova_pca", "none"
# ──────────────────────────────────────────────────────────────
FS_CONFIGS = [
    # ANOVA → PCA (dominant in top image results)
    ("ANOVA100_PCA5", "anova_pca", 100, 5),
    ("ANOVA100_PCA10", "anova_pca", 100, 10),
    ("ANOVA200_PCA5", "anova_pca", 200, 5),
    ("ANOVA200_PCA10", "anova_pca", 200, 10),
    # ANOVA only (appeared in image top-5)
    ("ANOVA10", "anova", 10, None),
    ("ANOVA20", "anova", 20, None),
    # PCA only
    ("PCA5", "pca", None, 5),
    ("PCA10", "pca", None, 10),
]


# ──────────────────────────────────────────────────────────────
# Early fusion CV
# ──────────────────────────────────────────────────────────────
def run_early_fusion_cv(X_cnn, X_clin, y, classifier, fs_mode, anova_k, n_pca):
    """5-fold stratified CV with early fusion.

    Per fold: variance filter CNN → [ANOVA] → [PCA] → concat clinical → scale → classify.
    Returns y_pred, proba (N×2).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros(len(y), dtype=int)
    proba = np.full((len(y), 2), np.nan)

    for train_idx, test_idx in skf.split(X_cnn, y):
        # Variance filter on CNN features (fit on train)
        variances = np.var(X_cnn[train_idx], axis=0)
        keep = variances > 1e-10
        X_cnn_train = X_cnn[train_idx][:, keep]
        X_cnn_test = X_cnn[test_idx][:, keep]

        # ANOVA feature selection on CNN features
        if fs_mode in ("anova", "anova_pca"):
            k = min(anova_k, X_cnn_train.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_cnn_train = selector.fit_transform(X_cnn_train, y[train_idx])
            X_cnn_test = selector.transform(X_cnn_test)

        # PCA on CNN features
        if fs_mode in ("pca", "anova_pca"):
            n = min(n_pca, X_cnn_train.shape[1])
            pca = PCA(n_components=n, random_state=RANDOM_STATE)
            X_cnn_train = pca.fit_transform(X_cnn_train)
            X_cnn_test = pca.transform(X_cnn_test)

        # Concatenate with clinical features
        X_train = np.hstack([X_cnn_train, X_clin[train_idx]])
        X_test = np.hstack([X_cnn_test, X_clin[test_idx]])

        # Scale + classify
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ])
        pipe.fit(X_train, y[train_idx])
        y_pred[test_idx] = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            proba[test_idx] = pipe.predict_proba(X_test)

    return y_pred, proba


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_existing():
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def is_done(existing, model, clf_name, fs_label):
    if existing.empty:
        return False
    return ((existing["model"] == model) &
            (existing["classifier"] == clf_name) &
            (existing["fs_config"] == fs_label)).any()


def compute_auc(y, proba):
    if proba is not None and not np.any(np.isnan(proba)):
        return roc_auc_score(y, proba[:, 1])
    return np.nan


def generate_report(df_all):
    """Generate detailed markdown report."""
    lines = [
        "# Early Fusion Sweep Results",
        "",
        f"**Total combinations evaluated:** {len(df_all)}",
        f"**Models:** {', '.join(MODELS)}",
        f"**Classifiers:** {len(CLASSIFIERS)}",
        f"**Feature selection configs:** {len(FS_CONFIGS)}",
        "",
        "## Top 15 Overall",
        "",
    ]

    top15 = df_all.head(15)
    lines.append("| Rank | Model | Classifier | FS Config | Acc. | Sens. | Spec. | F1 | AUC |")
    lines.append("|------|-------|-----------|-----------|------|-------|-------|----|-----|")
    for i, (_, row) in enumerate(top15.iterrows(), 1):
        auc_str = f"{row['auc']:.3f}" if not np.isnan(row["auc"]) else "N/A"
        lines.append(f"| {i} | {row['model']} | {row['classifier']} | {row['fs_config']} | "
                     f"{row['accuracy']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | "
                     f"{row['f1']:.3f} | {auc_str} |")
    lines.append("")

    # Best per model
    lines.append("## Best Per Model")
    lines.append("")
    for model in MODELS:
        subset = df_all[df_all["model"] == model]
        if subset.empty:
            continue
        best = subset.iloc[0]
        auc_str = f"{best['auc']:.3f}" if not np.isnan(best["auc"]) else "N/A"
        lines.append(f"- **{model}:** {best['classifier']} + {best['fs_config']} → "
                     f"acc={best['accuracy']:.3f}, sens={best['sensitivity']:.3f}, "
                     f"spec={best['specificity']:.3f}, F1={best['f1']:.3f}, AUC={auc_str}")
    lines.append("")

    # Best per classifier
    lines.append("## Best Per Classifier")
    lines.append("")
    for clf_name in CLASSIFIERS:
        subset = df_all[df_all["classifier"] == clf_name]
        if subset.empty:
            continue
        best = subset.iloc[0]
        auc_str = f"{best['auc']:.3f}" if not np.isnan(best["auc"]) else "N/A"
        lines.append(f"- **{clf_name}:** {best['model']} + {best['fs_config']} → "
                     f"acc={best['accuracy']:.3f}, F1={best['f1']:.3f}, AUC={auc_str}")
    lines.append("")

    # Best per FS config
    lines.append("## Best Per Feature Selection Config")
    lines.append("")
    for fs_label, _, _, _ in FS_CONFIGS:
        subset = df_all[df_all["fs_config"] == fs_label]
        if subset.empty:
            continue
        best = subset.iloc[0]
        auc_str = f"{best['auc']:.3f}" if not np.isnan(best["auc"]) else "N/A"
        lines.append(f"- **{fs_label}:** {best['model']} + {best['classifier']} → "
                     f"acc={best['accuracy']:.3f}, F1={best['f1']:.3f}, AUC={auc_str}")
    lines.append("")

    # Baselines comparison
    lines.append("## Comparison vs Baselines")
    lines.append("")
    lines.append("| Method | Accuracy | F1 | AUC |")
    lines.append("|--------|----------|----|-----|")
    if not df_all.empty:
        best = df_all.iloc[0]
        auc_str = f"{best['auc']:.3f}" if not np.isnan(best["auc"]) else "N/A"
        lines.append(f"| **Best early fusion** ({best['model']} + {best['classifier']} + "
                     f"{best['fs_config']}) | {best['accuracy']:.3f} | {best['f1']:.3f} | {auc_str} |")
    lines.append("| Best image-only (ResNet50+LDA) | 0.763 | 0.642 | 0.656 |")
    lines.append("| Best clinical-only (LDA) | 0.788 | 0.653 | 0.753 |")
    lines.append("| Best late fusion (Linear w=0.10) | 0.800 | 0.652 | 0.767 |")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load clinical features
    X_clin, y_clin, filenames_clin = load_clinical_data()
    clin_lookup = {fn: i for i, fn in enumerate(filenames_clin)}

    existing = load_existing()

    # Build all combos
    combos = [
        (model, clf_name, fs_label, fs_mode, anova_k, n_pca)
        for model in MODELS
        for clf_name in CLASSIFIERS
        for fs_label, fs_mode, anova_k, n_pca in FS_CONFIGS
    ]

    todo = [(m, c, fl, fm, ak, np_)
            for m, c, fl, fm, ak, np_ in combos
            if not is_done(existing, m, c, fl)]

    print(f"Early Fusion Sweep")
    print(f"  Models: {MODELS}")
    print(f"  Classifiers: {len(CLASSIFIERS)}")
    print(f"  FS configs: {len(FS_CONFIGS)}")
    print(f"  Total combinations: {len(combos)}")
    print(f"  Already done: {len(combos) - len(todo)}")
    print(f"  To run: {len(todo)}")

    if not todo:
        print("Nothing to do.")
    else:
        # Cache CNN features per model
        cnn_cache = {}
        needed_models = set(m for m, *_ in todo)

        for model_name in needed_models:
            print(f"\nLoading {model_name} features...")
            X_cnn, y_cnn, filenames_cnn = load_or_extract(model_name)

            # Align clinical data to CNN ordering
            clin_order = [clin_lookup[fn] for fn in filenames_cnn]
            X_clin_aligned = X_clin[clin_order]
            y_clin_aligned = y_clin[clin_order]
            assert np.array_equal(y_cnn, y_clin_aligned), \
                f"Label mismatch for {model_name}"

            cnn_cache[model_name] = (X_cnn, X_clin_aligned, y_cnn)
            print(f"  {model_name}: {X_cnn.shape[1]}-d CNN + {X_clin_aligned.shape[1]}-d clinical, "
                  f"{X_cnn.shape[0]} samples")

        for i, (model, clf_name, fs_label, fs_mode, anova_k, n_pca) in enumerate(todo, 1):
            X_cnn, X_clin_a, y = cnn_cache[model]
            clf = CLASSIFIERS[clf_name]()
            print(f"[{i}/{len(todo)}] {model} | {clf_name} | {fs_label} ... ",
                  end="", flush=True)

            try:
                y_pred, proba = run_early_fusion_cv(
                    X_cnn, X_clin_a, y, clf, fs_mode, anova_k, n_pca)
                m = compute_metrics(y, y_pred)
                auc = compute_auc(y, proba)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            row = {
                "model": model,
                "classifier": clf_name,
                "fs_config": fs_label,
                "accuracy": m["accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "auc": auc,
                "TP": int(m["cm"][1, 1]),
                "TN": int(m["cm"][0, 0]),
                "FP": int(m["cm"][0, 1]),
                "FN": int(m["cm"][1, 0]),
            }
            auc_str = f"auc={auc:.3f}" if not np.isnan(auc) else "auc=N/A"
            print(f"acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  "
                  f"spec={m['specificity']:.3f}  f1={m['f1']:.3f}  {auc_str}")

            df_new = pd.DataFrame([row])
            df_new.to_csv(RESULTS_CSV, mode="a",
                          header=not RESULTS_CSV.exists(), index=False)

    # Reload, sort, and save
    df_all = pd.read_csv(RESULTS_CSV)
    df_all = df_all.sort_values(["accuracy", "f1", "auc"],
                                ascending=[False, False, False])
    df_all.to_csv(RESULTS_CSV, index=False, float_format="%.4f")

    print(f"\nDone. Results saved to {RESULTS_CSV}")
    print(f"\nTop 15:")
    print(df_all.head(15).to_string(index=False))

    # Generate report
    report = generate_report(df_all)
    report_path = BASE_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")
