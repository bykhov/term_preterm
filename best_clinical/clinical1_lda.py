"""
Clinical Pipeline #1: Best Accuracy — StandardScaler + LDA
===========================================================
Runs 5-fold stratified CV on 10 clinical features, generates figures,
tables, proba CSV, and report.

Usage:
    conda run -n torch python clinical1_lda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import compute_fold_metrics

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
})

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR.parent.parent / "Data_meta" / "clinical_data.csv"
PREFIX = "clinical1_lda"
LABEL = "Clinical → StandardScaler + LDA"

FEATURE_COLS = [
    "Age", "Pregnancy number", "Past births", "Previous preterm births",
    "Previous cesarean section", "Cervical length (cm)", "Smoker",
    "Pre-pregnancy diseases", "Pregnancy diseases", "Treatment to stop labor",
]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_clinical_data():
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values.astype(int)
    filenames = df["filename"].tolist()
    return X, y, filenames


def compute_metrics(y, y_pred):
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


def run_cv(X, y):
    classifier = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros(len(y), dtype=int)
    decision_vals = np.full(len(y), np.nan)
    proba = np.full((len(y), 2), np.nan)
    fold_indices = []

    for train_idx, test_idx in skf.split(X, y):
        fold_indices.append((train_idx, test_idx))
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X[test_idx])

        X_test_scaled = pipe[:-1].transform(X[test_idx])
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "decision_function"):
            decision_vals[test_idx] = clf.decision_function(X_test_scaled)
        if hasattr(clf, "predict_proba"):
            proba[test_idx] = clf.predict_proba(X_test_scaled)

    return y_pred, decision_vals, proba, fold_indices


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(metrics, label, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Term", "Preterm"], yticklabels=["Term", "Preterm"],
                cbar=False, annot_kws={"size": 14})
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(
        f"{label}\n"
        f"Acc={metrics['accuracy']:.3f}  Sens={metrics['sensitivity']:.3f}  "
        f"Spec={metrics['specificity']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_decision_landscape(filenames, y, y_pred, decision_vals, label, path):
    correct = y_pred == y
    order = np.argsort(decision_vals)
    filenames = [filenames[i] for i in order]
    y, y_pred = y[order], y_pred[order]
    decision_vals, correct = decision_vals[order], correct[order]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i in range(len(y)):
        color = "#2196F3" if y[i] == 0 else "#F44336"
        marker = "o" if correct[i] else "X"
        edge = "k" if correct[i] else "#FFD600"
        lw = 0.5 if correct[i] else 2.0
        ax.scatter(i, decision_vals[i], c=color, marker=marker,
                   edgecolors=edge, linewidths=lw, s=60, zorder=3)

    ax.axhline(0, color="k", linestyle="--", alpha=0.4, label="Decision boundary")
    ax.set_xticks([])
    ax.set_ylabel("Decision Value", fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="Term (correct)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336",
               markersize=8, label="Preterm (correct)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#F44336",
               markeredgecolor="#FFD600", markersize=10, label="Misclassified"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y, proba, label, path):
    fpr, tpr, _ = roc_curve(y, proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(f"ROC Curve - {label}", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def plot_proba_histogram(y, proba, label, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.hist(proba[y == 0, 1], bins=15, alpha=0.6, color="#2196F3",
            edgecolor="k", linewidth=0.5, label="Term")
    ax.hist(proba[y == 1, 1], bins=15, alpha=0.6, color="#F44336",
            edgecolor="k", linewidth=0.5, label="Preterm")
    ax.axvline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("P(preterm)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"Probability Distribution - {label}", fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y, proba, label, path):
    prob_true, prob_pred = calibration_curve(y, proba[:, 1], n_bins=8,
                                             strategy="uniform")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(prob_pred, prob_true, "o-", color="#1976D2", linewidth=2,
            markersize=6, label="Classifier")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.set_title(f"Calibration - {label}", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running {LABEL}")

    X, y, filenames = load_clinical_data()
    print(f"Clinical data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Term: {np.sum(y == 0)}, Preterm: {np.sum(y == 1)}")

    y_pred, decision_vals, proba, fold_indices = run_cv(X, y)
    metrics = compute_metrics(y, y_pred)
    fold_result = compute_fold_metrics(y, y_pred, proba, fold_indices)
    ci = fold_result["ci"]

    print(f"\nResults:")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  F1:          {metrics['f1']:.3f}")

    # --- Figures ---
    print("\nGenerating figures...")
    plot_confusion_matrix(
        metrics, LABEL,
        BASE_DIR / f"{PREFIX}_confusion_matrix.pdf")
    print(f"  {PREFIX}_confusion_matrix.pdf")

    plot_decision_landscape(
        filenames, y.copy(), y_pred.copy(), decision_vals.copy(), LABEL,
        BASE_DIR / f"{PREFIX}_decision_landscape.pdf")
    print(f"  {PREFIX}_decision_landscape.pdf")

    roc_auc = plot_roc_curve(
        y, proba, LABEL,
        BASE_DIR / f"{PREFIX}_roc.pdf")
    print(f"  {PREFIX}_roc.pdf  (AUC={roc_auc:.3f})")

    plot_proba_histogram(
        y, proba, LABEL,
        BASE_DIR / f"{PREFIX}_proba_hist.pdf")
    print(f"  {PREFIX}_proba_hist.pdf")

    plot_calibration(
        y, proba, LABEL,
        BASE_DIR / f"{PREFIX}_calibration.pdf")
    print(f"  {PREFIX}_calibration.pdf")

    # --- Classification summary ---
    tn, fp, fn, tp = metrics["cm"].ravel()
    table1 = pd.DataFrame([{
        "pipeline": LABEL,
        "accuracy": metrics["accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "F1_score": metrics["f1"],
        "AUC": roc_auc,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy_ci_low": ci["accuracy"][1], "accuracy_ci_high": ci["accuracy"][2],
        "sensitivity_ci_low": ci["sensitivity"][1], "sensitivity_ci_high": ci["sensitivity"][2],
        "specificity_ci_low": ci["specificity"][1], "specificity_ci_high": ci["specificity"][2],
        "F1_score_ci_low": ci["f1"][1], "F1_score_ci_high": ci["f1"][2],
        "AUC_ci_low": ci["auc"][1], "AUC_ci_high": ci["auc"][2],
    }])
    table1.to_csv(BASE_DIR / f"{PREFIX}_classification.csv",
                   index=False, float_format="%.4f")
    print(f"  {PREFIX}_classification.csv")

    # --- Per-sample predictions ---
    table2 = pd.DataFrame({
        "filename": filenames,
        "true_label": y,
        "true_class": ["term" if l == 0 else "preterm" for l in y],
        "predicted": y_pred,
        "pred_class": ["term" if l == 0 else "preterm" for l in y_pred],
        "correct": (y_pred == y).astype(int),
        "decision_value": decision_vals,
        "proba_term": proba[:, 0],
        "proba_preterm": proba[:, 1],
    })
    table2.to_csv(BASE_DIR / f"{PREFIX}_per_sample.csv",
                   index=False, float_format="%.6f")
    print(f"  {PREFIX}_per_sample.csv")

    # --- Proba CSV for fusion ---
    fold_assignments = np.full(len(y), -1, dtype=int)
    for fold_num, (_, test_idx) in enumerate(fold_indices):
        fold_assignments[test_idx] = fold_num
    proba_df = pd.DataFrame({
        "filename": filenames,
        "true_label": y,
        "proba_term": proba[:, 0],
        "proba_preterm": proba[:, 1],
        "fold": fold_assignments,
    })
    proba_df.to_csv(BASE_DIR / f"{PREFIX}_proba.csv",
                     index=False, float_format="%.6f")
    print(f"  {PREFIX}_proba.csv")

    # --- Report ---
    n_term = int(np.sum(y == 0))
    n_preterm = int(np.sum(y == 1))

    report = f"""# Clinical Pipeline #1: {LABEL}

**Dataset:** {len(y)} patients ({n_term} term, {n_preterm} preterm)

## Pipeline
10 clinical features → StandardScaler → LDA

## Clinical Features
Age, Pregnancy number, Past births, Previous preterm births,
Previous cesarean section, Cervical length (cm), Smoker,
Pre-pregnancy diseases, Pregnancy diseases, Treatment to stop labor

## 5-Fold CV Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | {metrics['accuracy']:.3f} | ({ci['accuracy'][1]:.3f}–{ci['accuracy'][2]:.3f}) |
| Sensitivity | {metrics['sensitivity']:.3f} | ({ci['sensitivity'][1]:.3f}–{ci['sensitivity'][2]:.3f}) |
| Specificity | {metrics['specificity']:.3f} | ({ci['specificity'][1]:.3f}–{ci['specificity'][2]:.3f}) |
| F1 | {metrics['f1']:.3f} | ({ci['f1'][1]:.3f}–{ci['f1'][2]:.3f}) |
| AUC | {roc_auc:.3f} | ({ci['auc'][1]:.3f}–{ci['auc'][2]:.3f}) |
| TP | {tp} | |
| TN | {tn} | |
| FP | {fp} | |
| FN | {fn} | |

## Notes
- LDA provides posterior class probabilities via `predict_proba`
- High specificity ({metrics['specificity']:.3f}) — reliably identifies term births
- Moderate sensitivity ({metrics['sensitivity']:.3f}) — misses {fn} of {n_preterm} preterm cases
- Proba saved to {PREFIX}_proba.csv for late fusion analysis
"""

    (BASE_DIR / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    print(f"  {PREFIX}_report.md")

    print("\n=== Done ===")
