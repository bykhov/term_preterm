"""
Pipeline #2: Best F1 & AUC — DenseNet121 + ANOVA100→PCA5 + SGD(modified Huber)
================================================================================
Runs 5-fold stratified CV, generates figures, tables, proba CSV, and report.

Usage:
    conda run -n torch python pipeline2_sgd.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import RANDOM_STATE, main

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
})

BASE_DIR = Path(__file__).resolve().parent
PREFIX = "pipeline2"
LABEL = "DenseNet121 → ANOVA100→PCA5 + SGD(modified_huber)"


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(metrics, label, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["a", "b"], yticklabels=["a", "b"],
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
    ax.legend(handles=legend_elements, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y, proba, label, path):
    from sklearn.metrics import roc_curve, auc
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
    from sklearn.calibration import calibration_curve
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

    results = main(
        model_name="DenseNet121",
        classifier=SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        n_pca=5,
        fs_mode="anova_pca",
        anova_k=100,
    )

    y = results["y"]
    filenames = results["filenames"]
    y_pred = results["y_pred"]
    decision_vals = results["decision_vals"]
    proba = results["proba"]
    metrics = results["metrics"]

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

    if proba is not None:
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
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }])
    table1.to_csv(BASE_DIR / f"{PREFIX}_classification.csv",
                   index=False, float_format="%.4f")
    print(f"  {PREFIX}_classification.csv")

    # --- Per-sample predictions ---
    table2_data = {
        "filename": filenames,
        "true_label": y,
        "true_class": ["a" if l == 0 else "b" for l in y],
        "predicted": y_pred,
        "pred_class": ["a" if l == 0 else "b" for l in y_pred],
        "correct": (y_pred == y).astype(int),
        "decision_value": decision_vals,
    }
    if proba is not None:
        table2_data["proba_term"] = proba[:, 0]
        table2_data["proba_preterm"] = proba[:, 1]
    table2 = pd.DataFrame(table2_data)
    table2.to_csv(BASE_DIR / f"{PREFIX}_per_sample.csv",
                   index=False, float_format="%.6f")
    print(f"  {PREFIX}_per_sample.csv")

    # --- Proba CSV for fusion ---
    if proba is not None:
        proba_df = pd.DataFrame({
            "filename": filenames,
            "true_label": y,
            "proba_term": proba[:, 0],
            "proba_preterm": proba[:, 1],
        })
        proba_df.to_csv(BASE_DIR / f"{PREFIX}_proba.csv",
                         index=False, float_format="%.6f")
        print(f"  {PREFIX}_proba.csv")

    # --- Report ---
    n_a = int(np.sum(y == 0))
    n_b = int(np.sum(y == 1))
    auc_str = f"{roc_auc:.3f}" if proba is not None else "N/A"

    report = f"""# Pipeline #2: {LABEL}

**Dataset:** {len(y)} cropped ultrasound images (Term: {n_a}, Preterm: {n_b})

## Pipeline
DenseNet121 (1024-d) → Variance Filter → StandardScaler → ANOVA(k=100) → PCA(5) → SGD(modified_huber)

## Classifier Parameters
- loss: modified_huber (smooth hinge approximation, yields probability estimates)
- class_weight: balanced
- max_iter: 5000
- random_state: {RANDOM_STATE}

## 5-Fold CV Results

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.3f} |
| Sensitivity | {metrics['sensitivity']:.3f} |
| Specificity | {metrics['specificity']:.3f} |
| F1 | {metrics['f1']:.3f} |
| AUC | {auc_str} |
| TP | {tp} |
| TN | {tn} |
| FP | {fp} |
| FN | {fn} |

## Notes
- SGD with modified Huber loss provides probabilistic outputs
- Proba saved to {PREFIX}_proba.csv for late fusion analysis
"""

    (BASE_DIR / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    print(f"  {PREFIX}_report.md")

    print("\n=== Done ===")
