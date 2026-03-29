"""
Contingency Analysis: Image x Clinical Pipeline Combinations
=============================================================
Loads pre-calculated per-sample probabilities from best_image/ and
best_clinical/, then performs comparison statistics and combining
analysis for all 4 image x clinical combinations.

Combinations:
  1. Image#1 (ResNet50+LDA) x Clinical#1 (LDA)
  2. Image#1 (ResNet50+LDA) x Clinical#2 (LogReg C=0.1)
  3. Image#2 (DenseNet121+SGD) x Clinical#1 (LDA)
  4. Image#2 (DenseNet121+SGD) x Clinical#2 (LogReg C=0.1)

Usage:
    conda run -n torch python contingency_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
})

BASE_DIR = Path(__file__).resolve().parent
BEST_IMAGE_DIR = BASE_DIR.parent / "best_image"
BEST_CLINICAL_DIR = BASE_DIR.parent / "best_clinical"
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Combo definitions: (image_csv, clinical_csv, image_label, clinical_label, combo_tag)
COMBOS = [
    {
        "image_csv": BEST_IMAGE_DIR / "pipeline1_proba.csv",
        "clinical_csv": BEST_CLINICAL_DIR / "clinical1_lda_proba.csv",
        "image_label": "Image#1 (ResNet50+LDA)",
        "clinical_label": "Clinical#1 (LDA)",
        "tag": "combo1",
    },
    {
        "image_csv": BEST_IMAGE_DIR / "pipeline1_proba.csv",
        "clinical_csv": BEST_CLINICAL_DIR / "clinical2_logreg_proba.csv",
        "image_label": "Image#1 (ResNet50+LDA)",
        "clinical_label": "Clinical#2 (LogReg)",
        "tag": "combo2",
    },
    {
        "image_csv": BEST_IMAGE_DIR / "pipeline2_proba.csv",
        "clinical_csv": BEST_CLINICAL_DIR / "clinical1_lda_proba.csv",
        "image_label": "Image#2 (DenseNet121+SGD)",
        "clinical_label": "Clinical#1 (LDA)",
        "tag": "combo3",
    },
    {
        "image_csv": BEST_IMAGE_DIR / "pipeline2_proba.csv",
        "clinical_csv": BEST_CLINICAL_DIR / "clinical2_logreg_proba.csv",
        "image_label": "Image#2 (DenseNet121+SGD)",
        "clinical_label": "Clinical#2 (LogReg)",
        "tag": "combo4",
    },
]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def compute_metrics(y, y_pred):
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = f1_score(y, y_pred)
    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "f1": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def load_combo_data(combo):
    df_img = pd.read_csv(combo["image_csv"])
    df_clin = pd.read_csv(combo["clinical_csv"])

    df = df_img[["filename", "true_label", "proba_preterm"]].rename(
        columns={"proba_preterm": "p_preterm_image"}
    )
    df = df.merge(
        df_clin[["filename", "proba_preterm"]].rename(
            columns={"proba_preterm": "p_preterm_clinical"}
        ),
        on="filename",
    )
    df["pred_image"] = (df["p_preterm_image"] >= 0.5).astype(int)
    df["pred_clinical"] = (df["p_preterm_clinical"] >= 0.5).astype(int)
    return df


# ──────────────────────────────────────────────────────────────
# Part 1: Comparison statistics
# ──────────────────────────────────────────────────────────────
def contingency_table(pred_a, pred_b):
    n11 = int(np.sum((pred_a == 1) & (pred_b == 1)))
    n10 = int(np.sum((pred_a == 1) & (pred_b == 0)))
    n01 = int(np.sum((pred_a == 0) & (pred_b == 1)))
    n00 = int(np.sum((pred_a == 0) & (pred_b == 0)))
    return n11, n10, n01, n00


def disagreement_measure(pred_a, pred_b):
    return float(np.mean(pred_a != pred_b))


def mcnemar_test(pred_a, pred_b, y_true):
    err_a = (pred_a != y_true)
    err_b = (pred_b != y_true)
    b = int(np.sum(err_a & ~err_b))  # A wrong, B right
    c = int(np.sum(~err_a & err_b))  # A right, B wrong
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return chi2, p_value, b, c


def yule_q(n11, n10, n01, n00):
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    if den == 0:
        return 0.0
    return num / den


def kappa_interpretation(kappa):
    if kappa < 0.20:
        return "slight"
    elif kappa < 0.40:
        return "fair"
    elif kappa < 0.60:
        return "moderate"
    elif kappa < 0.80:
        return "substantial"
    return "almost perfect"


def yule_q_interpretation(q):
    if abs(q) < 0.25:
        return "low dependence (classifiers are diverse)"
    elif abs(q) < 0.75:
        return "moderate dependence"
    return "high dependence (classifiers agree strongly)"


def run_comparison(df, combo):
    y_true = df["true_label"].values
    pred_image = df["pred_image"].values
    pred_clinical = df["pred_clinical"].values
    p_image = df["p_preterm_image"].values
    p_clinical = df["p_preterm_clinical"].values

    tag = combo["tag"]
    img_label = combo["image_label"]
    clin_label = combo["clinical_label"]

    # Individual performance
    m_img = compute_metrics(y_true, pred_image)
    m_clin = compute_metrics(y_true, pred_clinical)
    auc_img = roc_auc_score(y_true, p_image)
    auc_clin = roc_auc_score(y_true, p_clinical)

    # Contingency table
    n11, n10, n01, n00 = contingency_table(pred_image, pred_clinical)

    # Disagreement
    disagree = disagreement_measure(pred_image, pred_clinical)

    # McNemar's test
    chi2, p_val, b, c = mcnemar_test(pred_image, pred_clinical, y_true)

    # Cohen's Kappa
    kappa = cohen_kappa_score(pred_image, pred_clinical)

    # Yule's Q
    q = yule_q(n11, n10, n01, n00)

    return {
        "tag": tag,
        "image_label": img_label,
        "clinical_label": clin_label,
        "m_img": m_img, "m_clin": m_clin,
        "auc_img": auc_img, "auc_clin": auc_clin,
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "disagreement": disagree,
        "mcnemar_chi2": chi2, "mcnemar_pval": p_val,
        "mcnemar_b": b, "mcnemar_c": c,
        "kappa": kappa, "kappa_interp": kappa_interpretation(kappa),
        "yule_q": q, "yule_q_interp": yule_q_interpretation(q),
    }


# ──────────────────────────────────────────────────────────────
# Part 2: Combining methods
# ──────────────────────────────────────────────────────────────
def llr_combine(p_image, p_clinical, y_true, n_pos, n_neg):
    eps = 1e-10
    llr_image = np.log((p_image + eps) / (1 - p_image + eps))
    llr_clinical = np.log((p_clinical + eps) / (1 - p_clinical + eps))
    log_prior = np.log(n_pos / n_neg)
    llr_total = llr_image + llr_clinical + log_prior
    y_pred = (llr_total > 0).astype(int)
    return y_pred, llr_total


def product_rule(p_image, p_clinical, n_pos, n_neg):
    n_total = n_pos + n_neg
    prior_pos = n_pos / n_total
    prior_neg = n_neg / n_total
    eps = 1e-10

    num = (p_image + eps) * (p_clinical + eps) / (prior_pos + eps)
    den_neg = ((1 - p_image + eps) * (1 - p_clinical + eps)) / (prior_neg + eps)
    p_combined = num / (num + den_neg)
    y_pred = (p_combined >= 0.5).astype(int)
    return y_pred, p_combined


def majority_and_soft_vote(p_image, p_clinical, pred_image, pred_clinical):
    # Soft vote
    p_soft = (p_image + p_clinical) / 2.0
    y_pred_soft = (p_soft >= 0.5).astype(int)

    # Hard vote with confidence tiebreak
    y_pred_hard = np.zeros(len(pred_image), dtype=int)
    for i in range(len(pred_image)):
        if pred_image[i] == pred_clinical[i]:
            y_pred_hard[i] = pred_image[i]
        else:
            conf_image = abs(p_image[i] - 0.5)
            conf_clinical = abs(p_clinical[i] - 0.5)
            if conf_image >= conf_clinical:
                y_pred_hard[i] = pred_image[i]
            else:
                y_pred_hard[i] = pred_clinical[i]

    return y_pred_hard, y_pred_soft, p_soft


# ──────────────────────────────────────────────────────────────
# Run all combining methods for one combo
# ──────────────────────────────────────────────────────────────
def run_combining(df, combo):
    y_true = df["true_label"].values
    pred_image = df["pred_image"].values
    pred_clinical = df["pred_clinical"].values
    p_image = df["p_preterm_image"].values
    p_clinical = df["p_preterm_clinical"].values
    tag = combo["tag"]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    METRICS = ["accuracy", "sensitivity", "specificity", "f1"]

    # --- Individual baselines ---
    m_img = compute_metrics(y_true, pred_image)
    m_clin = compute_metrics(y_true, pred_clinical)
    auc_img = roc_auc_score(y_true, p_image)
    auc_clin = roc_auc_score(y_true, p_clinical)

    # --- LLR ---
    y_pred_llr, llr_vals = llr_combine(p_image, p_clinical, y_true, n_pos, n_neg)
    m_llr = compute_metrics(y_true, y_pred_llr)
    llr_score = 1 / (1 + np.exp(-llr_vals))
    auc_llr = roc_auc_score(y_true, llr_score)

    # --- Product rule ---
    y_pred_prod, p_prod = product_rule(p_image, p_clinical, n_pos, n_neg)
    m_prod = compute_metrics(y_true, y_pred_prod)
    auc_prod = roc_auc_score(y_true, p_prod)

    # --- Majority / Soft vote ---
    y_pred_hard, y_pred_soft, p_soft = majority_and_soft_vote(
        p_image, p_clinical, pred_image, pred_clinical)
    m_hard = compute_metrics(y_true, y_pred_hard)
    m_soft = compute_metrics(y_true, y_pred_soft)
    auc_soft = roc_auc_score(y_true, p_soft)

    # --- Summary rows ---
    summary_rows = [
        {"method": "Image only", **{k: m_img[k] for k in METRICS}, "auc": auc_img},
        {"method": "Clinical only", **{k: m_clin[k] for k in METRICS}, "auc": auc_clin},
        {"method": "LLR", **{k: m_llr[k] for k in METRICS}, "auc": auc_llr},
        {"method": "Product rule", **{k: m_prod[k] for k in METRICS}, "auc": auc_prod},
        {"method": "Majority vote", **{k: m_hard[k] for k in METRICS}, "auc": np.nan},
        {"method": "Soft vote", **{k: m_soft[k] for k in METRICS}, "auc": auc_soft},
    ]

    # --- Per-sample combined predictions ---
    df_combined = df[["filename", "true_label"]].copy()
    df_combined["p_preterm_image"] = p_image
    df_combined["p_preterm_clinical"] = p_clinical
    df_combined["pred_image"] = pred_image
    df_combined["pred_clinical"] = pred_clinical
    df_combined["llr_value"] = llr_vals
    df_combined["pred_llr"] = y_pred_llr
    df_combined["p_product"] = p_prod
    df_combined["pred_product"] = y_pred_prod
    df_combined["pred_majority"] = y_pred_hard
    df_combined["p_soft"] = p_soft
    df_combined["pred_soft"] = y_pred_soft

    return {
        "summary_rows": summary_rows,
        "df_combined": df_combined,
    }


# ──────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────
def generate_report(all_comparisons, all_combining, path):
    lines = [
        "# Contingency Analysis Report",
        "",
        "Comparison and combining analysis for 4 Image x Clinical pipeline combinations.",
        "",
        "| Combo | Image Pipeline | Clinical Pipeline |",
        "|-------|---------------|-------------------|",
        "| 1 | Image#1 (ResNet50+LDA) | Clinical#1 (LDA) |",
        "| 2 | Image#1 (ResNet50+LDA) | Clinical#2 (LogReg C=0.1) |",
        "| 3 | Image#2 (DenseNet121+SGD) | Clinical#1 (LDA) |",
        "| 4 | Image#2 (DenseNet121+SGD) | Clinical#2 (LogReg C=0.1) |",
        "",
    ]

    for comp, comb in zip(all_comparisons, all_combining):
        tag = comp["tag"]
        img_l = comp["image_label"]
        clin_l = comp["clinical_label"]
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {tag.upper()}: {img_l} x {clin_l}")
        lines.append(f"")

        # Individual performance
        m_img = comp["m_img"]
        m_clin = comp["m_clin"]
        lines.append(f"### Individual Performance")
        lines.append(f"")
        lines.append(f"| Pipeline | Accuracy | Sensitivity | Specificity | F1 | AUC |")
        lines.append(f"|----------|----------|-------------|-------------|-----|-----|")
        lines.append(f"| {img_l} | {m_img['accuracy']:.3f} | {m_img['sensitivity']:.3f} | "
                     f"{m_img['specificity']:.3f} | {m_img['f1']:.3f} | {comp['auc_img']:.3f} |")
        lines.append(f"| {clin_l} | {m_clin['accuracy']:.3f} | {m_clin['sensitivity']:.3f} | "
                     f"{m_clin['specificity']:.3f} | {m_clin['f1']:.3f} | {comp['auc_clin']:.3f} |")
        lines.append(f"")

        # Contingency table
        lines.append(f"### Contingency Table")
        lines.append(f"")
        lines.append(f"|  | Clinical=Preterm | Clinical=Term |")
        lines.append(f"|--|-----------------|---------------|")
        lines.append(f"| Image=Preterm | {comp['n11']} | {comp['n10']} |")
        lines.append(f"| Image=Term | {comp['n01']} | {comp['n00']} |")
        lines.append(f"")

        # Statistics
        lines.append(f"### Comparison Statistics")
        lines.append(f"")
        lines.append(f"| Statistic | Value | Interpretation |")
        lines.append(f"|-----------|-------|----------------|")
        lines.append(f"| Disagreement | {comp['disagreement']:.4f} "
                     f"({int(comp['disagreement'] * 80)}/80 samples) | |")
        lines.append(f"| McNemar b (img wrong, clin right) | {comp['mcnemar_b']} | |")
        lines.append(f"| McNemar c (img right, clin wrong) | {comp['mcnemar_c']} | |")
        lines.append(f"| McNemar chi2 | {comp['mcnemar_chi2']:.4f} | "
                     f"{'Significant' if comp['mcnemar_pval'] < 0.05 else 'Not significant'} "
                     f"(p={comp['mcnemar_pval']:.4f}) |")
        lines.append(f"| Cohen's Kappa | {comp['kappa']:.4f} | {comp['kappa_interp']} agreement |")
        lines.append(f"| Yule's Q | {comp['yule_q']:.4f} | {comp['yule_q_interp']} |")
        lines.append(f"")

        # Combining summary
        lines.append(f"### Combining Results")
        lines.append(f"")
        lines.append(f"| Method | Accuracy | Sensitivity | Specificity | F1 | AUC |")
        lines.append(f"|--------|----------|-------------|-------------|-----|-----|")
        for row in comb["summary_rows"]:
            auc_str = f"{row['auc']:.3f}" if not np.isnan(row["auc"]) else "N/A"
            lines.append(f"| {row['method']} | {row['accuracy']:.3f} | "
                         f"{row['sensitivity']:.3f} | {row['specificity']:.3f} | "
                         f"{row['f1']:.3f} | {auc_str} |")
        lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    all_comparisons = []
    all_combining = []
    all_comparison_stats = []
    all_combining_results = []

    for combo in COMBOS:
        tag = combo["tag"]
        print(f"\n{'=' * 60}")
        print(f"  {tag.upper()}: {combo['image_label']} x {combo['clinical_label']}")
        print(f"{'=' * 60}")

        # Load data
        df = load_combo_data(combo)
        print(f"  Loaded {len(df)} samples")

        # --- Comparison ---
        comp = run_comparison(df, combo)
        all_comparisons.append(comp)

        print(f"\n  Contingency Table:")
        print(f"    N11={comp['n11']}  N10={comp['n10']}  N01={comp['n01']}  N00={comp['n00']}")
        print(f"  Disagreement: {comp['disagreement']:.4f}")
        print(f"  McNemar: chi2={comp['mcnemar_chi2']:.4f}, p={comp['mcnemar_pval']:.4f}")
        print(f"  Kappa: {comp['kappa']:.4f} ({comp['kappa_interp']})")
        print(f"  Yule's Q: {comp['yule_q']:.4f} ({comp['yule_q_interp']})")

        # Save comparison stats
        all_comparison_stats.append({
            "combo": tag,
            "image": combo["image_label"],
            "clinical": combo["clinical_label"],
            "N11_both_preterm": comp["n11"],
            "N10_img_pre_clin_term": comp["n10"],
            "N01_img_term_clin_pre": comp["n01"],
            "N00_both_term": comp["n00"],
            "disagreement": comp["disagreement"],
            "mcnemar_b": comp["mcnemar_b"],
            "mcnemar_c": comp["mcnemar_c"],
            "mcnemar_chi2": comp["mcnemar_chi2"],
            "mcnemar_pval": comp["mcnemar_pval"],
            "cohens_kappa": comp["kappa"],
            "kappa_interp": comp["kappa_interp"],
            "yules_q": comp["yule_q"],
            "yules_q_interp": comp["yule_q_interp"],
        })

        # --- Combining ---
        comb = run_combining(df, combo)
        all_combining.append(comb)

        print(f"\n  Combining Summary:")
        for row in comb["summary_rows"]:
            auc_str = f"auc={row['auc']:.3f}" if not np.isnan(row["auc"]) else "auc=N/A"
            print(f"    {row['method']:<22s}  acc={row['accuracy']:.3f}  "
                  f"sens={row['sensitivity']:.3f}  spec={row['specificity']:.3f}  "
                  f"f1={row['f1']:.3f}  {auc_str}")

        # Save combining summary
        for row in comb["summary_rows"]:
            all_combining_results.append({"combo": tag, **row})

        # Save per-sample CSV
        comb["df_combined"].to_csv(
            TABLE_DIR / f"per_sample_{tag}.csv",
            index=False, float_format="%.6f")
        print(f"  Saved per_sample_{tag}.csv")

    # --- Save global CSVs ---
    pd.DataFrame(all_comparison_stats).to_csv(
        TABLE_DIR / "comparison_stats.csv", index=False, float_format="%.6f")
    print(f"\nSaved comparison_stats.csv")

    pd.DataFrame(all_combining_results).to_csv(
        TABLE_DIR / "combining_results.csv", index=False, float_format="%.4f")
    print(f"Saved combining_results.csv")

    # --- Generate report ---
    generate_report(all_comparisons, all_combining, OUTPUT_DIR / "report.md")
    print(f"Saved report.md")

    print(f"\n=== Done ===")
