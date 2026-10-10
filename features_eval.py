"""
Selection-Free Evaluation of the Image-Feature Families and Late Fusion
======================================================================
Every image family goes through the same pipeline, fixed in advance:
    variance filter -> StandardScaler -> PCA(10) -> LogisticRegression(L2, C=0.1, balanced)
Families: hc, mid, clip (features_new.py) and the RadImageNet final-layer
features ref_densenet (DenseNet-121) and ref_resnet (ResNet-50).

Primary image stream (fixed in advance): img_ens = mean probability over
hc, mid, clip and ref_densenet. Fusion: soft vote with the fixed clinical
LDA (median-imputed); primary fusion = img_ens + LDA.

Nothing is selected from the data, so repeated stratified 5-fold CV over
20 seeds (42..61) gives unbiased estimates.

Outputs (results/):
  features_eval_per_seed.csv, features_eval_summary.csv,
  features_eval_paired.csv

Usage:
    python features_eval.py   (after features_new.py)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import (RANDOM_STATE, RESULTS_DIR, cv_splits, fit_predict_image, fit_predict_tabular,
                    fuse, lda, load_aligned, oof)
from features_new import load_family
from stats_utils import corrected_t, point_metrics

SEEDS = list(range(RANDOM_STATE, RANDOM_STATE + 20))
NEW = ["hc", "mid", "clip"]
ENS_MEMBERS = ["hc", "mid", "clip", "ref_densenet"]
PRIMARY_IMAGE, PRIMARY_FUSION = "img_ens", "fusion_img_ens"


def logreg():
    return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced",
                              random_state=RANDOM_STATE)


def image_oof(X, y, splits):
    return oof(splits, lambda tr, te: fit_predict_image(X[tr], y[tr], X[te], "pca", None, 10,
                                                        logreg), len(y))


def all_streams(F, X_clin, y, splits):
    """OOF probabilities of every image stream, the clinical model and the fusions."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    p = {fam: image_oof(X, y, splits) for fam, X in F.items()}
    p["img_ens"] = np.mean([p[f] for f in ENS_MEMBERS], axis=0)
    p["clinical_lda"] = oof(splits, lambda tr, te: fit_predict_tabular(
        X_clin[tr], y[tr], X_clin[te], lda), len(y))
    for s in list(F) + ["img_ens"]:
        p[f"fusion_{s}"] = fuse(p[s], p["clinical_lda"], "soft", n_pos, n_neg)
    return p


if __name__ == "__main__":
    X_res, X_clin, y, filenames = load_aligned("ResNet50")
    X_dense = load_aligned("DenseNet121")[0]
    F = {}
    for fam in NEW:
        X, y_f, fn_f = load_family(fam)
        assert fn_f == filenames and np.array_equal(y_f, y)
        F[fam] = X
    F["ref_densenet"], F["ref_resnet"] = X_dense, X_res

    rows, fold_auc = [], {}
    for seed in SEEDS:
        splits = cv_splits(y, seed)
        p = all_streams(F, X_clin, y, splits)
        for name, s in p.items():
            m = point_metrics(y, s)
            m["brier"] = float(np.mean((s - y) ** 2))
            rows.append(dict(seed=seed, stream=name, **m))
            for f, (_, te) in enumerate(splits):
                fold_auc[(seed, f, name)] = roc_auc_score(y[te], s[te])
        print(f"  seed {seed} done", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS_DIR / "features_eval_per_seed.csv", index=False, float_format="%.4f")
    metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1", "brier"]
    summary = per_seed.groupby("stream")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("auc_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "features_eval_summary.csv", index=False, float_format="%.4f")

    # Paired comparisons on identical splits
    n_test = len(y) // 5
    n_train = len(y) - n_test
    ps = per_seed.set_index(["seed", "stream"])["auc"]
    comparisons = [(s, "ref_densenet") for s in NEW + ["ref_resnet", "img_ens"]]
    comparisons += [(f"fusion_{s}", "clinical_lda") for s in list(F) + ["img_ens"]]
    comparisons += [("fusion_img_ens", "fusion_ref_densenet")]
    prow = []
    for a, b in comparisons:
        d = [fold_auc[(s, f, a)] - fold_auc[(s, f, b)] for s in SEEDS for f in range(5)]
        t, pv = corrected_t(d, n_train, n_test)
        sd = np.array([ps[(s, a)] - ps[(s, b)] for s in SEEDS])
        prow.append(dict(stream=a, reference=b, primary=a in (PRIMARY_IMAGE, PRIMARY_FUSION),
                         pooled_auc_diff_mean=sd.mean(), seeds_positive=int((sd > 0).sum()),
                         fold_auc_diff_mean=np.mean(d), corrected_t=t, corrected_p=pv))
    paired = pd.DataFrame(prow)
    paired.to_csv(RESULTS_DIR / "features_eval_paired.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    print(f"\n{len(SEEDS)} seeds x 5-fold CV, fixed pipeline (PCA-10 + L2-LogReg C=0.1); "
          f"primary: {PRIMARY_IMAGE}, {PRIMARY_FUSION}")
    print(summary[["stream", "auc_mean", "auc_std", "accuracy_mean", "sensitivity_mean",
                   "specificity_mean", "brier_mean"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nPaired comparisons (corrected resampled t-test over 100 folds):")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
