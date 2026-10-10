"""
Statistics: CL Ablation, Paired Tests, Permutation Tests
========================================================
For the pre-specified selection-free models (features_eval.py, ef_selection_free.py):

  late_fusion   : img_ens soft-voted with clinical LDA (primary)
  ef_ens        : mean of the four per-family early-fusion models (secondary)
  img_ens, clinical_lda
  cl_only       : cervical length alone (LDA)
  cl_only_lr    : cervical length alone (balanced LogReg, C=0.1)
  *_no_cl       : clinical LDA, late fusion and ef_ens without cervical length

1. 20 seeds x 5-fold CV on the features_eval.py splits: summary of all streams.
2. Paired comparisons: Nadeau-Bengio corrected t-test over 100 folds, plus
   DeLong and exact McNemar on the pooled OOF of every seed (median p, number
   of seeds with p < 0.05). The primary comparison is late_fusion vs
   clinical_lda; Holm correction over the secondary comparisons.
3. Complementarity of img_ens and clinical_lda (labels at 0.5): disagreement,
   Cohen's kappa, Yule's Q (stats_utils.py), exact McNemar, and the
   probability correlation.
4. Label-permutation test on the seed-42 split for late_fusion, ef_ens and
   img_ens (full CV refit per permutation).

Outputs (results/):
  statistical_tests_per_seed.csv, statistical_tests_summary.csv,
  statistical_tests_paired.csv, statistical_tests_complementarity.csv,
  statistical_tests_permutation.csv

Usage:
    python statistical_tests.py [N_PERM]
"""

import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import (CL_IDX, FEATURE_COLS, RANDOM_STATE, RESULTS_DIR, cv_splits,
                    fit_predict_tabular, fuse, lda, load_aligned, logreg01, oof)
from ef_selection_free import ef_streams
from features_eval import ENS_MEMBERS, NEW, SEEDS, image_oof
from features_new import load_family
from stats_utils import (contingency_table, corrected_t, delong_test, disagreement_measure, holm,
                         mcnemar_exact, point_metrics, yule_q)

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NO_CL = [i for i in range(len(FEATURE_COLS)) if i != CL_IDX]
PRIMARY = ("late_fusion", "clinical_lda")
SECONDARY = [
    ("ef_ens", "clinical_lda"), ("late_fusion", "ef_ens"), ("img_ens", "clinical_lda"),
    ("img_ens", "cl_only"), ("late_fusion", "cl_only"), ("ef_ens", "cl_only"),
    ("img_ens", "cl_only_lr"), ("late_fusion", "cl_only_lr"), ("ef_ens", "cl_only_lr"),
    ("clinical_lda", "clinical_no_cl"), ("late_fusion", "late_fusion_no_cl"),
    ("ef_ens", "ef_ens_no_cl"),
]
PERM_STREAMS = ["late_fusion", "ef_ens", "img_ens"]


def clinical_oof(X, y, splits, make_clf=lda):
    return oof(splits, lambda tr, te: fit_predict_tabular(X[tr], y[tr], X[te], make_clf), len(y))


def model_streams(F, X_clin, y, splits, with_ablation=True):
    """OOF probabilities of the pre-specified models (and the CL ablation)."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    p = {}
    p["img_ens"] = np.mean([image_oof(F[f], y, splits) for f in ENS_MEMBERS], axis=0)
    p["clinical_lda"] = clinical_oof(X_clin, y, splits)
    p["late_fusion"] = fuse(p["img_ens"], p["clinical_lda"], "soft", n_pos, n_neg)
    p["ef_ens"] = ef_streams(F, X_clin, y, splits)["ef_ens"]
    if with_ablation:
        X_no_cl = X_clin[:, NO_CL]
        p["cl_only"] = clinical_oof(X_clin[:, [CL_IDX]], y, splits)
        p["cl_only_lr"] = clinical_oof(X_clin[:, [CL_IDX]], y, splits, logreg01)
        p["clinical_no_cl"] = clinical_oof(X_no_cl, y, splits)
        p["late_fusion_no_cl"] = fuse(p["img_ens"], p["clinical_no_cl"], "soft", n_pos, n_neg)
        p["ef_ens_no_cl"] = ef_streams(F, X_no_cl, y, splits)["ef_ens"]
    return p


def one_perm(i, F, X_clin, y, splits):
    y_perm = np.random.default_rng(RANDOM_STATE + 1 + i).permutation(y)
    p = model_streams(F, X_clin, y_perm, splits, with_ablation=False)
    return {k: roc_auc_score(y_perm, p[k]) for k in PERM_STREAMS}


if __name__ == "__main__":
    _, X_clin, y, filenames = load_aligned("ResNet50")
    F = {}
    for fam in NEW:
        X, y_f, fn_f = load_family(fam)
        assert fn_f == filenames and np.array_equal(y_f, y)
        F[fam] = X
    F["ref_densenet"] = load_aligned("DenseNet121")[0]

    # 1. Repeated CV
    rows, fold_auc, pooled = [], {}, {}
    for seed in SEEDS:
        splits = cv_splits(y, seed)
        p = model_streams(F, X_clin, y, splits)
        pooled[seed] = p
        for name, s in p.items():
            m = point_metrics(y, s)
            m["brier"] = float(np.mean((s - y) ** 2))
            rows.append(dict(seed=seed, stream=name, **m))
            for f, (_, te) in enumerate(splits):
                fold_auc[(seed, f, name)] = roc_auc_score(y[te], s[te])
        print(f"  seed {seed} done", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS_DIR / "statistical_tests_per_seed.csv", index=False, float_format="%.4f")
    metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1", "brier"]
    summary = per_seed.groupby("stream")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("auc_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "statistical_tests_summary.csv", index=False, float_format="%.4f")

    # 2. Paired comparisons
    n_test = len(y) // 5
    n_train = len(y) - n_test
    ps = per_seed.set_index(["seed", "stream"])
    prow = []
    for a, b in [PRIMARY] + SECONDARY:
        d = [fold_auc[(s, f, a)] - fold_auc[(s, f, b)] for s in SEEDS for f in range(5)]
        t, pv = corrected_t(d, n_train, n_test)
        auc_d = np.array([ps.loc[(s, a), "auc"] - ps.loc[(s, b), "auc"] for s in SEEDS])
        acc_d = np.array([ps.loc[(s, a), "accuracy"] - ps.loc[(s, b), "accuracy"] for s in SEEDS])
        p_delong, p_mcn = [], []
        for s in SEEDS:
            pa, pb = pooled[s][a], pooled[s][b]
            p_delong.append(delong_test(y, pa, pb)[4])
            p_mcn.append(mcnemar_exact(y, (pa >= 0.5).astype(int), (pb >= 0.5).astype(int))[2])
        prow.append(dict(model=a, reference=b, primary=(a, b) == PRIMARY,
                         auc_diff_mean=auc_d.mean(), auc_seeds_positive=int((auc_d > 0).sum()),
                         acc_diff_mean=acc_d.mean(), corrected_t=t, corrected_p=pv,
                         delong_p_median=np.median(p_delong),
                         delong_seeds_p05=int(np.sum(np.array(p_delong) < 0.05)),
                         mcnemar_p_median=np.median(p_mcn),
                         mcnemar_seeds_p05=int(np.sum(np.array(p_mcn) < 0.05))))
    paired = pd.DataFrame(prow)
    sec = ~paired["primary"]
    paired["corrected_p_holm"] = paired["corrected_p"]
    paired.loc[sec, "corrected_p_holm"] = holm(paired.loc[sec, "corrected_p"].values)
    paired.to_csv(RESULTS_DIR / "statistical_tests_paired.csv", index=False, float_format="%.4f")

    # 3. Complementarity of the two late-fusion streams
    crow = []
    for s in SEEDS:
        pi, pc = pooled[s]["img_ens"], pooled[s]["clinical_lda"]
        li, lc = (pi >= 0.5).astype(int), (pc >= 0.5).astype(int)
        n11, n10, n01, n00 = contingency_table(li, lc)
        n_b, n_c, p_mcn = mcnemar_exact(y, li, lc)
        crow.append(dict(seed=s, disagreement=disagreement_measure(li, lc),
                         kappa=cohen_kappa_score(li, lc), yule_q=yule_q(n11, n10, n01, n00),
                         mcnemar_b=n_b, mcnemar_c=n_c, mcnemar_p=p_mcn,
                         both_wrong=int(np.sum((li != y) & (lc != y))),
                         prob_corr=np.corrcoef(pi, pc)[0, 1]))
    comp = pd.DataFrame(crow)
    comp.to_csv(RESULTS_DIR / "statistical_tests_complementarity.csv", index=False,
                float_format="%.4f")

    # 4. Permutation test (seed-42 split)
    splits = cv_splits(y, RANDOM_STATE)
    observed = {k: roc_auc_score(y, pooled[RANDOM_STATE][k]) for k in PERM_STREAMS}
    null = pd.DataFrame(Parallel(n_jobs=7, verbose=5)(
        delayed(one_perm)(i, F, X_clin, y, splits) for i in range(N_PERM)))
    perm = pd.DataFrame([dict(stream=k, auc=observed[k], auc_null_mean=null[k].mean(),
                              auc_null_95=np.percentile(null[k], 95),
                              auc_p=(1 + np.sum(null[k].values >= observed[k] - 1e-12))
                              / (1 + N_PERM)) for k in PERM_STREAMS])
    perm.to_csv(RESULTS_DIR / "statistical_tests_permutation.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 240)
    print(f"\n{len(SEEDS)} seeds x 5-fold CV, pre-specified models and CL ablation")
    print(summary[["stream", "auc_mean", "auc_std", "accuracy_mean", "sensitivity_mean",
                   "specificity_mean", "brier_mean"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSeed 42 AUC: " + ", ".join(
        f"{k} {roc_auc_score(y, v):.3f}" for k, v in pooled[RANDOM_STATE].items()))
    print("\nPaired comparisons (Holm over the secondary comparisons):")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nComplementarity img_ens vs clinical_lda (mean / SD over seeds; seed 42):")
    cm = comp.drop(columns="seed")
    print(pd.DataFrame({"mean": cm.mean(), "sd": cm.std(),
                        "seed42": comp.set_index("seed").loc[RANDOM_STATE]}).to_string(
        float_format=lambda v: f"{v:.3f}"))
    print(f"\nPermutation test (seed 42, N_PERM={N_PERM}):")
    print(perm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
