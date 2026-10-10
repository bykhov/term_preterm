"""
Selection-Free Early Fusion
===========================
Early-fusion counterparts of the selection-free design in features_eval.py,
all fixed in advance (no data-driven selection):

  ef_concat   : per family (hc, mid, clip, ref_densenet) variance filter ->
                StandardScaler -> PCA(10), fitted in-fold; concatenate the
                4 x 10 components with the 10 clinical variables (median-
                imputed) -> StandardScaler -> LogisticRegression(L2, C=0.1,
                balanced)
  ef_<family> : the same with one family + clinical
  ef_ens      : mean probability of the four ef_<family> models

References from features_eval.all_streams: fusion_img_ens (primary late
fusion), clinical_lda, img_ens. Same 20 seeds x 5-fold splits.

Outputs (results/):
  ef_selection_free_per_seed.csv, ef_selection_free_summary.csv,
  ef_selection_free_paired.csv

Usage:
    python ef_selection_free.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import RESULTS_DIR, cv_splits, load_aligned, oof, reduce_cnn
from features_eval import ENS_MEMBERS, NEW, SEEDS, all_streams, logreg
from features_new import load_family
from stats_utils import corrected_t, point_metrics

REFERENCES = ["fusion_img_ens", "clinical_lda"]


def fit_predict_ef(blocks, X_clin, y, tr, te):
    """Per-block in-fold PCA-10, concatenated with imputed clinical data -> scaler -> LogReg."""
    Z_tr, Z_te = [], []
    for X in blocks:
        a, b = reduce_cnn(X[tr], y[tr], X[te], "pca", None, 10, scale=True)
        Z_tr.append(a)
        Z_te.append(b)
    imp = SimpleImputer(strategy="median").fit(X_clin[tr])
    Z_tr.append(imp.transform(X_clin[tr]))
    Z_te.append(imp.transform(X_clin[te]))
    Z_tr, Z_te = np.hstack(Z_tr), np.hstack(Z_te)
    sc = StandardScaler().fit(Z_tr)
    clf = logreg().fit(sc.transform(Z_tr), y[tr])
    return clf.predict_proba(sc.transform(Z_te))[:, 1]


def ef_streams(F, X_clin, y, splits):
    n = len(y)
    p = {"ef_concat": oof(splits, lambda tr, te: fit_predict_ef(
        [F[f] for f in ENS_MEMBERS], X_clin, y, tr, te), n)}
    for f in ENS_MEMBERS:
        p[f"ef_{f}"] = oof(splits, lambda tr, te, f=f: fit_predict_ef([F[f]], X_clin, y, tr, te), n)
    p["ef_ens"] = np.mean([p[f"ef_{f}"] for f in ENS_MEMBERS], axis=0)
    return p


if __name__ == "__main__":
    X_res, X_clin, y, filenames = load_aligned("ResNet50")
    F = {}
    for fam in NEW:
        X, y_f, fn_f = load_family(fam)
        assert fn_f == filenames and np.array_equal(y_f, y)
        F[fam] = X
    F["ref_densenet"] = load_aligned("DenseNet121")[0]

    rows, fold_auc = [], {}
    for seed in SEEDS:
        splits = cv_splits(y, seed)
        ref = all_streams(F, X_clin, y, splits)
        p = ef_streams(F, X_clin, y, splits)
        p.update({k: ref[k] for k in REFERENCES + ["img_ens"]})
        for name, s in p.items():
            m = point_metrics(y, s)
            m["brier"] = float(np.mean((s - y) ** 2))
            rows.append(dict(seed=seed, stream=name, **m))
            for f, (_, te) in enumerate(splits):
                fold_auc[(seed, f, name)] = roc_auc_score(y[te], s[te])
        print(f"  seed {seed} done", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS_DIR / "ef_selection_free_per_seed.csv", index=False,
                    float_format="%.4f")
    metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1", "brier"]
    summary = per_seed.groupby("stream")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("auc_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "ef_selection_free_summary.csv", index=False, float_format="%.4f")

    n_test = len(y) // 5
    n_train = len(y) - n_test
    ps = per_seed.set_index(["seed", "stream"])["auc"]
    ef_names = ["ef_concat", "ef_ens"] + [f"ef_{f}" for f in ENS_MEMBERS]
    prow = []
    for a in ef_names:
        for b in REFERENCES:
            d = [fold_auc[(s, f, a)] - fold_auc[(s, f, b)] for s in SEEDS for f in range(5)]
            t, pv = corrected_t(d, n_train, n_test)
            sd = np.array([ps[(s, a)] - ps[(s, b)] for s in SEEDS])
            prow.append(dict(stream=a, reference=b, pooled_auc_diff_mean=sd.mean(),
                             seeds_positive=int((sd > 0).sum()), fold_auc_diff_mean=np.mean(d),
                             corrected_t=t, corrected_p=pv))
    paired = pd.DataFrame(prow)
    paired.to_csv(RESULTS_DIR / "ef_selection_free_paired.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    print(f"\n{len(SEEDS)} seeds x 5-fold CV, selection-free early fusion")
    print(summary[["stream", "auc_mean", "auc_std", "accuracy_mean", "sensitivity_mean",
                   "specificity_mean", "brier_mean"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nPaired comparisons (corrected resampled t-test over 100 folds):")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
