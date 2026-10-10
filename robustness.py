"""
Robustness (Multiverse) Analysis
================================
How much does the fusion result depend on its fixed design choices? The
primary model stays fixed; every specification of the grid below is evaluated
on the same 20 seeds x 5-fold splits and reported as a distribution. The grid
is never used to select a model.

  PCA components per representation : 5, 10 (primary), 20
  LogReg C (image and early fusion)  : 0.01, 0.1 (primary), 1
  Representation subset              : the 15 non-empty subsets of
                                       {HC, MID, CLIP, DenseNet} (primary: all 4)
  Fusion level                       : late (soft vote, primary), early
                                       (per-representation EF, then the mean)
  Clinical classifier (late only)    : LDA (primary), LogReg (L2, C=0.1, balanced)

270 late + 135 early fusion specifications; the image ensembles of every
(PCA, C, subset) and both clinical models are reported as references.
Image and early-fusion OOF probabilities are cached per (representation, PCA,
C, seed); an ensemble is the mean of its members' cached vectors.

Outputs (results/):
  robustness_specs.csv, robustness_oat.csv,
  robustness_summary.csv, fig_robustness_spec_curve.pdf/.png

Usage:
    python robustness.py
    PLOT_ONLY=1 python robustness.py  (redraw the figure only)
"""

import os
from itertools import combinations, product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import (RANDOM_STATE, RESULTS_DIR, cv_splits, fit_predict_image,  # noqa: E402
                    fit_predict_tabular, fuse, lda, load_aligned, logreg01, oof,
                    reduce_cnn)
from features_eval import ENS_MEMBERS, NEW, SEEDS  # noqa: E402
from features_new import load_family  # noqa: E402
from stats_utils import corrected_t, point_metrics  # noqa: E402

PCAS = [5, 10, 20]
CS = [0.01, 0.1, 1.0]
SUBSETS = [s for k in range(1, 5) for s in combinations(ENS_MEMBERS, k)]
REP_LABEL = {"hc": "HC", "mid": "MID", "clip": "CLIP", "ref_densenet": "DenseNet"}
FIG_LABEL = {"hc": "Handcrafted", "mid": "ResNet-50 L2-3", "clip": "BiomedCLIP",
             "ref_densenet": "DenseNet-121"}
CLIN = {"lda": lda, "logreg": logreg01}
PRIMARY = dict(fusion="late", clinical="lda", n_pca=10, C=0.1, subset=tuple(ENS_MEMBERS))
METRICS = ["auc", "accuracy", "sensitivity", "specificity", "brier"]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#6b6b6b"


def logreg_c(C):
    return lambda: LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                      random_state=RANDOM_STATE)


def fit_predict_ef(X, X_clin, y, tr, te, n_pca, C):
    """ef_selection_free.fit_predict_ef for one representation, with PCA n and C as arguments."""
    a, b = reduce_cnn(X[tr], y[tr], X[te], "pca", None, n_pca, scale=True)
    imp = SimpleImputer(strategy="median").fit(X_clin[tr])
    Z_tr = np.hstack([a, imp.transform(X_clin[tr])])
    Z_te = np.hstack([b, imp.transform(X_clin[te])])
    sc = StandardScaler().fit(Z_tr)
    clf = logreg_c(C)().fit(sc.transform(Z_tr), y[tr])
    return clf.predict_proba(sc.transform(Z_te))[:, 1]


def seed_cache(seed, F, X_clin, y):
    """OOF probabilities of every (representation, PCA, C) image and EF model, and both clinical models."""
    splits = cv_splits(y, seed)
    n = len(y)
    img, ef = {}, {}
    for rep, n_pca, C in product(ENS_MEMBERS, PCAS, CS):
        X = F[rep]
        img[(rep, n_pca, C)] = oof(splits, lambda tr, te: fit_predict_image(
            X[tr], y[tr], X[te], "pca", None, n_pca, logreg_c(C)), n)
        ef[(rep, n_pca, C)] = oof(splits, lambda tr, te: fit_predict_ef(
            X, X_clin, y, tr, te, n_pca, C), n)
    clin = {k: oof(splits, lambda tr, te, m=m: fit_predict_tabular(X_clin[tr], y[tr], X_clin[te], m), n)
            for k, m in CLIN.items()}
    return seed, splits, img, ef, clin


def all_specs():
    """(fusion, clinical, n_pca, C, subset) of every specification, references included."""
    specs = [("clinical", c, None, None, ()) for c in CLIN]
    for n_pca, C, sub in product(PCAS, CS, SUBSETS):
        specs.append(("image", None, n_pca, C, sub))
        specs.append(("early", None, n_pca, C, sub))
        for c in CLIN:
            specs.append(("late", c, n_pca, C, sub))
    return specs


def spec_score(spec, img, ef, clin, n_pos, n_neg):
    fusion, c, n_pca, C, sub = spec
    if fusion == "clinical":
        return clin[c]
    if fusion == "early":
        return np.mean([ef[(r, n_pca, C)] for r in sub], axis=0)
    p_img = np.mean([img[(r, n_pca, C)] for r in sub], axis=0)
    return p_img if fusion == "image" else fuse(p_img, clin[c], "soft", n_pos, n_neg)


def spec_name(spec):
    fusion, c, n_pca, C, sub = spec
    if fusion == "clinical":
        return f"clinical_{c}"
    reps = "+".join(REP_LABEL[r] for r in sub)
    tail = f"_{c}" if fusion == "late" else ""
    return f"{fusion}{tail}_pca{n_pca}_C{C:g}_{reps}"


def is_primary(row, **change):
    want = {**PRIMARY, **change}
    return all(row[k] == v for k, v in want.items())


def spec_curve(specs, clin_auc, path):
    """Sorted mean AUC of all fusion specifications with a +-1 SD band."""
    d = specs[specs["fusion"].isin(["late", "early"])].sort_values("auc_mean").reset_index(drop=True)
    x = np.arange(len(d))
    prim = int(d.index[d["primary"]][0])
    col = np.where(d["fusion"] == "late", BLUE, ORANGE)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.fill_between(x, d["auc_mean"] - d["auc_std"], d["auc_mean"] + d["auc_std"],
                    color="#d9d9d9", lw=0, label="±1 SD over 20 repetitions")
    ax.scatter(x, d["auc_mean"], c=col, s=4, lw=0, zorder=3)
    ax.scatter([], [], c=BLUE, s=12, label="Late fusion")
    ax.scatter([], [], c=ORANGE, s=12, label="Early fusion")
    ax.scatter([prim], [d.loc[prim, "auc_mean"]], s=60, facecolors="none", edgecolors="black",
               lw=1.2, zorder=4, label="Primary model")
    for k, ls, lab in [("lda", "--", "Clinical LDA"), ("logreg", ":", "Clinical LogReg")]:
        ax.axhline(clin_auc[k], color=GRAY, ls=ls, lw=1, label=f"{lab} ({clin_auc[k]:.3f})")
    ax.set_ylabel("AUC (mean over 20 repetitions)")
    ax.set_xlabel(f"Fusion model (n = {len(d)}), sorted by AUC")
    ax.set_xlim(-3, len(d) + 2)
    ax.grid(axis="y", color="#ebebeb", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__" and os.environ.get("PLOT_ONLY"):
    # Redraw the figure from the saved specification table
    df = pd.read_csv(RESULTS_DIR / "robustness_specs.csv")
    clin_auc = {c: df.loc[df["spec"] == f"clinical_{c}", "auc_mean"].iloc[0] for c in CLIN}
    spec_curve(df, clin_auc, RESULTS_DIR / "fig_robustness_spec_curve")
    raise SystemExit

if __name__ == "__main__":
    _, X_clin, y, filenames = load_aligned("ResNet50")
    F = {}
    for fam in NEW:
        X, y_f, fn_f = load_family(fam)
        assert fn_f == filenames and np.array_equal(y_f, y)
        F[fam] = X
    F["ref_densenet"] = load_aligned("DenseNet121")[0]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    n_test = len(y) // 5
    n_train = len(y) - n_test

    caches = Parallel(n_jobs=7, verbose=5)(delayed(seed_cache)(s, F, X_clin, y) for s in SEEDS)

    # Per seed metrics and fold AUCs of every specification
    specs = all_specs()
    seed_m = {sp: [] for sp in specs}
    fold_auc = {sp: [] for sp in specs}
    for seed, splits, img, ef, clin in caches:
        for sp in specs:
            s = spec_score(sp, img, ef, clin, n_pos, n_neg)
            m = point_metrics(y, s)
            m["brier"] = float(np.mean((s - y) ** 2))
            seed_m[sp].append(m)
            fold_auc[sp].extend(roc_auc_score(y[te], s[te]) for _, te in splits)

    ref = {c: ("clinical", c, None, None, ()) for c in CLIN}
    rows = []
    for sp in specs:
        fusion, c, n_pca, C, sub = sp
        m = pd.DataFrame(seed_m[sp])
        row = dict(spec=spec_name(sp), fusion=fusion, clinical=c, n_pca=n_pca, C=C,
                   subset=sub, n_reps=len(sub), reps="+".join(REP_LABEL[r] for r in sub))
        for k in METRICS:
            row[f"{k}_mean"], row[f"{k}_std"] = m[k].mean(), m[k].std()
        for c_ref in CLIN:
            if fusion == "clinical":
                continue
            r_auc = np.array([q["auc"] for q in seed_m[ref[c_ref]]])
            dlt = m["auc"].values - r_auc
            row[f"dauc_{c_ref}"] = dlt.mean()
            row[f"seeds_pos_{c_ref}"] = int((dlt > 0).sum())
            row[f"p_{c_ref}"] = corrected_t(np.subtract(fold_auc[sp], fold_auc[ref[c_ref]]),
                                            n_train, n_test)[1]
        # Own clinical reference: the model a late fusion uses; LDA (primary) for the rest
        own = c if fusion == "late" else "lda"
        if fusion != "clinical":
            row["ref_own"] = own
            for k in ["dauc", "seeds_pos", "p"]:
                row[f"{k}_own"] = row[f"{k}_{own}"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["primary"] = df.apply(lambda r: is_primary(r), axis=1)

    out = df.copy()
    out["subset"] = out["reps"]
    out.drop(columns="reps").to_csv(RESULTS_DIR / "robustness_specs.csv", index=False,
                                    float_format="%.4f")

    # One-at-a-time table
    full = tuple(ENS_MEMBERS)
    oat = [("Primary", {}),
           ("PCs 5", dict(n_pca=5)), ("PCs 20", dict(n_pca=20)),
           ("C 0.01", dict(C=0.01)), ("C 1", dict(C=1.0)),
           ("Clinical LogReg", dict(clinical="logreg")),
           ("Early fusion", dict(fusion="early", clinical=None))]
    oat += [(f"Without {REP_LABEL[r]}", dict(subset=tuple(m for m in full if m != r)))
            for r in ENS_MEMBERS]
    oat += [(f"{REP_LABEL[r]} only", dict(subset=(r,))) for r in ENS_MEMBERS]
    orow = []
    for lab, ch in oat:
        hit = df[df.apply(lambda r: is_primary(r, **ch), axis=1)]
        assert len(hit) == 1, lab
        orow.append(dict(variant=lab, **hit.iloc[0][["spec", "auc_mean", "auc_std",
                                                     "accuracy_mean", "sensitivity_mean",
                                                     "specificity_mean", "brier_mean", "ref_own",
                                                     "dauc_own", "seeds_pos_own",
                                                     "p_own"]].to_dict()))
    oat_df = pd.DataFrame(orow)
    oat_df.to_csv(RESULTS_DIR / "robustness_oat.csv", index=False, float_format="%.4f")

    # Distribution summary and marginal means
    clin_auc = {c: df.loc[df["spec"] == f"clinical_{c}", "auc_mean"].iloc[0] for c in CLIN}
    groups = {"all fusion": df["fusion"].isin(["late", "early"]),
              "late (LDA)": (df["fusion"] == "late") & (df["clinical"] == "lda"),
              "late (LogReg)": (df["fusion"] == "late") & (df["clinical"] == "logreg"),
              "early": df["fusion"] == "early", "image only": df["fusion"] == "image"}
    srow = []
    for g, mask in groups.items():
        a = df.loc[mask, "auc_mean"]
        sub = df.loc[mask]
        srow.append(dict(
            group=g, n=int(mask.sum()), auc_median=a.median(), auc_q1=a.quantile(0.25),
            auc_q3=a.quantile(0.75), auc_min=a.min(), auc_max=a.max(),
            share_above_clin_lda=float((a > clin_auc["lda"]).mean()),
            share_above_both_clin=float((a > max(clin_auc.values())).mean()),
            share_above_own=float((sub["dauc_own"] > 0).mean()),
            share_seeds_pos18_own=float((sub["seeds_pos_own"] >= 18).mean()),
            share_p05_own=float((sub["p_own"] < 0.05).mean()),
            acc_median=sub["accuracy_mean"].median()))
    summ = pd.DataFrame(srow)
    fus = df[df["fusion"].isin(["late", "early"])]
    mrow = []
    for fac in ["fusion", "clinical", "n_pca", "C", "n_reps"]:
        for lev, g in fus.groupby(fac, dropna=False):
            mrow.append(dict(factor=fac, level=str(lev), n=len(g), auc_mean=g["auc_mean"].mean(),
                             auc_min=g["auc_mean"].min(), auc_max=g["auc_mean"].max()))
    for r in ENS_MEMBERS:
        for has, g in fus.groupby(fus["subset"].apply(lambda s, r=r: r in s)):
            mrow.append(dict(factor=f"includes {REP_LABEL[r]}", level=str(has), n=len(g),
                             auc_mean=g["auc_mean"].mean(), auc_min=g["auc_mean"].min(),
                             auc_max=g["auc_mean"].max()))
    marg = pd.DataFrame(mrow)
    with open(RESULTS_DIR / "robustness_summary.csv", "w", newline="") as fh:
        summ.to_csv(fh, index=False, float_format="%.4f")
        fh.write("\n")
        marg.to_csv(fh, index=False, float_format="%.4f")

    spec_curve(df, clin_auc, RESULTS_DIR / "fig_robustness_spec_curve")

    # Report
    pd.set_option("display.width", 240)
    n_late, n_early = int((df["fusion"] == "late").sum()), int((df["fusion"] == "early").sum())
    print(f"\n{len(SEEDS)} seeds x 5-fold CV; {n_late} late + {n_early} early fusion specifications")
    print(f"NaN AUCs: {int(df['auc_mean'].isna().sum())}")
    prim = df[df["primary"]].iloc[0]
    ef_prim = df[df.apply(lambda r: is_primary(r, fusion="early", clinical=None), axis=1)].iloc[0]
    clip = df[df.apply(lambda r: is_primary(r, subset=("clip",)), axis=1)].iloc[0]
    print(f"Sanity: primary {prim['auc_mean']:.4f} (expect 0.8259), EF-ENS "
          f"{ef_prim['auc_mean']:.4f} (0.8251), clinical LDA {clin_auc['lda']:.4f} (0.7415), "
          f"CLIP-only late {clip['auc_mean']:.4f} (0.8195); clinical LogReg "
          f"{clin_auc['logreg']:.4f}")
    print("\nDistribution of mean AUC over specifications:")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMarginal means (fusion specifications):")
    print(marg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nOne at a time (each change applied alone to the primary):")
    print(oat_df.drop(columns="spec").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nLowest 10 fusion specifications:")
    print(fus.nsmallest(10, "auc_mean")[["spec", "auc_mean", "auc_std", "dauc_own",
                                         "seeds_pos_own", "p_own"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nHighest 10 fusion specifications:")
    print(fus.nlargest(10, "auc_mean")[["spec", "auc_mean", "auc_std", "dauc_own",
                                        "seeds_pos_own", "p_own"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
