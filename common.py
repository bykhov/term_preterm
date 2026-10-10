"""
Shared helpers
==============
Paths, data loading, and fold-level fit/predict functions for the image,
clinical, early-fusion, and late-fusion models.
"""

import warnings
from pathlib import Path

from PIL import Image  # noqa: F401  (must load before torch on this Windows env: DLL clash)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import DATA_DIR, RANDOM_STATE, load_or_extract

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLINICAL_CSV = DATA_DIR / "clinical_data.csv"

FEATURE_COLS = [
    "Age", "Pregnancy number", "Past births", "Previous preterm births",
    "Previous cesarean section", "Cervical length (cm)", "Smoker",
    "Pre-pregnancy diseases", "Pregnancy diseases", "Treatment to stop labor",
]
CL_COL = "Cervical length (cm)"
CL_IDX = FEATURE_COLS.index(CL_COL)
CS_COL = "Previous cesarean section"
CS_IDX = FEATURE_COLS.index(CS_COL)
CS_MISSING_CODE = 9
N_SPLITS = 5


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
def load_clinical_data():
    """Load clinical_data.csv and return X, y, filenames."""
    df = pd.read_csv(CLINICAL_CSV)
    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values.astype(int)
    filenames = df["filename"].tolist()
    return X, y, filenames


def load_aligned(model_name):
    """CNN features and clinical features aligned by filename.

    The missing-value code 9 in "Previous cesarean section" is recoded to NaN;
    every clinical pipeline imputes it with the training-fold median.
    Returns X_cnn, X_clin, y, filenames.
    """
    X_cnn, y, filenames = load_or_extract(model_name)
    X_clin_all, y_clin, fn_clin = load_clinical_data()
    lookup = {fn: i for i, fn in enumerate(fn_clin)}
    order = [lookup[fn] for fn in filenames]
    X_clin = X_clin_all[order].copy()
    X_clin[X_clin[:, CS_IDX] == CS_MISSING_CODE, CS_IDX] = np.nan
    assert np.array_equal(y, y_clin[order]), "label mismatch"
    return X_cnn, X_clin, y, filenames


def cv_splits(y, seed=RANDOM_STATE, n_splits=N_SPLITS):
    """Shuffled stratified K-fold splits for the given seed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ──────────────────────────────────────────────────────────────
# Fold-level fit / predict
# ──────────────────────────────────────────────────────────────
def reduce_cnn(X_tr, y_tr, X_te, fs_mode, anova_k, n_pca, scale=True):
    """Variance filter -> [scaler] -> [ANOVA] -> [PCA], fitted on the training part."""
    keep = np.var(X_tr, axis=0) > 1e-10
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]
    if scale:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    if fs_mode in ("anova", "anova_pca"):
        sel = SelectKBest(f_classif, k=min(anova_k, X_tr.shape[1])).fit(X_tr, y_tr)
        X_tr, X_te = sel.transform(X_tr), sel.transform(X_te)
    if fs_mode in ("pca", "anova_pca"):
        pca = PCA(n_components=min(n_pca, X_tr.shape[1]),
                  random_state=RANDOM_STATE).fit(X_tr)
        X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)
    return X_tr, X_te


def fit_predict_image(X_tr, y_tr, X_te, fs_mode, anova_k, n_pca, make_clf):
    """Image pipeline as in pipeline.run_cv. Returns P(preterm) on X_te."""
    Z_tr, Z_te = reduce_cnn(X_tr, y_tr, X_te, fs_mode, anova_k, n_pca, scale=True)
    clf = make_clf().fit(Z_tr, y_tr)
    return clf.predict_proba(Z_te)[:, 1]


def fit_predict_tabular(X_tr, y_tr, X_te, make_clf):
    """Median imputer -> StandardScaler -> classifier (clinical pipelines). Returns P(preterm)."""
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("clf", make_clf())])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def oof(splits, fit_predict, n):
    """Generic out-of-fold probability vector. fit_predict(train_idx, test_idx) -> p."""
    p = np.full(n, np.nan)
    for tr, te in splits:
        p[te] = fit_predict(tr, te)
    return p


# ──────────────────────────────────────────────────────────────
# Late fusion rules
# ──────────────────────────────────────────────────────────────
def fuse(p_img, p_clin, rule, n_pos, n_neg):
    """Return fused P(preterm)-like score; threshold 0.5 gives the label."""
    eps = 1e-10
    if rule == "llr":
        llr = (np.log((p_img + eps) / (1 - p_img + eps))
               + np.log((p_clin + eps) / (1 - p_clin + eps)) + np.log(n_pos / n_neg))
        return 1 / (1 + np.exp(-llr))
    if rule == "product":
        n = n_pos + n_neg
        num = (p_img + eps) * (p_clin + eps) / (n_pos / n + eps)
        den = (1 - p_img + eps) * (1 - p_clin + eps) / (n_neg / n + eps)
        return num / (num + den)
    if rule == "soft":
        return (p_img + p_clin) / 2.0
    raise ValueError(rule)


# ──────────────────────────────────────────────────────────────
# Classifiers
# ──────────────────────────────────────────────────────────────
def lda():
    return LinearDiscriminantAnalysis()


def logreg01():
    return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced",
                              random_state=RANDOM_STATE)
