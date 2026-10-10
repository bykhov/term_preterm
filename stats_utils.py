"""
Statistical utilities: DeLong AUC test, exact McNemar, Holm correction,
corrected resampled t-test, agreement measures, metrics
================================================================================
DeLong follows the fast algorithm of Sun & Xu (IEEE SPL, 2014).
"""

import numpy as np
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _delong_components(y, scores):
    """scores: (k, n). Returns aucs (k,), covariance (k, k)."""
    y = np.asarray(y).astype(int)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_ci(y, score, alpha=0.05):
    """AUC with DeLong (1 - alpha) CI, clipped to [0, 1]."""
    aucs, cov = _delong_components(y, np.atleast_2d(score))
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se)


def delong_test(y, score_a, score_b):
    """Paired two-sided DeLong test. Returns auc_a, auc_b, diff, z, p."""
    aucs, cov = _delong_components(y, np.vstack([score_a, score_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float(diff), 0.0, 1.0
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(diff), float(z), float(p)


def mcnemar_exact(y, pred_a, pred_b):
    """Exact (binomial) McNemar test on paired correctness.

    Returns b (A wrong, B right), c (A right, B wrong), p.
    """
    err_a, err_b = pred_a != y, pred_b != y
    b = int(np.sum(err_a & ~err_b))
    c = int(np.sum(~err_a & err_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def corrected_t(d, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on per-fold differences."""
    d = np.asarray(d, float)
    J = len(d)
    se = np.sqrt((1 / J + n_test / n_train) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else np.nan
    return t, 2 * t_dist.sf(abs(t), J - 1)


def contingency_table(pred_a, pred_b):
    n11 = int(np.sum((pred_a == 1) & (pred_b == 1)))
    n10 = int(np.sum((pred_a == 1) & (pred_b == 0)))
    n01 = int(np.sum((pred_a == 0) & (pred_b == 1)))
    n00 = int(np.sum((pred_a == 0) & (pred_b == 0)))
    return n11, n10, n01, n00


def disagreement_measure(pred_a, pred_b):
    return float(np.mean(pred_a != pred_b))


def yule_q(n11, n10, n01, n00):
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    if den == 0:
        return 0.0
    return num / den


def point_metrics(y, p, thr=0.5):
    """Pooled OOF metrics from P(preterm)."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y, pred, zero_division=0.0),
        "auc": roc_auc_score(y, p),
    }
