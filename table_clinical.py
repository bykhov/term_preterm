"""
Maternal Clinical Characteristics Table
=======================================
Computes the descriptive statistics and group tests directly from
Data/clinical_data.csv:
  continuous (age, cervical length)  : mean +/- SD, independent-samples t-test
  ordinal (gravidity, past births)   : mean +/- SD, Mann-Whitney U test
  binary                             : % yes, Fisher's exact test
"Previous cesarean section" is reported as any vs none; its missing-value
code 9 is excluded from that row's denominator.

Outputs (results/):
  table_clinical.csv

Usage:
    python table_clinical.py
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, ttest_ind

from common import CS_IDX, CS_MISSING_CODE, FEATURE_COLS, RESULTS_DIR, load_clinical_data

# (column, LaTeX label, kind, footnote letters)
ROWS = [
    ("Age", "Age (years)", "t", "a"),
    ("Pregnancy number", "Gravidity", "u", "b"),
    ("Past births", "Past births", "u", "b"),
    ("Previous preterm births", "Prev. preterm births (\\%)", "fisher", "c"),
    ("Previous cesarean section", "Prev. cesarean sect. (\\%)", "fisher", "c,f"),
    ("Cervical length (cm)", "Cervical length (cm)", "t", "a"),
    ("Smoker", "Smoking (\\%)", "fisher", "c"),
    ("Pre-pregnancy diseases", "Maternal comorbidities (\\%)", "fisher", "c,d"),
    ("Pregnancy diseases", "Antenatal complications (\\%)", "fisher", "c,e"),
    ("Treatment to stop labor", "Tocolytic therapy (\\%)", "fisher", "c"),
]


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


if __name__ == "__main__":
    X, y, _ = load_clinical_data()
    X = X.copy()
    X[X[:, CS_IDX] == CS_MISSING_CODE, CS_IDX] = np.nan
    term, pre = y == 0, y == 1

    recs, tex = [], []
    for col, label, kind, fn in ROWS:
        v = X[:, FEATURE_COLS.index(col)]
        a, b = v[term & ~np.isnan(v)], v[pre & ~np.isnan(v)]
        r = {"variable": col, "test": kind, "n_term": len(a), "n_preterm": len(b)}
        if kind in ("t", "u"):
            r.update(term_mean=a.mean(), term_sd=a.std(ddof=1),
                     preterm_mean=b.mean(), preterm_sd=b.std(ddof=1))
            r["p"] = (ttest_ind(a, b).pvalue if kind == "t"
                      else mannwhitneyu(a, b, alternative="two-sided").pvalue)
            cells = (f"${r['term_mean']:.2f} \\pm {r['term_sd']:.2f}$",
                     f"${r['preterm_mean']:.2f} \\pm {r['preterm_sd']:.2f}$")
        else:
            ka, kb = int((a > 0).sum()), int((b > 0).sum())
            r.update(term_yes=ka, preterm_yes=kb,
                     term_pct=100 * ka / len(a), preterm_pct=100 * kb / len(b))
            r["p"] = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
            cells = (f"{r['term_pct']:.1f} ({ka}/{len(a)})", f"{r['preterm_pct']:.1f} ({kb}/{len(b)})")
        recs.append(r)
        tex.append(f"\t{label}\\textsuperscript{{{fn}}} & {cells[0]} & {cells[1]} & "
                   f"{fmt_p(r['p'])} \\\\")

    df = pd.DataFrame(recs)
    df.to_csv(RESULTS_DIR / "table_clinical.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + "\n".join(tex))
