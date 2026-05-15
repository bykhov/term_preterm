# Best Clinical Classification Pipelines

Selected from 36 classifiers evaluated via 5-fold stratified cross-validation on 80 patients (51 term, 29 preterm).

## Summary

| Criterion | Classifier | Accuracy | Sensitivity | Specificity | F1 | AUC | TP | TN | FP | FN |
|-----------|-----------|----------|-------------|-------------|----|-----|----|----|----|----|
| Best Accuracy and F1 | LDA | **0.788** | 0.552 | 0.922 | **0.653** | 0.753 | 16 | 47 | 4 | 13 |
| Best AUC (no proba) | Passive Aggressive | 0.750 | 0.621 | 0.824 | 0.643 | **0.764** | 18 | 42 | 9 | 11 |
| Best AUC (with proba) | LogReg C=0.1 | 0.750 | 0.621 | 0.824 | 0.643 | 0.762 | 18 | 42 | 9 | 11 |

---

## Pipeline #1: Best Accuracy (78.8%)

```
Clinical Data (10 features) → StandardScaler → LDA
```

### Stages

1. **Clinical Features:** Ten maternal variables — age, pregnancy number, past births, previous preterm births, previous cesarean section, cervical length (cm), smoking status, pre-pregnancy diseases, pregnancy diseases, treatment to stop labor.

2. **StandardScaler:** Zero-mean, unit-variance normalization fitted on training fold.

3. **Linear Discriminant Analysis (LDA):** Projects data onto the axis maximizing between-class to within-class variance ratio. Provides posterior class probabilities via `predict_proba`. No hyperparameters to tune.

### Performance Profile
- High specificity (92.2%) — reliably identifies term births
- Moderate sensitivity (55.2%) — misses 13 of 29 preterm cases
- Highest overall accuracy in the clinical sweep
- Probabilistic outputs available for late fusion

---

## Pipeline #2: Best AUC with Probabilistic Output (75.0%)

```
Clinical Data (10 features) → StandardScaler → LogisticRegression(C=0.1, balanced)
```

### Stages

1. **Clinical Features:** Same ten maternal variables as Pipeline #1.

2. **StandardScaler:** Zero-mean, unit-variance normalization fitted on training fold.

3. **Logistic Regression (C=0.1):** L2-regularized linear classifier with balanced class weights. Provides calibrated posterior probabilities via `predict_proba`.

### Performance Profile
- Balanced sensitivity (62.1%) and specificity (82.4%)
- Tied accuracy with Passive Aggressive (75.0%), comparable AUC (0.762 vs 0.764)
- Selected over Passive Aggressive because `predict_proba` is required for late fusion combining methods
- Probabilistic outputs available for late fusion

### Note on Passive Aggressive
Passive Aggressive achieves marginally higher AUC (0.764 vs 0.762) but lacks `predict_proba`, which is required for probability-based combining methods (linear weighting, LLR, product rule, stacking). LogReg C=0.1 matches its accuracy and F1 while providing calibrated probabilities.

---

## Cross-Validation Methodology

- **5-fold Stratified K-Fold** (`shuffle=True`, `random_state=42`), preserving class proportions
- **StandardScaler fitted inside each fold** — no data leakage
- Metrics computed from aggregated out-of-fold predictions across all 80 samples
- AUC computed from `predict_proba` values (both LDA and LogReg provide `predict_proba`)
