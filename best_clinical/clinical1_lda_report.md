# Clinical Pipeline #1: Clinical → StandardScaler + LDA

**Dataset:** 80 patients (51 term, 29 preterm)

## Pipeline
10 clinical features → StandardScaler → LDA

## Clinical Features
Age, Pregnancy number, Past births, Previous preterm births,
Previous cesarean section, Cervical length (cm), Smoker,
Pre-pregnancy diseases, Pregnancy diseases, Treatment to stop labor

## 5-Fold CV Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.787 | (0.699–0.876) |
| Sensitivity | 0.552 | (0.341–0.753) |
| Specificity | 0.922 | (0.828–1.000) |
| F1 | 0.653 | (0.445–0.835) |
| AUC | 0.753 | (0.684–0.846) |
| TP | 16 | |
| TN | 47 | |
| FP | 4 | |
| FN | 13 | |

## Notes
- LDA provides posterior class probabilities via `predict_proba`
- High specificity (0.922) — reliably identifies term births
- Moderate sensitivity (0.552) — misses 13 of 29 preterm cases
- Proba saved to clinical1_lda_proba.csv for late fusion analysis
