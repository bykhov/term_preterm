# Pipeline #1: ResNet50 → ANOVA200→PCA5 + LDA

**Dataset:** 80 cropped ultrasound images (Term: 51, Preterm: 29)

## Pipeline
ResNet50 (2048-d) → Variance Filter → StandardScaler → ANOVA(k=200) → PCA(5) → LDA

## 5-Fold CV Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.762 |
| Sensitivity | 0.586 |
| Specificity | 0.863 |
| F1 | 0.642 |
| AUC | 0.656 |
| TP | 17 |
| TN | 44 |
| FP | 7 |
| FN | 12 |

## Notes
- LDA provides probabilistic outputs via posterior class probabilities
- Proba saved to pipeline1_proba.csv for late fusion analysis
