# Pipeline #2: DenseNet121 → ANOVA100→PCA5 + SGD(modified_huber)

**Dataset:** 80 cropped ultrasound images (Term: 51, Preterm: 29)

## Pipeline
DenseNet121 (1024-d) → Variance Filter → StandardScaler → ANOVA(k=100) → PCA(5) → SGD(modified_huber)

## Classifier Parameters
- loss: modified_huber (smooth hinge approximation, yields probability estimates)
- class_weight: balanced
- max_iter: 5000
- random_state: 42

## 5-Fold CV Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.688 | (0.565–0.810) |
| Sensitivity | 0.862 | (0.602–1.000) |
| Specificity | 0.588 | (0.311–0.860) |
| F1 | 0.667 | (0.548–0.776) |
| AUC | 0.725 | (0.620–0.819) |
| TP | 25 | |
| TN | 30 | |
| FP | 21 | |
| FN | 4 | |

## Notes
- SGD with modified Huber loss provides probabilistic outputs
- Proba saved to pipeline2_proba.csv for late fusion analysis
