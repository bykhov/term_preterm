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

| Metric | Value |
|--------|-------|
| Accuracy | 0.688 |
| Sensitivity | 0.828 |
| Specificity | 0.608 |
| F1 | 0.658 |
| AUC | 0.718 |
| TP | 24 |
| TN | 31 |
| FP | 20 |
| FN | 5 |

## Notes
- SGD with modified Huber loss provides probabilistic outputs
- Proba saved to pipeline2_proba.csv for late fusion analysis
