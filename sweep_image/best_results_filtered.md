# Best Image Classification Pipelines

Selected from 3,432 sweep combinations (3 CNN models × 44 classifiers × 26 feature selection configs).
Evaluated via 5-fold stratified cross-validation on 80 transvaginal cervical ultrasound images (51 term, 29 preterm).

## Summary

| Criterion | Model | Classifier | Feature Selection | Accuracy | Sensitivity | Specificity | F1 | AUC | TP | TN | FP | FN |
|-----------|-------|-----------|-------------------|----------|-------------|-------------|----|-----|----|----|----|----|
| Best Accuracy | ResNet50 | LDA | ANOVA200→PCA5 | **0.763** | 0.586 | 0.863 | 0.642 | 0.656 | 17 | 44 | 7 | 12 |
| Best F1 & AUC | DenseNet121 | SGD (modified Huber) | ANOVA100→PCA5 | 0.688 | 0.828 | 0.608 | **0.658** | **0.757** | 24 | 31 | 20 | 5 |

---

## Pipeline 1: Best Accuracy (76.3%)

```
ResNet50 (2048-d) → Variance Filter → StandardScaler → ANOVA(k=200) → PCA(5) → LDA
```

### Stages

1. **Feature Extraction:** ResNet50 pretrained on RadImageNet. Input images resized to 224×224, ImageNet-normalized. Backbone output passed through AdaptiveAvgPool2d → Flatten, producing a 2048-dimensional feature vector per image.

2. **Variance Filter:** Features with training-fold variance ≤ 1e-10 are removed (applied per fold to prevent data leakage).

3. **StandardScaler:** Zero-mean, unit-variance normalization fitted on training fold.

4. **ANOVA Feature Selection (k=200):** `SelectKBest(f_classif, k=200)` — selects the 200 features with the highest F-statistic (univariate ANOVA) between term and preterm classes.

5. **PCA (n=5):** Principal Component Analysis reduces from 200 to 5 components, capturing the dominant variance directions.

6. **Linear Discriminant Analysis (LDA):** Projects data onto the axis maximizing between-class to within-class variance ratio. Single hyperplane decision boundary. No hyperparameters to tune.

### Performance Profile
- High specificity (86.3%) — reliably identifies term births
- Moderate sensitivity (58.6%) — misses 12 of 29 preterm cases
- Highest overall accuracy in the sweep

---

## Pipeline 2: Best F1 & AUC (F1=0.658, AUC=0.757)

```
DenseNet121 (1024-d) → Variance Filter → StandardScaler → ANOVA(k=100) → PCA(5) → SGD(modified_huber)
```

### Stages

1. **Feature Extraction:** DenseNet121 pretrained on RadImageNet. Input images resized to 224×224, ImageNet-normalized. Dense block output → ReLU → AdaptiveAvgPool2d → Flatten, producing a 1024-dimensional feature vector per image.

2. **Variance Filter:** Same as Pipeline 1.

3. **StandardScaler:** Same as Pipeline 1.

4. **ANOVA Feature Selection (k=100):** Selects the top 100 features by F-statistic.

5. **PCA (n=5):** Reduces from 100 to 5 principal components.

6. **SGDClassifier (modified Huber loss):** Stochastic Gradient Descent with modified Huber loss — a smooth approximation to hinge loss that also yields probability estimates. Parameters: `class_weight="balanced"`, `max_iter=5000`, `random_state=42`.

### Performance Profile
- High sensitivity (82.8%) — detects 24 of 29 preterm cases (only 5 missed)
- Lower specificity (60.8%) — trades off more false positives (20)
- Best F1 score and AUC across the entire sweep
- Clinically relevant: prioritizes preterm detection over overall accuracy

---

## Cross-Validation Methodology

- **5-fold Stratified K-Fold** (`shuffle=True`, `random_state=42`), preserving class proportions
- **All preprocessing fitted inside each fold** (variance filter, scaler, ANOVA, PCA) — no data leakage
- Metrics computed from aggregated out-of-fold predictions across all 80 samples
- AUC computed from `predict_proba` values for both pipelines (LDA and SGD modified Huber both provide `predict_proba`; it takes priority over `decision_function` in the sweep code)
