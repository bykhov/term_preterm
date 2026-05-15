# Image Sweep — Top Results

**Sweep:** 3 CNN models × 44 classifiers × 26 feature selection configs = 3,432 combinations
**Evaluation:** 5-fold stratified cross-validation, 80 samples (51 term, 29 preterm)

## Top 5 Configurations

| Rank | Model | Classifier | Feature Selection | Accuracy | Sensitivity | Specificity | F1 | AUC |
|------|-------|-----------|-------------------|----------|-------------|-------------|----|-----|
| 1 | ResNet50 | LDA | ANOVA200→PCA5 | 0.763 | 0.586 | 0.863 | 0.642 | 0.656 |
| 2 | ResNet50 | SVM (RBF, C=1) | ANOVA10 | 0.750 | 0.655 | 0.804 | 0.655 | 0.682 |
| 3 | ResNet50 | MLP (64, tanh) | ANOVA200→PCA10 | 0.750 | 0.621 | 0.824 | 0.643 | 0.687 |
| 4 | DenseNet121 | GBM (200 trees) | ANOVA100→PCA20 | 0.750 | 0.552 | 0.863 | 0.615 | 0.719 |
| 5 | ResNet50 | MLP (128) | ANOVA200→PCA10 | 0.750 | 0.552 | 0.863 | 0.615 | 0.662 |
