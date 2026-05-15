# Sweep Methods Report — Image-Based Classifier Grid Search

## Overview

This sweep performs an exhaustive grid search over **3 CNN backbone models × 44 classifiers × 26 feature selection configurations = 3,432 combinations**. Each combination is evaluated using 5-fold stratified cross-validation on 80 cropped transvaginal cervical ultrasound images (51 term, 29 preterm births; acquired at 14–16 weeks gestation).

Results are appended incrementally to `sweep_results.csv`, allowing the sweep to be interrupted and resumed.

---

## 1. Feature Extraction

Deep features are extracted from three CNN backbones pretrained on RadImageNet (a medical imaging dataset):

| Model | Output Dim | Input Size | Architecture |
|-------|-----------|------------|--------------|
| ResNet50 | 2048 | 224×224 | Residual blocks → AdaptiveAvgPool → Flatten |
| DenseNet121 | 1024 | 224×224 | Dense blocks → ReLU → AdaptiveAvgPool → Flatten |
| InceptionV3 | 2048 | 299×299 | Inception modules → AdaptiveAvgPool → Flatten |

- **Preprocessing:** Resize to model input size, convert to tensor, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- **Caching:** Extracted features are pickled with a file manifest for cache invalidation when the dataset changes.

---

## 2. Feature Selection Configurations (26 total)

### 2.1 PCA Only (6 configs)
Dimensionality reduction via Principal Component Analysis.

| Config | Components |
|--------|-----------|
| PCA5 | 5 |
| PCA10 | 10 |
| PCA15 | 15 |
| PCA20 | 20 |
| PCA30 | 30 |
| PCA50 | 50 |

### 2.2 ANOVA Only (5 configs)
Univariate feature selection using F-test (ANOVA) via `SelectKBest(f_classif, k)`.

| Config | k (features selected) |
|--------|----------------------|
| ANOVA10 | 10 |
| ANOVA20 | 20 |
| ANOVA50 | 50 |
| ANOVA100 | 100 |
| ANOVA200 | 200 |

### 2.3 ANOVA → PCA (15 configs)
Two-stage selection: ANOVA first reduces to k features, then PCA further reduces dimensionality.

| ANOVA k \ PCA n | 5 | 10 | 15 | 20 | 30 |
|-----------------|---|----|----|----|----|
| 50 | ANOVA50_PCA5 | ANOVA50_PCA10 | ANOVA50_PCA15 | ANOVA50_PCA20 | ANOVA50_PCA30 |
| 100 | ANOVA100_PCA5 | ANOVA100_PCA10 | ANOVA100_PCA15 | ANOVA100_PCA20 | ANOVA100_PCA30 |
| 200 | ANOVA200_PCA5 | ANOVA200_PCA10 | ANOVA200_PCA15 | ANOVA200_PCA20 | ANOVA200_PCA30 |

---

## 3. Classifiers (44 total)

### 3.1 Support Vector Machines (12 variants)

**RBF Kernel (6):** Regularization parameter C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 10.0}. All use `gamma="scale"`, `class_weight="balanced"`.

**Linear Kernel (2):** C ∈ {0.1, 1.0}, `class_weight="balanced"`.

**Polynomial Kernel (2):** Degree 2 and 3, C=0.1, `gamma="scale"`, `class_weight="balanced"`.

**Calibrated SVM (1):** RBF SVM (C=0.1) wrapped in `CalibratedClassifierCV(cv=3, method="sigmoid")` to produce probability estimates. `n_jobs=7`.

**Bagging SVM (1):** 10 bagged RBF SVMs (C=0.1) via `BaggingClassifier`. `n_jobs=7`.

### 3.2 Logistic Regression (6 variants)

| Name | Penalty | Solver | C | Notes |
|------|---------|--------|---|-------|
| LogReg_C0.1 | L2 | lbfgs | 0.1 | |
| LogReg_C1 | L2 | lbfgs | 1.0 | |
| LogReg_C10 | L2 | lbfgs | 10.0 | |
| LogReg_l1_C0.1 | L1 | saga | 0.1 | Sparse solution |
| LogReg_l1_C1 | L1 | saga | 1.0 | Sparse solution |
| LogReg_elasticnet_C0.1 | Elastic Net | saga | 0.1 | l1_ratio=0.5 |

All use `class_weight="balanced"`, `max_iter=5000`, `n_jobs=7`.

### 3.3 Naive Bayes (1)

**GaussianNB:** Gaussian Naive Bayes with default parameters. Assumes features are normally distributed within each class.

### 3.4 Discriminant Analysis (2)

- **LDA (Linear Discriminant Analysis):** Projects data onto a lower-dimensional space maximizing class separability.
- **QDA (Quadratic Discriminant Analysis):** Fits a separate covariance matrix per class, allowing quadratic decision boundaries.

### 3.5 Ridge Classifier (3 variants)

Linear classifier using L2-regularized least squares. Alpha ∈ {0.1, 1.0, 10.0}. All use `class_weight="balanced"`.

### 3.6 Stochastic Gradient Descent (4 variants)

SGDClassifier with different loss functions, all with `class_weight="balanced"`, `max_iter=5000`:

| Name | Loss | Decision Boundary |
|------|------|-------------------|
| SGD_hinge | Hinge (linear SVM) | Linear |
| SGD_log | Log loss (logistic regression) | Linear |
| SGD_modified_huber | Modified Huber | Linear |
| SGD_perceptron | Perceptron | Linear |

### 3.7 Passive Aggressive Classifier (1)

Online learning algorithm with C=0.1, `class_weight="balanced"`, `max_iter=5000`.

### 3.8 Ensemble Methods (6 variants)

| Name | Method | n_estimators | n_jobs |
|------|--------|-------------|--------|
| RF_100 | Random Forest | 100 | 7 |
| RF_500 | Random Forest | 500 | 7 |
| ExtraTrees_100 | Extra Trees | 100 | 7 |
| GBM_100 | Gradient Boosting | 100 | — |
| GBM_200 | Gradient Boosting | 200 | — |
| AdaBoost_100 | AdaBoost | 100 | — |

Random Forest and Extra Trees use `class_weight="balanced"`.

### 3.9 Multi-Layer Perceptron (8 variants)

| Name | Architecture | Activation | Alpha (L2) |
|------|-------------|-----------|-------------|
| MLP_32 | (32,) | ReLU | 0.01 |
| MLP_64 | (64,) | ReLU | 0.0001 |
| MLP_128 | (128,) | ReLU | 0.0001 |
| MLP_64_reg | (64,) | ReLU | 0.1 |
| MLP_128_64 | (128, 64) | ReLU | 0.0001 |
| MLP_128_64_reg | (128, 64) | ReLU | 0.1 |
| MLP_32_16 | (32, 16) | ReLU | 0.01 |
| MLP_64_tanh | (64,) | Tanh | 0.01 |

All use `max_iter=3000`, Adam optimizer (default).

### 3.10 Nearest Centroid (1)

Classifies samples by proximity to the centroid of each class. No hyperparameters.

---

## 4. Cross-Validation Methodology

- **Strategy:** 5-fold Stratified K-Fold (`shuffle=True`, `random_state=42`), preserving class proportions in each fold.
- **Variance Filter:** Applied inside each fold — features with training-set variance ≤ 1e-10 are removed. This prevents data leakage.
- **Pipeline per fold:** `StandardScaler → [ANOVA] → [PCA] → Classifier`, fitted on training data and applied to test data.
- **Data leakage prevention:** Scaler, ANOVA, and PCA are fitted exclusively on training folds.

---

## 5. Evaluation Metrics

For each combination, the following metrics are computed from the aggregated out-of-fold predictions:

| Metric | Definition |
|--------|-----------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Sensitivity (Recall)** | TP / (TP + FN) — ability to detect preterm births |
| **Specificity** | TN / (TN + FP) — ability to correctly identify term births |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **AUC** | Area under the ROC curve, computed from predicted probabilities (preferred, for classifiers with `predict_proba`) or decision values (fallback, for classifiers with `decision_function`). N/A for classifiers providing neither. |
| **Confusion Matrix** | TP, TN, FP, FN counts |

---

## 6. Parallelization

Classifiers that support parallel execution use `n_jobs=7`:
- RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
- LogisticRegression (all variants)
- CalibratedClassifierCV

Classifiers without native `n_jobs` support (SVM, GBM, AdaBoost, MLP, SGD, etc.) run single-threaded.

---

## 7. Output

Results are saved to `sweep_results.csv` in the same directory as the script, sorted by accuracy (descending), F1, and AUC. Each row contains: model, classifier, fs_config, accuracy, sensitivity, specificity, f1, auc, TP, TN, FP, FN.
