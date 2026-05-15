# Sweep Methods Report — Clinical Data Classifier Grid Search

## Overview

This sweep performs a grid search over **36 classifiers** using 10 clinical tabular features. Each classifier is evaluated using 5-fold stratified cross-validation on 80 patients (51 term, 29 preterm births; ultrasound acquired at 14–16 weeks gestation).

Results are appended incrementally to `clinical_sweep_results.csv`, allowing the sweep to be interrupted and resumed.

---

## 1. Clinical Features

Data source: `Data_meta/clinical_data.csv`

| Feature | Type | Description |
|---------|------|-------------|
| Age | Continuous | Maternal age (years) |
| Pregnancy number | Discrete | Total number of pregnancies |
| Past births | Discrete | Number of previous births |
| Previous preterm births | Discrete | Count of previous preterm births |
| Previous cesarean section | Binary (0/1) | History of cesarean delivery |
| Cervical length (cm) | Continuous | Measured at 14–16 weeks |
| Smoker | Binary (0/1) | Smoking status |
| Pre-pregnancy diseases | Binary (0/1) | Diseases prior to pregnancy |
| Pregnancy diseases | Binary (0/1) | Diseases during pregnancy |
| Treatment to stop labor | Binary (0/1) | Retrospective: treatment administered |

---

## 2. Pipeline

```
StandardScaler → Classifier
```

No variance filter (all 10 features have meaningful variance), no feature selection or PCA (only 10 features — dimensionality reduction unnecessary).

---

## 3. Classifiers (36 total)

### 3.1 Support Vector Machines (8 variants)

**RBF Kernel (4):** C ∈ {0.1, 0.5, 1.0, 10.0}. All use `gamma="scale"`, `class_weight="balanced"`.

**Linear Kernel (2):** C ∈ {0.1, 1.0}, `class_weight="balanced"`.

**Polynomial Kernel (2):** Degree 2 and 3, C=0.1, `gamma="scale"`, `class_weight="balanced"`.

No calibration wrapper — raw SVM without `predict_proba`.

### 3.2 Logistic Regression (6 variants)

| Name | Penalty | Solver | C |
|------|---------|--------|---|
| LogReg_C0.1 | L2 | lbfgs | 0.1 |
| LogReg_C1 | L2 | lbfgs | 1.0 |
| LogReg_C10 | L2 | lbfgs | 10.0 |
| LogReg_l1_C0.1 | L1 | saga | 0.1 |
| LogReg_l1_C1 | L1 | saga | 1.0 |
| LogReg_elasticnet_C0.1 | Elastic Net | saga | 0.1 |

All use `class_weight="balanced"`, `max_iter=5000`.

### 3.3 Naive Bayes (1)

**GaussianNB:** Default parameters.

### 3.4 Discriminant Analysis (2)

- **LDA (Linear Discriminant Analysis)**
- **QDA (Quadratic Discriminant Analysis)**

### 3.5 Ridge Classifier (3 variants)

Alpha ∈ {0.1, 1.0, 10.0}. All use `class_weight="balanced"`. No `predict_proba`; AUC from `decision_function`.

### 3.6 Stochastic Gradient Descent (4 variants)

All use `class_weight="balanced"`, `max_iter=5000`:

| Name | Loss |
|------|------|
| SGD_hinge | Hinge (linear SVM) |
| SGD_log | Log loss (logistic regression) |
| SGD_modified_huber | Modified Huber |
| SGD_perceptron | Perceptron |

### 3.7 Passive Aggressive Classifier (1)

C=0.1, `class_weight="balanced"`, `max_iter=5000`.

### 3.8 Tree-based / Ensemble (7 variants)

| Name | Method | n_estimators |
|------|--------|-------------|
| DecisionTree | Decision Tree | — |
| RF_100 | Random Forest | 100 |
| RF_500 | Random Forest | 500 |
| ExtraTrees_100 | Extra Trees | 100 |
| GBM_100 | Gradient Boosting | 100 |
| GBM_200 | Gradient Boosting | 200 |
| AdaBoost_100 | AdaBoost | 100 |

DecisionTree, RF, and ExtraTrees use `class_weight="balanced"`.

### 3.9 K-Nearest Neighbors (3 variants)

k ∈ {3, 5, 7}. Default distance metric (Euclidean).

### 3.10 Nearest Centroid (1)

Classifies by proximity to class centroid. No hyperparameters.

---

## 4. Cross-Validation Methodology

- **Strategy:** 5-fold Stratified K-Fold (`shuffle=True`, `random_state=42`), preserving class proportions.
- **Pipeline per fold:** `StandardScaler → Classifier`, fitted on training data and applied to test data.
- **Data leakage prevention:** Scaler fitted exclusively on training folds.

---

## 5. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Sensitivity (Recall)** | TP / (TP + FN) — ability to detect preterm births |
| **Specificity** | TN / (TN + FP) — ability to correctly identify term births |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **AUC** | Area under the ROC curve, computed from predicted probabilities (preferred, for classifiers with `predict_proba`) or decision values (fallback, for classifiers with `decision_function`). N/A for classifiers providing neither. |
| **Confusion Matrix** | TP, TN, FP, FN counts |

---

## 6. Output

Results are saved to `clinical_sweep_results.csv`, sorted by accuracy (descending), F1, and AUC. Each row contains: classifier, accuracy, sensitivity, specificity, f1, auc, TP, TN, FP, FN.
