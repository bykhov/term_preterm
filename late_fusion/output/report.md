# Contingency Analysis Report

Comparison and combining analysis for 4 Image x Clinical pipeline combinations.

| Combo | Image Pipeline | Clinical Pipeline |
|-------|---------------|-------------------|
| 1 | Image#1 (ResNet50+LDA) | Clinical#1 (LDA) |
| 2 | Image#1 (ResNet50+LDA) | Clinical#2 (LogReg C=0.1) |
| 3 | Image#2 (DenseNet121+SGD) | Clinical#1 (LDA) |
| 4 | Image#2 (DenseNet121+SGD) | Clinical#2 (LogReg C=0.1) |

---

## COMBO1: Image#1 (ResNet50+LDA) x Clinical#1 (LDA)

### Individual Performance

| Pipeline | Accuracy | Sensitivity | Specificity | F1 | AUC |
|----------|----------|-------------|-------------|-----|-----|
| Image#1 (ResNet50+LDA) | 0.762 | 0.586 | 0.863 | 0.642 | 0.656 |
| Clinical#1 (LDA) | 0.787 | 0.552 | 0.922 | 0.653 | 0.753 |

### Contingency Table

|  | Clinical=Preterm | Clinical=Term |
|--|-----------------|---------------|
| Image=Preterm | 10 | 14 |
| Image=Term | 10 | 46 |

### Comparison Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Disagreement | 0.3000 (24/80 samples) | |
| McNemar b (img wrong, clin right) | 13 | |
| McNemar c (img right, clin wrong) | 11 | |
| McNemar chi2 | 0.0417 | Not significant (p=0.8383) |
| Cohen's Kappa | 0.2500 | fair agreement |
| Yule's Q | 0.5333 | moderate dependence |

### Combining Results

| Method | Accuracy | Sensitivity | Specificity | F1 | AUC |
|--------|----------|-------------|-------------|-----|-----|
| Image only | 0.762 | 0.586 | 0.863 | 0.642 | 0.656 |
| Clinical only | 0.787 | 0.552 | 0.922 | 0.653 | 0.753 |
| LLR | 0.738 | 0.414 | 0.922 | 0.533 | 0.741 |
| Product rule | 0.750 | 0.552 | 0.863 | 0.615 | 0.741 |
| Majority vote | 0.762 | 0.517 | 0.902 | 0.612 | N/A |
| Soft vote | 0.762 | 0.517 | 0.902 | 0.612 | 0.774 |

---

## COMBO2: Image#1 (ResNet50+LDA) x Clinical#2 (LogReg)

### Individual Performance

| Pipeline | Accuracy | Sensitivity | Specificity | F1 | AUC |
|----------|----------|-------------|-------------|-----|-----|
| Image#1 (ResNet50+LDA) | 0.762 | 0.586 | 0.863 | 0.642 | 0.656 |
| Clinical#2 (LogReg) | 0.750 | 0.621 | 0.824 | 0.643 | 0.762 |

### Contingency Table

|  | Clinical=Preterm | Clinical=Term |
|--|-----------------|---------------|
| Image=Preterm | 15 | 9 |
| Image=Term | 12 | 44 |

### Comparison Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Disagreement | 0.2625 (21/80 samples) | |
| McNemar b (img wrong, clin right) | 10 | |
| McNemar c (img right, clin wrong) | 11 | |
| McNemar chi2 | 0.0000 | Not significant (p=1.0000) |
| Cohen's Kappa | 0.3966 | fair agreement |
| Yule's Q | 0.7188 | moderate dependence |

### Combining Results

| Method | Accuracy | Sensitivity | Specificity | F1 | AUC |
|--------|----------|-------------|-------------|-----|-----|
| Image only | 0.762 | 0.586 | 0.863 | 0.642 | 0.656 |
| Clinical only | 0.750 | 0.621 | 0.824 | 0.643 | 0.762 |
| LLR | 0.700 | 0.379 | 0.882 | 0.478 | 0.705 |
| Product rule | 0.750 | 0.621 | 0.824 | 0.643 | 0.705 |
| Majority vote | 0.762 | 0.552 | 0.882 | 0.627 | N/A |
| Soft vote | 0.762 | 0.552 | 0.882 | 0.627 | 0.750 |

---

## COMBO3: Image#2 (DenseNet121+SGD) x Clinical#1 (LDA)

### Individual Performance

| Pipeline | Accuracy | Sensitivity | Specificity | F1 | AUC |
|----------|----------|-------------|-------------|-----|-----|
| Image#2 (DenseNet121+SGD) | 0.688 | 0.862 | 0.588 | 0.667 | 0.725 |
| Clinical#1 (LDA) | 0.787 | 0.552 | 0.922 | 0.653 | 0.753 |

### Contingency Table

|  | Clinical=Preterm | Clinical=Term |
|--|-----------------|---------------|
| Image=Preterm | 14 | 32 |
| Image=Term | 6 | 28 |

### Comparison Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Disagreement | 0.4750 (38/80 samples) | |
| McNemar b (img wrong, clin right) | 23 | |
| McNemar c (img right, clin wrong) | 15 | |
| McNemar chi2 | 1.2895 | Not significant (p=0.2561) |
| Cohen's Kappa | 0.1163 | slight agreement |
| Yule's Q | 0.3425 | moderate dependence |

### Combining Results

| Method | Accuracy | Sensitivity | Specificity | F1 | AUC |
|--------|----------|-------------|-------------|-----|-----|
| Image only | 0.688 | 0.862 | 0.588 | 0.667 | 0.725 |
| Clinical only | 0.787 | 0.552 | 0.922 | 0.653 | 0.753 |
| LLR | 0.688 | 0.862 | 0.588 | 0.667 | 0.838 |
| Product rule | 0.688 | 0.862 | 0.588 | 0.667 | 0.838 |
| Majority vote | 0.688 | 0.862 | 0.588 | 0.667 | N/A |
| Soft vote | 0.688 | 0.862 | 0.588 | 0.667 | 0.838 |

---

## COMBO4: Image#2 (DenseNet121+SGD) x Clinical#2 (LogReg)

### Individual Performance

| Pipeline | Accuracy | Sensitivity | Specificity | F1 | AUC |
|----------|----------|-------------|-------------|-----|-----|
| Image#2 (DenseNet121+SGD) | 0.688 | 0.862 | 0.588 | 0.667 | 0.725 |
| Clinical#2 (LogReg) | 0.750 | 0.621 | 0.824 | 0.643 | 0.762 |

### Contingency Table

|  | Clinical=Preterm | Clinical=Term |
|--|-----------------|---------------|
| Image=Preterm | 21 | 25 |
| Image=Term | 6 | 28 |

### Comparison Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Disagreement | 0.3875 (31/80 samples) | |
| McNemar b (img wrong, clin right) | 18 | |
| McNemar c (img right, clin wrong) | 13 | |
| McNemar chi2 | 0.5161 | Not significant (p=0.4725) |
| Cohen's Kappa | 0.2610 | fair agreement |
| Yule's Q | 0.5935 | moderate dependence |

### Combining Results

| Method | Accuracy | Sensitivity | Specificity | F1 | AUC |
|--------|----------|-------------|-------------|-----|-----|
| Image only | 0.688 | 0.862 | 0.588 | 0.667 | 0.725 |
| Clinical only | 0.750 | 0.621 | 0.824 | 0.643 | 0.762 |
| LLR | 0.688 | 0.862 | 0.588 | 0.667 | 0.835 |
| Product rule | 0.688 | 0.862 | 0.588 | 0.667 | 0.835 |
| Majority vote | 0.688 | 0.862 | 0.588 | 0.667 | N/A |
| Soft vote | 0.688 | 0.862 | 0.588 | 0.667 | 0.835 |
