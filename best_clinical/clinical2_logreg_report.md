# Clinical Pipeline #2: Clinical → StandardScaler + LogReg(C=0.1)

**Dataset:** 80 patients (51 term, 29 preterm)

## Pipeline
10 clinical features → StandardScaler → LogisticRegression(C=0.1, balanced)

## Clinical Features
Age, Pregnancy number, Past births, Previous preterm births,
Previous cesarean section, Cervical length (cm), Smoker,
Pre-pregnancy diseases, Pregnancy diseases, Treatment to stop labor

## 5-Fold CV Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.750 |
| Sensitivity | 0.621 |
| Specificity | 0.824 |
| F1 | 0.643 |
| AUC | 0.762 |
| TP | 18 |
| TN | 42 |
| FP | 9 |
| FN | 11 |

## Notes
- LogisticRegression provides calibrated posterior probabilities via `predict_proba`
- Uses L2 penalty with C=0.1 and balanced class weights
- Proba saved to clinical2_logreg_proba.csv for late fusion analysis
