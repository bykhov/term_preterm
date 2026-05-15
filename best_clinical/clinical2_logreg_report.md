# Clinical Pipeline #2: Clinical → StandardScaler + LogReg(C=0.1)

**Dataset:** 80 patients (51 term, 29 preterm)

## Pipeline
10 clinical features → StandardScaler → LogisticRegression(C=0.1, balanced)

## Clinical Features
Age, Pregnancy number, Past births, Previous preterm births,
Previous cesarean section, Cervical length (cm), Smoker,
Pre-pregnancy diseases, Pregnancy diseases, Treatment to stop labor

## 5-Fold CV Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.750 | (0.655–0.845) |
| Sensitivity | 0.621 | (0.351–0.889) |
| Specificity | 0.824 | (0.770–0.878) |
| F1 | 0.643 | (0.438–0.817) |
| AUC | 0.762 | (0.645–0.871) |
| TP | 18 | |
| TN | 42 | |
| FP | 9 | |
| FN | 11 | |

## Notes
- LogisticRegression provides calibrated posterior probabilities via `predict_proba`
- Uses L2 penalty with C=0.1 and balanced class weights
- Proba saved to clinical2_logreg_proba.csv for late fusion analysis
