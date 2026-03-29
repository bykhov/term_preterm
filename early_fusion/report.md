# Early Fusion Sweep Results

**Total combinations evaluated:** 192
**Models:** ResNet50, DenseNet121
**Classifiers:** 12
**Feature selection configs:** 8

## Top 15 Overall

| Rank | Model | Classifier | FS Config | Acc. | Sens. | Spec. | F1 | AUC |
|------|-------|-----------|-----------|------|-------|-------|----|-----|
| 1 | DenseNet121 | LDA | ANOVA10 | 0.812 | 0.621 | 0.922 | 0.706 | 0.803 |
| 2 | DenseNet121 | SVM_linear_C0.1 | ANOVA200_PCA5 | 0.800 | 0.483 | 0.980 | 0.636 | 0.744 |
| 3 | DenseNet121 | SVM_linear_C0.1 | ANOVA100_PCA5 | 0.800 | 0.483 | 0.980 | 0.636 | 0.742 |
| 4 | DenseNet121 | LogReg_C10 | ANOVA10 | 0.787 | 0.690 | 0.843 | 0.702 | 0.784 |
| 5 | ResNet50 | LogReg_C0.1 | ANOVA200_PCA10 | 0.787 | 0.690 | 0.843 | 0.702 | 0.741 |
| 6 | DenseNet121 | LogReg_C1 | ANOVA10 | 0.775 | 0.621 | 0.863 | 0.667 | 0.762 |
| 7 | ResNet50 | LogReg_C1 | ANOVA100_PCA10 | 0.762 | 0.690 | 0.804 | 0.678 | 0.744 |
| 8 | ResNet50 | LogReg_C0.1 | ANOVA100_PCA10 | 0.762 | 0.655 | 0.824 | 0.667 | 0.742 |
| 9 | ResNet50 | LogReg_C0.1 | PCA5 | 0.762 | 0.621 | 0.843 | 0.655 | 0.770 |
| 10 | ResNet50 | SVM_C0.1 | PCA10 | 0.762 | 0.621 | 0.843 | 0.655 | 0.761 |
| 11 | ResNet50 | LDA | ANOVA100_PCA10 | 0.762 | 0.621 | 0.843 | 0.655 | 0.740 |
| 12 | ResNet50 | SVM_C0.1 | ANOVA100_PCA10 | 0.762 | 0.586 | 0.863 | 0.642 | 0.748 |
| 13 | DenseNet121 | SVM_linear_C1 | ANOVA20 | 0.762 | 0.448 | 0.941 | 0.578 | 0.788 |
| 14 | ResNet50 | LogReg_C1 | PCA5 | 0.750 | 0.690 | 0.784 | 0.667 | 0.770 |
| 15 | ResNet50 | LogReg_C10 | ANOVA100_PCA10 | 0.750 | 0.690 | 0.784 | 0.667 | 0.746 |

## Best Per Model

- **ResNet50:** LogReg_C0.1 + ANOVA200_PCA10 → acc=0.787, sens=0.690, spec=0.843, F1=0.702, AUC=0.741
- **DenseNet121:** LDA + ANOVA10 → acc=0.812, sens=0.621, spec=0.922, F1=0.706, AUC=0.803

## Best Per Classifier

- **LDA:** DenseNet121 + ANOVA10 → acc=0.812, F1=0.706, AUC=0.803
- **LogReg_C0.1:** ResNet50 + ANOVA200_PCA10 → acc=0.787, F1=0.702, AUC=0.741
- **LogReg_C1:** DenseNet121 + ANOVA10 → acc=0.775, F1=0.667, AUC=0.762
- **LogReg_C10:** DenseNet121 + ANOVA10 → acc=0.787, F1=0.702, AUC=0.784
- **SVM_C0.1:** ResNet50 + PCA10 → acc=0.762, F1=0.655, AUC=0.761
- **SVM_C1:** ResNet50 + ANOVA100_PCA10 → acc=0.725, F1=0.593, AUC=0.734
- **SVM_linear_C0.1:** DenseNet121 + ANOVA200_PCA5 → acc=0.800, F1=0.636, AUC=0.744
- **SVM_linear_C1:** DenseNet121 + ANOVA20 → acc=0.762, F1=0.578, AUC=0.788
- **GBM_100:** DenseNet121 + ANOVA20 → acc=0.713, F1=0.566, AUC=0.721
- **GBM_200:** DenseNet121 + ANOVA10 → acc=0.713, F1=0.582, AUC=0.712
- **RF_100:** ResNet50 + ANOVA10 → acc=0.738, F1=0.571, AUC=0.704
- **AdaBoost_100:** ResNet50 + ANOVA20 → acc=0.725, F1=0.577, AUC=0.685

## Best Per Feature Selection Config

- **ANOVA100_PCA5:** DenseNet121 + SVM_linear_C0.1 → acc=0.800, F1=0.636, AUC=0.742
- **ANOVA100_PCA10:** ResNet50 + LogReg_C1 → acc=0.762, F1=0.678, AUC=0.744
- **ANOVA200_PCA5:** DenseNet121 + SVM_linear_C0.1 → acc=0.800, F1=0.636, AUC=0.744
- **ANOVA200_PCA10:** ResNet50 + LogReg_C0.1 → acc=0.787, F1=0.702, AUC=0.741
- **ANOVA10:** DenseNet121 + LDA → acc=0.812, F1=0.706, AUC=0.803
- **ANOVA20:** DenseNet121 + SVM_linear_C1 → acc=0.762, F1=0.578, AUC=0.788
- **PCA5:** ResNet50 + LogReg_C0.1 → acc=0.762, F1=0.655, AUC=0.770
- **PCA10:** ResNet50 + SVM_C0.1 → acc=0.762, F1=0.655, AUC=0.761

## Comparison vs Baselines

| Method | Accuracy | F1 | AUC |
|--------|----------|----|-----|
| **Best early fusion** (DenseNet121 + LDA + ANOVA10) | 0.812 | 0.706 | 0.803 |
| Best image-only (ResNet50+LDA) | 0.763 | 0.642 | 0.656 |
| Best clinical-only (LDA) | 0.788 | 0.653 | 0.753 |
| Best late fusion (Linear w=0.10) | 0.800 | 0.652 | 0.767 |
