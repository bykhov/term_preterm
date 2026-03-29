# Early Preterm Birth Prediction from Cervical Ultrasound Using Deep Texture Features

Supplementary code and data.

## Overview

Binary classification of cropped transvaginal ultrasound images for preterm birth prediction. The pipeline extracts deep texture features from RadImageNet-pretrained backbones (ResNet50, DenseNet121, InceptionV3), reduces dimensionality with PCA, and classifies using a configurable classifier (default: SVM).

**Default pipeline:** ResNet50 avgpool (2048-d) &rarr; variance filter &rarr; StandardScaler &rarr; ANOVA(200) &rarr; PCA(5) &rarr; LDA &rarr; 5-Fold CV

## Dataset

80 cervical ultrasound images acquired at 14-16 weeks of gestation:
- `Data/term/` &mdash; 51 term birth images
- `Data/pre-term/` &mdash; 29 preterm birth images
- `Data/clinical_data.csv` &mdash; 10 clinical features per subject

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download RadImageNet weights (see `RadImageNet_pytorch/README.md`) and place them in the `RadImageNet_pytorch/` directory. Only the model(s) you plan to use need to be present:
   - `ResNet50.pt` (default)
   - `DenseNet121.pt`
   - `InceptionV3.pt`

   Prepopulated feature caches (`features_*.pkl`) are included so feature extraction can be skipped on first run.

## Usage

### Image classification pipelines

```bash
# Best accuracy: ResNet50 + ANOVA(200)->PCA(5) + LDA
python best_image/pipeline1_lda.py

# Alternative: DenseNet121 + ANOVA(100)->PCA(5) + SGD
python best_image/pipeline2_sgd.py
```

### Clinical classification pipelines

```bash
# Best accuracy: StandardScaler + LDA
python best_clinical/clinical1_lda.py

# Best AUC: StandardScaler + LogReg(C=0.1)
python best_clinical/clinical2_logreg.py
```

### Classifier sweeps

```bash
# Image feature sweep: 28 classifiers x 3 models x 6 PCA dims
python sweep_image/sweep_image.py

# Clinical feature sweep
python sweep_clinical/clinical_sweep.py
```

### Fusion

```bash
# Early fusion (image + clinical features)
python early_fusion/early_fusion.py

# Late fusion (combining classifier outputs)
python late_fusion/contingency_analysis.py
```

### Explainability (Grad-CAM)

```bash
# Grad-CAM heatmaps for pipeline 1 (ResNet50 + LDA)
python explainable/run_gradcam_pipeline1.py

# Grad-CAM heatmaps for pipeline 2 (DenseNet121 + SGD)
python explainable/run_gradcam_pipeline2.py
```

## Repository Structure

```
.
├── pipeline.py                          # Computation engine (feature extraction, caching, CV, metrics)
├── requirements.txt
├── Data/
│   ├── term/                            # 51 term birth ultrasound images
│   ├── pre-term/                        # 29 preterm birth ultrasound images
│   └── clinical_data.csv                # Clinical features
├── RadImageNet_pytorch/
│   ├── README.md                        # Download instructions for model weights
│   └── features_*.pkl                   # Prepopulated feature caches
├── best_image/
│   ├── pipeline1_lda.py                 # Best accuracy image pipeline
│   └── pipeline2_sgd.py                 # Alternative image pipeline
├── best_clinical/
│   ├── clinical1_lda.py                 # Best accuracy clinical pipeline
│   └── clinical2_logreg.py              # Best AUC clinical pipeline
├── sweep_image/
│   └── sweep_image.py                   # Image classifier sweep
├── sweep_clinical/
│   └── clinical_sweep.py                # Clinical classifier sweep
├── early_fusion/
│   └── early_fusion.py                  # Early fusion pipeline
├── late_fusion/
│   └── contingency_analysis.py          # Late fusion analysis
└── explainable/
    ├── xai_utils.py                     # Shared explainability utilities
    ├── run_gradcam_pipeline1.py          # Grad-CAM for pipeline 1
    └── run_gradcam_pipeline2.py          # Grad-CAM for pipeline 2
```

## License

This project is provided for research purposes.
