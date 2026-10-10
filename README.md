# Prediction of Preterm Birth by Combining Deep Texture Features From Transvaginal Ultrasound and Clinical Data

Source code and anonymized dataset for the article of the same title.

## Overview

A pre-specified model, fixed before it was evaluated and involving no data-driven selection, predicts preterm birth from a cropped transvaginal ultrasound image of the cervix and 10 routine clinical variables:

- **Image stream:** four frozen image representations (handcrafted texture, mid-level RadImageNet ResNet50 features, BiomedCLIP, RadImageNet DenseNet121 final-layer features), each followed by variance filter &rarr; StandardScaler &rarr; PCA(10) &rarr; logistic regression (L2, C = 0.1, balanced). The stream is the mean of the four probabilities.
- **Clinical stream:** median imputation &rarr; StandardScaler &rarr; LDA.
- **Late fusion (primary):** soft vote of the two streams.
- **Early fusion (secondary):** for each representation, its 10 principal components concatenated with the clinical variables &rarr; StandardScaler &rarr; logistic regression; the mean of the four models.

All models are evaluated with 20 repetitions (seeds 42-61) of stratified 5-fold cross-validation.

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

2. Download the RadImageNet weights (see `RadImageNet_pytorch/README.md`) and place `ResNet50.pt` and `DenseNet121.pt` in the `RadImageNet_pytorch/` directory. The weights are needed only to re-extract features and for Grad-CAM: the extracted features are included (`RadImageNet_pytorch/features_*.pkl` and `results/features_new/*.pkl`), so the analyses run without them.

3. `features_new.py` downloads the BiomedCLIP weights from Hugging Face on first use.

## Usage

Run the scripts from the repository root, in this order:

```bash
python features_new.py        # handcrafted, mid-level ResNet50 and BiomedCLIP features
python features_eval.py       # image representations, image ensemble, clinical model, late fusion
python ef_selection_free.py   # early fusion
python statistical_tests.py   # cervical-length ablation, paired tests, complementarity, permutation test
python robustness.py          # robustness analysis over 405 fusion models
python gradcam.py             # Grad-CAM for the DenseNet121 member
python table_clinical.py      # maternal clinical characteristics
```

All outputs are written to `results/`.

## Repository Structure

```
.
├── common.py                # Paths, data loading, fold-level models, fusion rule
├── stats_utils.py           # DeLong, McNemar, Holm, corrected resampled t-test, agreement measures
├── pipeline.py              # RadImageNet feature extraction and caching
├── features_new.py          # Handcrafted, mid-level ResNet50 and BiomedCLIP features
├── features_eval.py         # Image representations, image ensemble and late fusion
├── ef_selection_free.py     # Early fusion
├── statistical_tests.py     # Ablation, paired and permutation tests
├── robustness.py            # Robustness analysis
├── gradcam.py               # Grad-CAM
├── table_clinical.py        # Clinical characteristics table
├── explainable/
│   └── xai_utils.py         # Differentiable model head and heatmap utilities
├── requirements.txt
├── Data/
│   ├── term/                # 51 term birth ultrasound images
│   ├── pre-term/            # 29 preterm birth ultrasound images
│   └── clinical_data.csv    # Clinical features
├── RadImageNet_pytorch/
│   ├── README.md            # Download instructions for model weights
│   └── features_*.pkl       # Extracted ResNet50 and DenseNet121 features
└── results/                 # Outputs: tables (CSV), figures, extracted features, Grad-CAM maps
```

## License

This project is provided for research purposes.
