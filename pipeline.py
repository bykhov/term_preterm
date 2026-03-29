"""
Classification Pipeline — Computation Engine
=============================================
Feature extraction (with pickle caching), 5-fold stratified CV evaluation,
and metrics for binary classification of cropped ultrasound images using
RadImageNet CNN features.

Pipeline: CNN avgpool → variance filter → StandardScaler → PCA → classifier → 5-Fold CV
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
WEIGHTS_DIR = BASE_DIR / "RadImageNet_pytorch"
OUTPUT_DIR = BASE_DIR / "output"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────────
# Model configurations
# ──────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ResNet50": {
        "factory": models.resnet50,
        "n_remove": 2,
        "dim": 2048,
        "extra": [],
        "ckpt": "ResNet50.pt",
        "input_size": 224,
    },
    "DenseNet121": {
        "factory": models.densenet121,
        "n_remove": 1,
        "dim": 1024,
        "extra": [nn.ReLU()],
        "ckpt": "DenseNet121.pt",
        "input_size": 224,
    },
    "InceptionV3": {
        "factory": lambda **kw: models.inception_v3(aux_logits=False, **kw),
        "n_remove": 3,
        "dim": 2048,
        "extra": [],
        "ckpt": "InceptionV3.pt",
        "input_size": 299,
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _make_transform(input_size):
    """Build preprocessing transform for a given input resolution."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────
def load_feature_extractor(model_name="ResNet50"):
    """Load a RadImageNet pretrained model and return a feature extractor."""
    cfg = MODEL_CONFIGS[model_name]
    model = cfg["factory"](weights=None)
    ckpt_path = WEIGHTS_DIR / cfg["ckpt"]

    state_dict_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    backbone = nn.Sequential(*list(model.children())[:-cfg["n_remove"]])
    backbone.load_state_dict(
        {k.replace("backbone.", ""): v for k, v in state_dict_raw.items()}
    )

    layers = [backbone] + cfg["extra"] + [
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ]
    extractor = nn.Sequential(*layers)
    extractor.eval()
    return extractor


def extract_features(extractor, folder, label, input_size=224):
    """Extract CNN features from all images in a folder."""
    transform = _make_transform(input_size)
    records = []
    for f in sorted(os.listdir(folder)):
        if Path(f).suffix.lower() in IMAGE_EXTS:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                feats = extractor(tensor).squeeze(0).numpy()
            records.append((feats, label, f))
    return records


def _file_manifest():
    """Build a sorted list of (filename, file_size) for cache validation."""
    manifest = []
    for subdir in ["term", "pre-term"]:
        folder = DATA_DIR / subdir
        if folder.exists():
            for f in sorted(os.listdir(folder)):
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    manifest.append((f"{subdir}/{f}", os.path.getsize(folder / f)))
    return manifest


def load_or_extract(model_name="ResNet50"):
    """Load features from pickle cache or extract them.

    Cache is invalidated when the data files change (based on filenames and sizes).
    Returns X, y, filenames.
    """
    cache_path = WEIGHTS_DIR / f"features_{model_name}.pkl"
    current_manifest = _file_manifest()

    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("file_manifest") == current_manifest:
            print(f"  Loaded cached features from {cache_path.name}")
            return cached["X"], cached["y"], cached["filenames"]
        else:
            print(f"  Cache invalidated (data files changed), re-extracting...")

    print(f"Loading RadImageNet {model_name}...")
    extractor = load_feature_extractor(model_name)
    n_params = sum(p.numel() for p in extractor.parameters())
    print(f"  {n_params:,} parameters")

    input_size = MODEL_CONFIGS[model_name]["input_size"]
    print(f"Extracting CNN features (input {input_size}×{input_size})...")
    data_a = extract_features(extractor, DATA_DIR / "term", 0, input_size)
    data_b = extract_features(extractor, DATA_DIR / "pre-term", 1, input_size)
    all_data = data_a + data_b
    X = np.array([d[0] for d in all_data])
    y = np.array([d[1] for d in all_data])
    filenames = [d[2] for d in all_data]
    print(f"  Class a: {len(data_a)}, Class b: {len(data_b)}, Total: {len(all_data)}")
    print(f"  Feature vector: {X.shape[1]}-d")

    with open(cache_path, "wb") as fh:
        pickle.dump({
            "X": X, "y": y, "filenames": filenames,
            "file_manifest": current_manifest,
        }, fh)
    print(f"  Cached features to {cache_path.name}")

    return X, y, filenames


# ──────────────────────────────────────────────────────────────
# Classification pipeline + 5-Fold CV
# ──────────────────────────────────────────────────────────────
def build_pipeline(n_pca=20, classifier=None, fs_mode="pca", anova_k=None):
    """Build a sklearn Pipeline with scaler, feature selection, and classifier.

    fs_mode: "pca" | "anova" | "anova_pca"
    """
    if classifier is None:
        classifier = SVC(
            kernel="rbf", C=0.1, gamma="scale",
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    steps = [("scaler", StandardScaler())]
    if fs_mode in ("anova", "anova_pca"):
        steps.append(("anova", SelectKBest(f_classif, k=anova_k)))
    if fs_mode in ("pca", "anova_pca"):
        steps.append(("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def run_cv(X, y, n_pca=20, classifier=None, fs_mode="pca", anova_k=None):
    """Run 5-fold stratified CV with variance filter inside the loop.

    Returns y_pred, decision_vals, proba (proba is None if classifier lacks predict_proba).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros(len(y), dtype=int)
    decision_vals = np.full(len(y), np.nan)
    proba = np.full((len(y), 2), np.nan) if True else None  # placeholder
    has_proba = None
    kept_masks = []

    for train_idx, test_idx in skf.split(X, y):
        variances = np.var(X[train_idx], axis=0)
        keep = variances > 1e-10
        kept_masks.append(keep)
        X_train = X[train_idx][:, keep]
        X_test = X[test_idx][:, keep]

        pipe = build_pipeline(n_pca, classifier, fs_mode, anova_k)
        pipe.fit(X_train, y[train_idx])
        y_pred[test_idx] = pipe.predict(X_test)

        X_test_transformed = pipe[:-1].transform(X_test)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "decision_function"):
            decision_vals[test_idx] = clf.decision_function(X_test_transformed)
        if hasattr(clf, "predict_proba"):
            has_proba = True
            proba[test_idx] = clf.predict_proba(X_test_transformed)

    if not has_proba:
        proba = None

    # Report variance filter consistency
    kept_masks = np.array(kept_masks)
    if np.all(kept_masks == kept_masks[0]):
        n_removed = np.sum(~kept_masks[0])
        print(f"  Variance filter: same {np.sum(kept_masks[0])} features kept in all "
              f"{len(kept_masks)} folds ({n_removed} removed)")
    else:
        diffs = np.any(kept_masks != kept_masks[0], axis=1)
        n_diff = np.sum(diffs)
        print(f"  WARNING: variance filter differed in {n_diff}/{len(kept_masks)} folds")
        for i in np.where(diffs)[0]:
            changed = np.where(kept_masks[i] != kept_masks[0])[0]
            print(f"    Fold {i}: columns {changed.tolist()} differ from fold 0")

    return y_pred, decision_vals, proba


def compute_metrics(y, y_pred):
    """Compute classification metrics."""
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = f1_score(y, y_pred)

    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "f1": f1, "cm": cm,
    }


# ──────────────────────────────────────────────────────────────
# Single-fold helpers (used by explainability scripts)
# ──────────────────────────────────────────────────────────────
def fit_fold_pipeline(X_train, y_train, n_pca=20, classifier=None,
                      fs_mode="pca", anova_k=None):
    """Fit variance filter + sklearn pipeline on one fold, return artifacts.

    Returns dict with 'keep_mask' and 'pipe' (fitted sklearn Pipeline).
    """
    variances = np.var(X_train, axis=0)
    keep_mask = variances > 1e-10
    X_filt = X_train[:, keep_mask]

    pipe = build_pipeline(n_pca, classifier, fs_mode, anova_k)
    pipe.fit(X_filt, y_train)
    return {"keep_mask": keep_mask, "pipe": pipe}


def predict_with_fold_pipeline(X_test, artifacts):
    """Predict using fold artifacts. Returns (predicted, decision_values)."""
    X_filt = X_test[:, artifacts["keep_mask"]]
    pipe = artifacts["pipe"]
    predicted = pipe.predict(X_filt)

    X_transformed = pipe[:-1].transform(X_filt)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        decision_values = clf.decision_function(X_transformed)
    elif hasattr(clf, "predict_proba"):
        decision_values = clf.predict_proba(X_transformed)[:, 1] - 0.5
    else:
        decision_values = np.zeros(len(predicted))

    return predicted, decision_values


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────
def main(model_name="ResNet50", classifier=None, n_pca=20, fs_mode="pca", anova_k=None):
    """Run the full pipeline and return a results dict.

    No file I/O except the feature pickle cache.
    """
    X, y, filenames = load_or_extract(model_name)

    clf_name = type(classifier).__name__ if classifier is not None else "SVC"
    fs_label = {"pca": f"PCA({n_pca})", "anova": f"ANOVA({anova_k})",
                "anova_pca": f"ANOVA({anova_k})->PCA({n_pca})"}[fs_mode]
    print(f"\nRunning 5-fold CV - {model_name} -> {fs_label} + {clf_name}...")
    y_pred, decision_vals, proba = run_cv(X, y, n_pca, classifier, fs_mode, anova_k)
    metrics = compute_metrics(y, y_pred)

    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  F1:          {metrics['f1']:.3f}")

    misclassified = np.where(y_pred != y)[0]
    if len(misclassified) > 0:
        print(f"  Misclassified ({len(misclassified)}):")
        for i in misclassified:
            true_cls = "a" if y[i] == 0 else "b"
            pred_cls = "a" if y_pred[i] == 0 else "b"
            print(f"    {filenames[i]:30s}  true={true_cls}  pred={pred_cls}  "
                  f"decision={decision_vals[i]:.3f}")

    return {
        "X": X, "y": y, "filenames": filenames,
        "y_pred": y_pred, "decision_vals": decision_vals,
        "proba": proba, "metrics": metrics, "model_name": model_name,
        "n_pca": n_pca, "classifier": classifier,
    }
