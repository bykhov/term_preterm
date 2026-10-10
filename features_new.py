"""
Image-Feature Families
======================
Extracts three image feature families from the cropped images
(Data/{term,pre-term}, same file order as pipeline.load_or_extract); the
fourth family, the RadImageNet DenseNet-121 final-layer features, comes from
pipeline.load_or_extract:

  hc   : handcrafted texture (scikit-image), grayscale crop, 54 features
         first-order (8), GLCM 32 levels x distances {1,2,4}, angle-averaged (18),
         uniform LBP histograms (P=8,R=1) and (P=16,R=2) (28)
  mid  : mid-level RadImageNet ResNet-50 texture, global-average-pooled
         layer2 (512) + layer3 (1024) = 1536-d
  clip : BiomedCLIP ViT-B/16 image embedding (512-d, L2-normalized)

mid and clip use test-time augmentation fixed in advance: features averaged
over 4 views (original, horizontal flip, rotation +5 and -5 degrees).

Outputs: results/features_new/{family}.pkl
         (dict with X, y, filenames, tta_cosine)

Usage:
    python features_new.py [hc] [mid] [clip]   (default: all)
"""

import os
import pickle
import sys
from pathlib import Path

from PIL import Image, ImageOps  # noqa: F401  (must load before torch: DLL clash)
import numpy as np
import torch
from scipy.stats import kurtosis, skew

from common import RESULTS_DIR, load_aligned
from pipeline import DATA_DIR, IMAGE_EXTS, _make_transform, load_feature_extractor

OUT_DIR = RESULTS_DIR / "features_new"
CLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def image_list():
    """(path, label, filename) in the same order as pipeline.load_or_extract."""
    out = []
    for sub, label in [("term", 0), ("pre-term", 1)]:
        for f in sorted(os.listdir(DATA_DIR / sub)):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                out.append((DATA_DIR / sub / f, label, f))
    return out


def views(img):
    """Test-time augmentation views (fixed in advance)."""
    return [img, ImageOps.mirror(img),
            img.rotate(5, resample=Image.BILINEAR), img.rotate(-5, resample=Image.BILINEAR)]


# ──────────────────────────────────────────────────────────────
# Families
# ──────────────────────────────────────────────────────────────
def hc_features(img):
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    g = np.asarray(img.convert("L"))
    v = g.ravel().astype(float)
    hist = np.bincount(g.ravel(), minlength=256) / v.size
    nz = hist[hist > 0]
    first = [v.mean(), v.std(), skew(v), kurtosis(v), -np.sum(nz * np.log2(nz)),
             *np.percentile(v, [10, 50, 90])]
    glcm = graycomatrix(g // 8, distances=[1, 2, 4],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    tex = [x for p in GLCM_PROPS for x in graycoprops(glcm, p).mean(axis=1)]
    lbp = []
    for P, R in [(8, 1), (16, 2)]:
        codes = local_binary_pattern(g, P, R, method="uniform")
        h = np.bincount(codes.astype(int).ravel(), minlength=P + 2)
        lbp.extend(h / h.sum())
    return np.array(first + tex + lbp, dtype=float)


def make_mid():
    backbone = load_feature_extractor("ResNet50")[0]  # conv1 ... layer4
    to_l2, l3 = backbone[:6], backbone[6]
    tf = _make_transform(224)

    def f(img):
        with torch.no_grad():
            a = to_l2(tf(img.convert("RGB")).unsqueeze(0))
            b = l3(a)
        return torch.cat([a.mean(dim=(2, 3)), b.mean(dim=(2, 3))], dim=1).squeeze(0).numpy()
    return f


def make_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ID)
    model.eval()

    def f(img):
        # squash to 224 x 224 first (as pipeline._make_transform does): the default
        # short-side resize + center crop would cut the lateral tissue of wide crops
        img = img.convert("RGB").resize((224, 224), Image.BICUBIC)
        with torch.no_grad():
            e = model.encode_image(preprocess(img).unsqueeze(0))
        e = e / e.norm(dim=-1, keepdim=True)
        return e.squeeze(0).numpy()
    return f


def extract(family):
    items = image_list()
    if family == "hc":
        fn, tta = hc_features, False
    else:
        fn, tta = (make_mid() if family == "mid" else make_clip()), True
    X, cos = [], []
    for path, _, _ in items:
        img = Image.open(path).convert("RGB")
        if tta:
            fv = [fn(v) for v in views(img)]
            a, b = fv[0], fv[1]
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            x = np.mean(fv, axis=0)
            if family == "clip":
                x = x / np.linalg.norm(x)
            X.append(x)
        else:
            X.append(fn(img))
    X = np.array(X)
    y = np.array([lab for _, lab, _ in items])
    filenames = [f for _, _, f in items]
    _, _, y_ref, fn_ref = load_aligned("ResNet50")
    assert filenames == fn_ref and np.array_equal(y, y_ref), "order mismatch with load_aligned"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{family}.pkl", "wb") as fh:
        pickle.dump({"X": X, "y": y, "filenames": filenames,
                     "tta_cosine": float(np.mean(cos)) if cos else np.nan}, fh)
    msg = f", TTA cosine(original, flip) = {np.mean(cos):.3f}" if cos else ""
    print(f"  {family}: {X.shape}{msg}")


def load_family(family):
    with open(OUT_DIR / f"{family}.pkl", "rb") as fh:
        d = pickle.load(fh)
    return d["X"], d["y"], d["filenames"]


if __name__ == "__main__":
    for fam in (sys.argv[1:] or ["hc", "mid", "clip"]):
        print(f"Extracting {fam}...", flush=True)
        extract(fam)
