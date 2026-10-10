"""
Grad-CAM for the DenseNet-121 Member
====================================
Explains the ref_densenet member of the pre-specified image ensemble
(features_eval.py): RadImageNet DenseNet-121 final-layer features ->
variance filter -> StandardScaler -> PCA(10) -> LogisticRegression(L2,
C=0.1, balanced). The head is linear after the backbone, so the whole model
is differentiable (explainable/xai_utils.DifferentiableLinearHead).

Each image is explained by the model of the seed-42 fold that did not train
on it (out-of-fold), for its predicted class.

Caliper check: share of the heatmap mass inside the caliper mask
(caliper_mask below: the dotted cervical-length measurement line and its
endpoint crosses), against the mask's share of the ultrasound area
(enrichment = mass share / area share; 1 means no preference).

Outputs (results/gradcam/):
  overlays/<name>_overlay.png, heatmap_stats.csv,
  fig_gradcam.pdf/.png (2 term + 2 preterm correctly classified + t105)

Usage:
    python gradcam.py
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import RANDOM_STATE, RESULTS_DIR, cv_splits, load_aligned
from explainable.xai_utils import (BinaryTarget, DifferentiableLinearHead, DifferentiableModel,
                                   get_gradcam_target_layer, save_heatmap_outputs)
from features_eval import logreg
from pipeline import (DATA_DIR, MODEL_CONFIGS, _make_transform, fit_fold_pipeline,
                      load_feature_extractor, predict_with_fold_pipeline)

MODEL_NAME = "DenseNet121"
CLASS_DIRS = {0: "term", 1: "pre-term"}
CONTINUITY_IMAGE = "t105_cropped.JPG"   # always shown in the figure
OUT_DIR = RESULTS_DIR / "gradcam"

TOPHAT_KERNEL = 5      # structuring element (px) for white/black top-hat
DOT_THRESH = 20        # top-hat response of a caliper-dot candidate (gray levels)
DOT_MAX_AREA = 30      # larger components are tissue, not caliper dots
MAX_TILT_DEG = 8       # crops are aligned with the canal: caliper line near horizontal
MIN_DOTS = 12          # minimum collinear dots to accept a caliper line
BAND_HALF = 3.5        # half-width (px) of the masked band along the line
END_BOX_HALF = 14      # half-size (px) of the boxes masking the endpoint crosses

torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# Caliper detection
# ──────────────────────────────────────────────────────────────
def _dot_centroids(gray):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    t = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
    n, lab, st, cen = cv2.connectedComponentsWithStats((t > DOT_THRESH).astype(np.uint8), 8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] <= DOT_MAX_AREA]
    return cen[keep]


def _best_line(cen, H, W):
    """(angle, offset, n_dots, x_min, x_max) of the densest dot row, or None."""
    best = None
    for a in np.radians(np.arange(-MAX_TILT_DEG, MAX_TILT_DEG + 0.01, 0.25)):
        r = -np.sin(a) * cen[:, 0] + np.cos(a) * cen[:, 1]
        hist, edges = np.histogram(r, bins=np.arange(-W, H + W, 2.0))
        win = hist + np.r_[hist[1:], 0]
        i = int(win.argmax())
        if best is None or win[i] > best[0]:
            best = (win[i], a, edges[i] + 2.0)
    _, a, r0 = best
    r = -np.sin(a) * cen[:, 0] + np.cos(a) * cen[:, 1]
    on = np.abs(r - r0) <= 3
    if on.sum() < MIN_DOTS:
        return None
    xs = cen[on, 0]
    if xs.max() - xs.min() < 0.2 * W:  # a tissue cluster, not a measurement line
        return None
    return a, r0, int(on.sum()), float(xs.min()), float(xs.max())


def caliper_mask(gray):
    """Mask of the caliper band and its endpoint boxes. Returns mask, line."""
    H, W = gray.shape
    mask = np.zeros_like(gray)
    cen = _dot_centroids(gray)
    line = _best_line(cen, H, W) if len(cen) else None
    if line is None:
        return mask, None
    a, r0, _, x0, x1 = line
    yy, xx = np.mgrid[0:H, 0:W]
    rp = -np.sin(a) * xx + np.cos(a) * yy
    mask[(np.abs(rp - r0) <= BAND_HALF) & (xx >= x0 - END_BOX_HALF)
         & (xx <= x1 + END_BOX_HALF)] = 1
    for xe in (x0, x1):
        ye = (r0 + np.sin(a) * xe) / np.cos(a)
        mask[max(0, int(ye - END_BOX_HALF)):int(ye + END_BOX_HALF) + 1,
             max(0, int(xe - END_BOX_HALF)):int(xe + END_BOX_HALF) + 1] = 1
    return mask, line


# ──────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────
def grad_cam(model, layer, x, target):
    """Grad-CAM (Selvaraju et al.) as in pytorch_grad_cam.GradCAM: channel weights are
    the spatially averaged gradients; ReLU; upsampled to the input; min-max scaled."""
    store = {}

    def fwd(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    h_fwd = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        target(model(x)).sum().backward()
    finally:
        h_fwd.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = nnf.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def caliper_attention(cam_full, us_mask, gray):
    """Heatmap mass share and area share of the caliper mask inside the US region."""
    cal, line = caliper_mask(gray)
    region = us_mask > 127
    cal = (cal > 0) & region
    if line is None or cal.sum() == 0:
        return dict(caliper_found=False, caliper_area_share=0.0, caliper_mass_share=np.nan,
                    caliper_enrichment=np.nan)
    area = cal.sum() / region.sum()
    mass = cam_full[cal].sum() / max(cam_full[region].sum(), 1e-12)
    return dict(caliper_found=True, caliper_area_share=area, caliper_mass_share=mass,
                caliper_enrichment=mass / area)


def make_figure(df, images):
    pick = []
    for label in (0, 1):
        ok = df[(df.true_label == label) & (df.correct == 1)
                & (df.filename != CONTINUITY_IMAGE)]
        pick += list(ok.reindex(ok.decision_value.abs().sort_values(ascending=False).index)
                     .filename[:2])
    pick.append(CONTINUITY_IMAGE)
    fig, axes = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 4.6))
    for j, fn in enumerate(pick):
        r = df.set_index("filename").loc[fn]
        orig, overlay = images[fn]
        axes[0, j].imshow(orig)
        axes[1, j].imshow(overlay)
        truth = "preterm" if r.true_label else "term"
        pred = "preterm" if r.predicted else "term"
        axes[0, j].set_title(f"{fn.split('_')[0]}\ntrue {truth}, pred {pred}", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    axes[0, 0].text(-0.08, 0.5, "Original", rotation=90, va="center", ha="right",
                    transform=axes[0, 0].transAxes, fontsize=9)
    axes[1, 0].text(-0.08, 0.5, "Grad-CAM", rotation=90, va="center", ha="right",
                    transform=axes[1, 0].transAxes, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_gradcam.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pick


if __name__ == "__main__":
    (OUT_DIR / "overlays").mkdir(parents=True, exist_ok=True)
    X, _, y, filenames = load_aligned(MODEL_NAME)
    splits = cv_splits(y, RANDOM_STATE)

    extractor = load_feature_extractor(MODEL_NAME).to(device)
    backbone = extractor[0]
    extra_layers = list(MODEL_CONFIGS[MODEL_NAME]["extra"])
    transform = _make_transform(MODEL_CONFIGS[MODEL_NAME]["input_size"])

    rows, images = [], {}
    for fold, (tr, te) in enumerate(splits):
        artifacts = fit_fold_pipeline(X[tr], y[tr], n_pca=10, classifier=logreg(), fs_mode="pca")
        model = DifferentiableModel(backbone, DifferentiableLinearHead(artifacts),
                                    extra_layers).to(device).eval()
        for p in model.backbone.parameters():
            p.requires_grad = True
        layer = get_gradcam_target_layer(model, MODEL_NAME)
        for i in te:
            fn = filenames[i]
            pred, dec = predict_with_fold_pipeline(X[i:i + 1], artifacts)
            img_pil = Image.open(DATA_DIR / CLASS_DIRS[int(y[i])] / fn).convert("RGB")
            img_np = np.array(img_pil)
            # the differentiable head must reproduce the sklearn decision value
            with torch.no_grad():
                d_torch = model(transform(img_pil).unsqueeze(0).to(device)).item()
            g = grad_cam(model, layer, transform(img_pil).unsqueeze(0).to(device),
                         BinaryTarget(int(pred[0])))
            # the crops are ultrasound throughout; xai_utils.compute_us_mask (made for full
            # sector images) would drop dark anechoic tissue as background
            us_mask = np.full(img_np.shape[:2], 255, np.uint8)
            base = fn.rsplit(".", 1)[0]
            save_heatmap_outputs(base, img_np, g, OUT_DIR / "overlays", save_original=False,
                                 us_mask=us_mask)
            overlay = cv2.cvtColor(cv2.imread(str(OUT_DIR / "overlays" / f"{base}_overlay.png")),
                                   cv2.COLOR_BGR2RGB)
            images[fn] = (img_np, overlay)
            h, w = img_np.shape[:2]
            cam_full = cv2.resize(g, (w, h))
            rows.append(dict(filename=fn, fold=fold, true_label=int(y[i]), predicted=int(pred[0]),
                             correct=int(pred[0] == y[i]), decision_value=float(dec[0]),
                             decision_torch=d_torch,
                             **caliper_attention(cam_full, us_mask,
                                                 cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY))))
        print(f"  fold {fold} done", flush=True)
        del model

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "heatmap_stats.csv", index=False, float_format="%.5f")
    pick = make_figure(df, images)

    dec_gap = np.max(np.abs(df.decision_value - df.decision_torch))
    auc = roc_auc_score(df.true_label, df.decision_value)
    cal = df[df.caliper_found]
    print(f"\nOOF AUC of the explained model (seed 42): {auc:.3f}; "
          f"accuracy {df.correct.mean():.3f}")
    print(f"Max |sklearn - torch| decision difference: {dec_gap:.2e}")
    print(f"Caliper line found in {len(cal)}/{len(df)} images")
    print(cal.groupby("true_label")[["caliper_area_share", "caliper_mass_share",
                                     "caliper_enrichment"]].agg(["mean", "median"]).to_string(
        float_format=lambda v: f"{v:.3f}"))
    print(f"All: enrichment mean {cal.caliper_enrichment.mean():.3f}, "
          f"median {cal.caliper_enrichment.median():.3f}; "
          f"share of images with enrichment > 1: {(cal.caliper_enrichment > 1).mean():.2f}")
    print(f"Figure images: {pick}")
