"""
Shared XAI utilities for the image classification pipelines.

Provides:
  - DifferentiableLinearHead: differentiable wrapper around the fold-specific
    variance-filter -> StandardScaler -> ANOVA -> PCA -> linear classifier path,
    enabling gradient-based XAI methods (Grad-CAM, DeepLIFT).
  - DifferentiableModel: backbone + avgpool + linear head.
  - BinaryTarget: GradCAM target for single-logit binary models.
  - Helper functions for loading images, saving heatmaps, and generating
    composite grid figures.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def disable_inplace_relu(module):
    """Set inplace=False on all ReLU modules (required by DeepLIFT/Captum)."""
    for child in module.modules():
        if isinstance(child, nn.ReLU):
            child.inplace = False


def _bottleneck_forward_unique_relu(self, x):
    """Patched Bottleneck forward with three separate ReLU instances."""
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    out = self.relu3(out)
    return out


def make_relu_unique(module):
    """Patch ResNet Bottleneck blocks so each ReLU activation uses a unique
    ``nn.ReLU`` instance.  Required by Captum's DeepLIFT.
    """
    import types
    from torchvision.models.resnet import Bottleneck, BasicBlock

    for m in module.modules():
        if isinstance(m, Bottleneck):
            m.relu1 = nn.ReLU(inplace=False)
            m.relu2 = nn.ReLU(inplace=False)
            m.relu3 = nn.ReLU(inplace=False)
            m.relu = nn.Identity()
            m.forward = types.MethodType(_bottleneck_forward_unique_relu, m)
        elif isinstance(m, BasicBlock):
            m.relu1 = nn.ReLU(inplace=False)
            m.relu2 = nn.ReLU(inplace=False)
            m.relu = nn.Identity()
        elif isinstance(m, nn.ReLU):
            m.inplace = False


# ---------------------------------------------------------------------------
# Differentiable linear head
# ---------------------------------------------------------------------------

class DifferentiableLinearHead(nn.Module):
    """Differentiable variance-filter -> scale -> ANOVA -> PCA -> linear decision.

    Works for any linear classifier with ``coef_`` and ``intercept_`` attributes
    (LDA, SGD, LogisticRegression, LinearSVC, etc.).

    Parameters
    ----------
    artifacts : dict
        Output of ``pipeline.fit_fold_pipeline`` with keys
        ``keep_mask`` and ``pipe`` (fitted sklearn Pipeline).
    """

    def __init__(self, artifacts):
        super().__init__()
        keep_mask = artifacts["keep_mask"]
        pipe = artifacts["pipe"]

        # Variance filter
        self.register_buffer("keep_idx", torch.nonzero(
            torch.tensor(keep_mask, dtype=torch.bool)
        ).squeeze(1))

        # StandardScaler
        scaler = pipe.named_steps["scaler"]
        self.register_buffer("sc_mean", torch.tensor(
            scaler.mean_, dtype=torch.float64))
        self.register_buffer("sc_scale", torch.tensor(
            scaler.scale_, dtype=torch.float64))

        # ANOVA (optional)
        if "anova" in pipe.named_steps:
            anova = pipe.named_steps["anova"]
            anova_mask = anova.get_support()
            self.register_buffer("anova_idx", torch.nonzero(
                torch.tensor(anova_mask, dtype=torch.bool)
            ).squeeze(1))
        else:
            self.anova_idx = None

        # PCA (optional)
        if "pca" in pipe.named_steps:
            pca = pipe.named_steps["pca"]
            self.register_buffer("pca_mean", torch.tensor(
                pca.mean_, dtype=torch.float64))
            self.register_buffer("pca_components", torch.tensor(
                pca.components_, dtype=torch.float64))
        else:
            self.pca_mean = None
            self.pca_components = None

        # Linear classifier: coef_ and intercept_
        clf = pipe.named_steps["clf"]
        coef = clf.coef_
        intercept = clf.intercept_
        # LDA/SGD may have shape (1, n_features) or (n_features,)
        if coef.ndim == 2:
            coef = coef[0]
        if hasattr(intercept, '__len__'):
            intercept = intercept[0]
        self.register_buffer("weight", torch.tensor(coef, dtype=torch.float64))
        self.register_buffer("bias", torch.tensor(float(intercept), dtype=torch.float64))

    def forward(self, features):
        """
        Parameters
        ----------
        features : Tensor, shape (batch, feat_dim), dtype float32

        Returns
        -------
        decision : Tensor, shape (batch,), dtype float32
        """
        x = features.double()

        # Variance filter
        x = x[:, self.keep_idx]

        # StandardScaler
        x = (x - self.sc_mean) / self.sc_scale

        # ANOVA
        if self.anova_idx is not None:
            x = x[:, self.anova_idx]

        # PCA
        if self.pca_components is not None:
            x = (x - self.pca_mean) @ self.pca_components.t()

        # Linear decision
        decision = x @ self.weight + self.bias

        return decision.float()


class DifferentiableModel(nn.Module):
    """Full end-to-end model: backbone -> avgpool -> flatten -> linear head.

    Used by gradient-based XAI methods (Grad-CAM, DeepLIFT) that need a
    differentiable path from input pixels to the decision value.
    """

    def __init__(self, backbone, head, extra_layers=None):
        super().__init__()
        self.backbone = backbone
        self.extra = nn.Sequential(*extra_layers) if extra_layers else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        features = self.extra(features)
        pooled = self.avgpool(features).flatten(1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# GradCAM target
# ---------------------------------------------------------------------------

class BinaryTarget:
    """GradCAM target for binary (single-logit) models.

    For predicted class 1 (preterm): returns logits directly.
    For predicted class 0 (term): returns negated logits.
    """

    def __init__(self, positive_class):
        self.positive_class = positive_class

    def __call__(self, model_output):
        out = model_output.squeeze(-1)
        return out if self.positive_class else -out


# ---------------------------------------------------------------------------
# Ultrasound sector mask
# ---------------------------------------------------------------------------

def compute_us_mask(img_np, threshold=15):
    """Detect the ultrasound sector cone and return a binary mask.

    Parameters
    ----------
    img_np : ndarray, HxWx3, uint8, RGB
    threshold : int

    Returns
    -------
    mask : ndarray, HxW, uint8, values 0 or 255
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def mask_heatmap(heatmap, us_mask):
    """Zero out heatmap outside the ultrasound sector and re-normalize."""
    mask_resized = cv2.resize(us_mask, (heatmap.shape[1], heatmap.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized > 127

    masked = heatmap.copy()
    masked[~mask_bool] = 0.0

    vals = masked[mask_bool]
    if len(vals) > 0:
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-10:
            masked[mask_bool] = (vals - vmin) / (vmax - vmin)
        else:
            masked[mask_bool] = 0.0

    return masked


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(filenames, y, data_dir, transform, class_dirs, device):
    """Load all images as a stacked tensor and return PIL images too.

    Parameters
    ----------
    class_dirs : dict
        Mapping from label int to subdirectory name, e.g. {0: "term", 1: "pre-term"}.

    Returns
    -------
    X_img : Tensor, shape (N, C, H, W)
    pil_images : list of PIL.Image
    """
    tensors = []
    pil_images = []
    for i in range(len(filenames)):
        cls_dir = class_dirs[int(y[i])]
        img = Image.open(data_dir / cls_dir / filenames[i]).convert("RGB")
        pil_images.append(img)
        tensors.append(transform(img))
    X_img = torch.stack(tensors).to(device)
    return X_img, pil_images


# ---------------------------------------------------------------------------
# Heatmap I/O
# ---------------------------------------------------------------------------

def save_heatmap_outputs(base_name, img_np, grayscale_cam, out_dir,
                         heatmap_dir=None, save_original=True, us_mask=None):
    """Save original, heatmap, and overlay images."""
    h, w = img_np.shape[:2]

    heatmap_resized = cv2.resize(grayscale_cam, (w, h))

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if img_np.dtype != np.uint8:
        img_uint8 = np.uint8(255 * img_np) if img_np.max() <= 1.0 else np.uint8(img_np)
    else:
        img_uint8 = img_np

    if us_mask is not None:
        mask_bool = us_mask > 127
        overlay = img_uint8.copy()
        blended = cv2.addWeighted(img_uint8, 0.5, heatmap_colored_rgb, 0.5, 0)
        overlay[mask_bool] = blended[mask_bool]
    else:
        overlay = cv2.addWeighted(img_uint8, 0.5, heatmap_colored_rgb, 0.5, 0)

    if save_original:
        cv2.imwrite(
            str(out_dir / f"{base_name}_original.png"),
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
        )

    target_heatmap_dir = heatmap_dir if heatmap_dir is not None else out_dir
    cv2.imwrite(
        str(target_heatmap_dir / f"{base_name}_heatmap.png"),
        heatmap_colored,
    )

    cv2.imwrite(
        str(out_dir / f"{base_name}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def heatmap_statistics(grayscale_cam):
    """Return dict with mean, max, std of a grayscale heatmap."""
    return {
        "heatmap_mean": float(np.mean(grayscale_cam)),
        "heatmap_max": float(np.max(grayscale_cam)),
        "heatmap_std": float(np.std(grayscale_cam)),
    }


# ---------------------------------------------------------------------------
# Composite grid figure
# ---------------------------------------------------------------------------

def generate_grid_figure(results, out_dir, method_name, n_samples=4):
    """Generate a 2-row x N-col grid: originals on top, overlays on bottom.

    Selects the N samples with highest |decision_value|.
    """
    sorted_results = sorted(results, key=lambda r: abs(r["decision_value"]),
                            reverse=True)[:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(2.5 * n_samples, 5))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    class_names = {0: "term", 1: "preterm"}

    for col, r in enumerate(sorted_results):
        base_name = Path(r["filename"]).stem
        orig_path = out_dir / f"{base_name}_original.png"
        overlay_path = out_dir / f"{base_name}_overlay.png"

        if orig_path.exists():
            orig = cv2.cvtColor(cv2.imread(str(orig_path)), cv2.COLOR_BGR2RGB)
            axes[0, col].imshow(orig)
        axes[0, col].set_title(
            f"{class_names[r['true_label']]}\nd={r['decision_value']:.2f}",
            fontsize=8,
        )
        axes[0, col].axis("off")

        if overlay_path.exists():
            overlay = cv2.cvtColor(cv2.imread(str(overlay_path)), cv2.COLOR_BGR2RGB)
            axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel(method_name, fontsize=9)

    fig.suptitle(f"{method_name} — top-{n_samples} by |decision value|", fontsize=10)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"grid_{method_name.lower().replace(' ', '_')}.pdf",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grad-CAM target layer helpers
# ---------------------------------------------------------------------------

def get_gradcam_target_layer(model, model_name):
    """Return the appropriate target layer for Grad-CAM.

    Parameters
    ----------
    model : DifferentiableModel
    model_name : str
        "ResNet50" or "DenseNet121"
    """
    if model_name == "ResNet50":
        return model.backbone[7][-1]  # layer4[-1]
    elif model_name == "DenseNet121":
        return model.backbone[0].denseblock4  # features.denseblock4
    else:
        raise ValueError(f"Unknown model: {model_name}")
