"""
Grad-CAM — Pipeline #1: ResNet50 + ANOVA200->PCA5 + LDA (5-fold stratified CV).

Usage:
    conda run -n torch python term_preterm/explainable/run_gradcam_pipeline1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from pipeline import (
    DATA_DIR,
    RANDOM_STATE,
    load_or_extract,
    load_feature_extractor,
    fit_fold_pipeline,
    predict_with_fold_pipeline,
    _make_transform,
    MODEL_CONFIGS,
)
from explainable.xai_utils import (
    DifferentiableLinearHead,
    DifferentiableModel,
    BinaryTarget,
    save_heatmap_outputs,
    heatmap_statistics,
    compute_us_mask,
    mask_heatmap,
    get_gradcam_target_layer,
)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

CLASS_DIRS = {0: "term", 1: "pre-term"}
MODEL_NAME = "ResNet50"


def main():
    print("Grad-CAM — Pipeline #1: ResNet50 + ANOVA200->PCA5 + LDA (5-fold CV)")
    input_size = MODEL_CONFIGS[MODEL_NAME]["input_size"]
    transform = _make_transform(input_size)

    X, y, filenames = load_or_extract(MODEL_NAME)

    extractor = load_feature_extractor(MODEL_NAME)
    extractor.to(device)
    backbone = extractor[0]

    out_dir = Path(__file__).resolve().parent / "output_gradcam" / "pipeline1"
    heatmap_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(exist_ok=True)

    total = len(filenames)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Running Grad-CAM for {total} samples (5-fold CV)...")
    results = []
    sample_count = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        artifacts = fit_fold_pipeline(
            X[train_idx], y[train_idx],
            n_pca=5,
            classifier=LinearDiscriminantAnalysis(),
            fs_mode="anova_pca",
            anova_k=200,
        )

        head = DifferentiableLinearHead(artifacts)
        model = DifferentiableModel(backbone, head)
        model.to(device)
        model.eval()

        for p in model.backbone.parameters():
            p.requires_grad = True

        target_layer = get_gradcam_target_layer(model, MODEL_NAME)
        cam = GradCAM(model=model, target_layers=[target_layer])

        for idx in test_idx:
            filename = filenames[idx]
            true_label = y[idx]
            cls_dir = CLASS_DIRS[int(true_label)]

            predicted, decision_value = predict_with_fold_pipeline(
                X[idx:idx + 1], artifacts
            )

            img_pil = Image.open(DATA_DIR / cls_dir / filename).convert("RGB")
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[BinaryTarget(int(predicted[0]))]
            )[0]

            img_np = np.array(img_pil)
            us_mask = compute_us_mask(img_np)
            grayscale_cam = mask_heatmap(grayscale_cam, us_mask)
            base_name = Path(filename).stem

            save_heatmap_outputs(base_name, img_np, grayscale_cam, out_dir,
                                 heatmap_dir=heatmap_dir, us_mask=us_mask)

            stats = heatmap_statistics(grayscale_cam)
            sample_count += 1
            results.append({
                "filename": filename,
                "true_label": int(true_label),
                "predicted": int(predicted[0]),
                "correct": int(predicted[0] == true_label),
                "decision_value": float(decision_value[0]),
                "fold": fold_idx,
                **stats,
            })
            print(f"  [{sample_count}/{total}] {base_name} "
                  f"(fold={fold_idx}, true={cls_dir}, "
                  f"pred={'pre-term' if predicted[0] else 'term'}, "
                  f"d={decision_value[0]:.3f}, hm_mean={stats['heatmap_mean']:.3f})")

        del cam, model, head

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "5fold_gradcam_summary.csv",
                      index=False, float_format="%.6f")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
