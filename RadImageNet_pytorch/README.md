# RadImageNet Pretrained Weights

Download the pretrained checkpoints from the RadImageNet project and place them in this directory.

## Download

The RadImageNet pretrained weights are available from:
- GitHub: https://github.com/BMEII-AI/RadImageNet
- Paper: Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"

Download the checkpoint(s) you need and place them here:

```
RadImageNet_pytorch/ResNet50.pt      # ~94 MB, 2048-d features
RadImageNet_pytorch/DenseNet121.pt   # ~29 MB, 1024-d features
RadImageNet_pytorch/InceptionV3.pt   # ~88 MB, 2048-d features
```

Only the model(s) you plan to use need to be present. The default pipeline uses ResNet50.

## Notes

- All checkpoints use a `backbone.*` key prefix that is automatically stripped by the loading code in `pipeline.py`.
- Extracted features are cached to `RadImageNet_pytorch/features_{model}.pkl` so subsequent runs skip feature extraction. The cache auto-invalidates when data files change (based on filenames and sizes).
- Prepopulated feature caches are included in this directory for all three models, so feature extraction can be skipped on first run.
