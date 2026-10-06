# Attribution and licenses

| Asset | Source | License | How it is used |
| :-- | :-- | :-- | :-- |
| VisA dataset (Visual Anomaly) | Amazon Science, [spot-diff](https://github.com/amazon-science/spot-diff), paper: Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022 | CC BY 4.0 | Training and evaluating the demo models; sample images in the demo |
| Anomalib | [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib) | Apache-2.0 | Anomaly-detection model implementations and training loop |
| PyTorch / torchvision | [pytorch.org](https://pytorch.org) | BSD-3-Clause | Deep learning runtime; ImageNet-pretrained backbones |
| Imagenette (320 px) | fast.ai, [imagenette](https://github.com/fastai/imagenette), a 10-class subset of ImageNet | ImageNet terms of access apply to the images | Only as EfficientAD's training-time penalty set (a folder of generic photos). Never redistributed, never shown in the demo. For a client project, swap in any image folder the client is licensed to use. |
| EfficientAD pretrained teacher weights | Anomalib release assets | Apache-2.0 (Anomalib) | Downloaded automatically by Anomalib when training EfficientAD |
| Figtree typeface | [Google Fonts](https://fonts.google.com/specimen/Figtree) | SIL Open Font License 1.1 | Landing page typography |

**Not used:** MVTec AD (CC BY-NC-SA 4.0, non-commercial) is deliberately excluded from this portfolio project.
