# Defecto — visual quality inspection demo

Visual quality inspection that learns from photos of good parts (unsupervised anomaly detection), with held-out evaluation, pass/review/fail thresholds and a browser demo.

## Status (6 October 2026)

| Piece | State |
| :-- | :-- |
| Models | Trained and evaluated for pcb1, capsules and cashew with `ml/patchcore_lite.py` (CPU, about 2 minutes per product). |
| Landing page `web/index.html` | Redesigned as a scrollable page. Every number and the hero heatmap are read from the exported evaluation at load time, so they cannot drift from the results. The placeholder video, avatars and example scores are gone. |
| Workbench `web/app.html` | Demo (conveyor replay with real scores, verdicts and heatmaps), **Live inspect** (the model runs in the browser: WebGPU, falling back to WebAssembly), Evaluation (ROC, threshold explorer, misses and false alarms), Product, Pricing, FAQ, Book a Pilot. |

### Results on the held-out test split

Training used good parts only; thresholds and score normalisation were fitted on the validation split; nothing below was fitted on test.

| Product | Image AUROC | Pixel AUROC | AUPRO@0.3 | Caught at FAIL | Good parts rejected | Defects auto-passed (below REVIEW) |
| :-- | --: | --: | --: | --: | --: | --: |
| PCB1 | 0.931 | 0.997 | 0.918 | 76.7% (46/60) | 10.0% (6/60) | 3.3% (2/60) |
| Capsules | 0.708 | 0.959 | 0.697 | 46.7% (28/60) | 2.8% (1/36) | 1.7% (1/60) |
| Cashew | 0.934 | 0.989 | 0.852 | 66.7% (40/60) | 3.3% (1/30) | 8.3% (5/60) |

Capsules is the weak product: a tray of about 20 capsules leaves each defect only a few pixels wide at the 256 px model input. A production build would tile the image or raise the resolution.

Browser parity: on all 306 test images the in-browser model (WebGPU and WebAssembly) reproduces the Python score and verdict exactly, because the browser is fed the exported 256 px model input (`demo/<category>/input/*.webp`, lossless). Uploaded photos go through a JavaScript box resize that matches Pillow's `Image.BOX`; on full-resolution originals the scores differ from Python by 0.002 to 0.016.

## How the model works (`ml/patchcore_lite.py`)

- **Features:** ImageNet ResNet18, layer2 (128 channels, 32x32) and layer3 (256 channels, upsampled to 32x32), 3x3 neighbourhood averaging, concatenated into 384-dim patch embeddings.
- **Memory bank:** embeddings of 300 good training parts, reduced to 3,072 by greedy k-center coreset selection on a random 128-d projection.
- **Scoring:** each patch's distance to its nearest memory embedding; the 32x32 map is upsampled to 256x256 and Gaussian-smoothed (sigma 4); the image score is the map's maximum.
- **Thresholds (validation only):** FAIL rejects at most 5% of good validation parts; REVIEW is the highest score that still sends 98% of validation defects to REVIEW or FAIL. Scores are normalised so FAIL = 0.5.
- **Export:** one ONNX graph (`web/public/models/patchcore-resnet18.onnx`, 11 MB) takes the image and a product's memory bank (`<category>.bank.bin`, 4.7 MB). Bilinear upsampling and the replicate-padded blur are expressed as fixed matrices (`MatMul`) and the graph is constant-folded, because ONNX Runtime's WebGPU layout pass rejects `Resize`/`Pad` with run-time sizes. Outputs match the PyTorch model to within 3e-7.

The original anomalib scripts (`ml/train.py`, `ml/evaluate.py`, wide_resnet50 PatchCore and EfficientAD) remain for a machine with a GPU that has enough memory; they write the same `eval.json` format.

## Setup

Requires Python 3.12 and Node 22. All caches live under `.cache/` in this folder (see `ml/_env.py`) because C: is nearly full.

```sh
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install "anomalib[core]"
```

Data (VisA, CC BY 4.0, 1.9 GB tar) and EfficientAD's penalty images (Imagenette 320 px):

```sh
curl -L -o data/VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
.venv/Scripts/python ml/prepare_visa.py --categories pcb1 capsules cashew
# Imagenette: extract https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz into data/imagenette/
```

## Train, evaluate and export

```sh
# CPU only. Writes models/<category>/patchcore/{train_run,eval}.json, heatmaps, and the browser model + banks
.venv/Scripts/python ml/patchcore_lite.py --categories pcb1 capsules cashew
# Re-evaluate from saved memory banks without retraining
.venv/Scripts/python ml/patchcore_lite.py --categories pcb1 --reuse-bank --no-export
# Refresh web data: display images, exact 256 px model inputs, masks, heatmaps, manifests
.venv/Scripts/python ml/export_web.py --categories pcb1 capsules cashew

cd web && npm install
npm run dev      # copies ONNX Runtime binaries to public/ort for the dev server
npm run build    # production build in web/dist (about 138 MB, assets load on demand)
```

Note: `ort-wasm-simd-threaded.asyncify.wasm` (the WebGPU runtime) is 26.8 MB, above Cloudflare Pages' 25 MiB per-file limit; host on a service without that limit (Netlify, Vercel, S3/CloudFront) or serve it from a CDN.

## Evaluation protocol

- VisA's official split: `train` = good parts only; the official test set is split 40/60 (stratified, seed 42) into `val` and `test`.
- The FAIL threshold (score 0.5) is fitted on `val` so at most 5 % of good val parts are rejected; the REVIEW threshold is fitted on `val` (≥ 98 % of val defects reach REVIEW or FAIL).
- Every reported number comes from `test`, which no training or tuning step touches.
- PatchCore-lite uses a seeded 300-image subset of the good training parts to fit the laptop's memory.

## Layout

```
ml/        prepare_visa.py, patchcore_lite.py, export_web.py, train.py, evaluate.py, _env.py
web/       index.html (landing), app.html + src/ (workbench), public/models (browser model + banks)
data/      datasets (not committed)
models/    train_run.json, eval.json, heatmaps per category/model
results/   Anomalib checkpoints (not committed)
logs/      training logs
```

Licenses: see `ATTRIBUTION.md`.
