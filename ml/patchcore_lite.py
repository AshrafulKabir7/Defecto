"""PatchCore-lite: train, evaluate and export a browser-ready anomaly detector on the CPU.

Why this exists: the anomalib PatchCore/EfficientAD runs in train.py need a GPU that the
dev laptop cannot back with enough memory. This is the same PatchCore idea, sized for a
CPU and for in-browser inference:

    features   ImageNet ResNet18, layer2 (128 ch, 32x32) + layer3 (256 ch, 16x16 -> 32x32),
               3x3 neighbourhood averaging, concatenated to 384-dim patch embeddings
    memory     embeddings of good training parts, reduced by greedy k-center coreset
    score      per patch: distance to the nearest memory embedding; map upsampled to
               256x256 and Gaussian-smoothed (sigma 4); image score = map maximum

One ONNX graph (`web/public/models/patchcore-resnet18.onnx`) takes the image and a
category's memory bank, so the browser runs exactly the computation evaluated here.

Protocol: train on train/good only. Thresholds and score normalisation are fitted on val;
every reported number comes from test, which nothing is fitted on.
    FAIL    automatic reject; set so at most 5% of good validation parts are rejected
    REVIEW  human check; set so >= 98% of validation defects reach REVIEW or FAIL
Scores are normalised so the FAIL threshold is 0.5.

Writes:
    models/<category>/patchcore/train_run.json, eval.json, heatmaps/*.png
    web/public/models/patchcore-resnet18.onnx          (shared feature + scoring graph)
    web/public/models/<category>.bank.bin / .json       (memory bank float32 + normalisation)

Usage:
    python ml/patchcore_lite.py --categories pcb1 capsules cashew
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import _env  # noqa: F401  (caches on the project drive)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: no CUDA memory commit

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
WEB_MODELS = ROOT / "web" / "public" / "models"
SIZE = 256
GRID = 32
DIM = 384
FAIL_T = 0.5
REVIEW_RECALL_TARGET = 0.98
FAIL_MAX_GOOD_REJECT = 0.05  # FAIL may reject at most 5% of good validation parts
SIGMA = 4

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def linear_upsample_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """[n_out, n_in] matrix equal to 1-D linear interpolation (align_corners=False).

    Bilinear resizing is separable, so U @ X @ U.T reproduces F.interpolate(..., "bilinear")
    exactly, using only MatMul (ONNX Resize trips ONNX Runtime's WebGPU layout pass).
    """
    eye = torch.eye(n_in).unsqueeze(1)
    return F.interpolate(eye, size=n_out, mode="linear", align_corners=False).squeeze(1).t().contiguous()


def gaussian_matrix(n: int, sigma: float) -> torch.Tensor:
    """[n, n] matrix applying a 1-D Gaussian blur with replicate padding."""
    k = gaussian_kernel(sigma)
    r = k.numel() // 2
    g = torch.zeros(n, n)
    for i in range(n):
        for t in range(k.numel()):
            g[i, min(max(i + t - r, 0), n - 1)] += k[t]
    return g


class FeatureNet(nn.Module):
    """Image in [0, 1], shape [B, 3, 256, 256] -> patch embeddings [B, 1024, 384]."""

    def __init__(self) -> None:
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("up16", linear_upsample_matrix(GRID // 2, GRID))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = self.up16 @ f3 @ self.up16.t()  # bilinear 16x16 -> 32x32
        f = torch.cat([f2, f3], 1)
        return f.permute(0, 2, 3, 1).reshape(x.shape[0], GRID * GRID, DIM)


class Scorer(nn.Module):
    """(image [1,3,256,256], bank [M,384]) -> (raw anomaly map [1,1,256,256], raw score [1])."""

    def __init__(self, features: FeatureNet) -> None:
        super().__init__()
        self.features = features
        # Bilinear 32 -> 256 upsampling followed by a replicate-padded Gaussian blur,
        # folded into one [256, 32] matrix per axis: map = A @ dist @ A.T.
        self.register_buffer("smooth", gaussian_matrix(SIZE, SIGMA) @ linear_upsample_matrix(GRID, SIZE))

    def forward(self, image: torch.Tensor, bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(image)[0]
        d2 = (f * f).sum(1, keepdim=True) - 2 * f @ bank.t() + (bank * bank).sum(1).unsqueeze(0)
        dist = torch.sqrt(torch.clamp(d2.min(1).values, min=0.0)).reshape(GRID, GRID)
        amap = (self.smooth @ dist @ self.smooth.t()).reshape(1, 1, SIZE, SIZE)
        return amap, amap.amax().reshape(1)


def model_input(path: Path) -> Image.Image:
    """256x256 area-averaged (box) resize: simple enough for the browser to reproduce exactly."""
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BOX)


def load_image(path: Path) -> torch.Tensor:
    img = model_input(path)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127


def collect(split_dir: Path) -> list[tuple[Path, int, Path | None]]:
    items: list[tuple[Path, int, Path | None]] = [(p, 0, None) for p in sorted((split_dir / "good").iterdir())]
    for p in sorted((split_dir / "bad").iterdir()):
        items.append((p, 1, split_dir / "masks" / f"{p.stem}.png"))
    return items


def greedy_coreset(features: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """k-center greedy selection on a random 128-d projection (as in PatchCore)."""
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(features.shape[1], 128, generator=g) / math.sqrt(128)
    z = features @ proj
    z_sq = (z * z).sum(1)
    first = int(torch.randint(len(z), (1,), generator=g))
    selected = [first]
    # Squared distances via |a|^2 - 2ab + |b|^2: one matrix-vector product per step.
    min_d = z_sq - 2 * (z @ z[first]) + z_sq[first]
    for _ in range(n - 1):
        j = int(torch.argmax(min_d))
        selected.append(j)
        torch.minimum(min_d, z_sq - 2 * (z @ z[j]) + z_sq[j], out=min_d)
    return torch.tensor(selected)


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3, steps: int = 200) -> float:
    """Area under the per-region-overlap curve up to `fpr_limit`, normalised to [0, 1]."""
    regions = []
    for i, m in enumerate(masks):
        labelled, n = ndimage.label(m)
        regions += [(i, labelled == r) for r in range(1, n + 1)]
    if not regions:
        return float("nan")
    negatives = ~masks
    neg_total = negatives.sum()
    thresholds = np.linspace(maps.max(), maps.min(), steps)
    fprs, pros = [], []
    for t in thresholds:
        pred = maps >= t
        fprs.append((pred & negatives).sum() / neg_total)
        pros.append(np.mean([(pred[i] & r).sum() / r.sum() for i, r in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    f, p = np.concatenate([[0], fprs[keep]]), np.concatenate([[0], pros[keep]])
    if f[-1] < fpr_limit:  # extend to the limit by interpolation
        nxt = np.argmax(fprs > fpr_limit)
        if fprs[nxt] > f[-1]:
            p_lim = p[-1] + (pros[nxt] - p[-1]) * (fpr_limit - f[-1]) / (fprs[nxt] - f[-1])
            f, p = np.append(f, fpr_limit), np.append(p, p_lim)
    return float(np.trapezoid(p, f) / fpr_limit)


def decision_counts(scores: np.ndarray, labels: np.ndarray, review_t: float) -> dict:
    fail = scores >= FAIL_T
    review = (scores >= review_t) & ~fail
    passed = ~fail & ~review
    good, bad = labels == 0, labels == 1
    tp, fn = int((fail & bad).sum()), int((~fail & bad).sum())
    fp, tn = int((fail & good).sum()), int((~fail & good).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "at_fail_threshold": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "escape_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
            "false_reject_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        },
        "with_review_band": {
            "good": {"pass": int((passed & good).sum()), "review": int((review & good).sum()), "fail": int((fail & good).sum())},
            "bad": {"pass": int((passed & bad).sum()), "review": int((review & bad).sum()), "fail": int((fail & bad).sum())},
            "escape_rate": round(int((passed & bad).sum()) / max(1, int(bad.sum())), 4),
            "review_share": round(int(review.sum()) / max(1, len(scores)), 4),
        },
    }


def export_onnx(scorer: Scorer) -> Path:
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    path = WEB_MODELS / "patchcore-resnet18.onnx"
    image = torch.rand(1, 3, SIZE, SIZE)
    bank = torch.rand(500, DIM)
    torch.onnx.export(
        scorer,
        (image, bank),
        str(path),
        input_names=["image", "bank"],
        output_names=["anomaly_map", "score"],
        dynamic_axes={"bank": {0: "m"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    # Fold the exporter's Shape/Slice/Concat subgraphs into constants. ONNX Runtime's
    # WebGPU layout pass rejects the unfolded graph; outputs are bit-identical.
    import onnxruntime as ort

    folded = path.with_suffix(".folded.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    import onnx

    model = onnx.load(str(folded))
    del model.graph.value_info[:]  # stale shape annotations from removed nodes
    onnx.save(model, str(path))
    folded.unlink()
    return path


def run_category(category: str, scorer: Scorer, max_train: int, coreset_ratio: float, reuse_bank: bool = False) -> dict:
    root = DATA / category
    out = ROOT / "models" / category / "patchcore"
    heat_dir = out / "heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted((root / "train" / "good").iterdir())
    random.Random(f"patchcore-lite-{category}").shuffle(train_files)
    train_files = train_files[:max_train]

    bank_file = WEB_MODELS / f"{category}.bank.bin"
    previous_run = out / "train_run.json"
    if reuse_bank and bank_file.exists() and previous_run.exists():
        bank = torch.from_numpy(np.fromfile(bank_file, dtype="<f4").reshape(-1, DIM).copy())
        train_seconds = json.loads(previous_run.read_text())["train_seconds"]
    else:
        t0 = time.perf_counter()
        # Filled in place: concatenating per-batch chunks would briefly need twice the memory.
        feats = torch.empty(len(train_files) * GRID * GRID, DIM)
        with torch.inference_mode():
            for i in range(0, len(train_files), 8):
                batch = torch.stack([load_image(p) for p in train_files[i : i + 8]])
                chunk = scorer.features(batch).reshape(-1, DIM)
                feats[i * GRID * GRID : i * GRID * GRID + len(chunk)] = chunk
        n_bank = max(256, int(len(feats) * coreset_ratio))
        bank = feats[greedy_coreset(feats, n_bank)].contiguous()
        del feats
        train_seconds = time.perf_counter() - t0
    print(f"[{category}] {len(train_files)} train images -> bank {tuple(bank.shape)} in {train_seconds:.0f}s", flush=True)

    raw: dict[str, dict] = {}
    latencies = []
    with torch.inference_mode():
        for split in ("val", "test"):
            items = collect(root / split)
            scores, labels, records, maps, masks = [], [], [], [], []
            for path, label, mask_path in items:
                x = load_image(path)[None]
                t = time.perf_counter()
                amap, score = scorer(x, bank)
                latencies.append(time.perf_counter() - t)
                scores.append(float(score))
                labels.append(label)
                records.append({"file": path.name, "label": "bad" if label else "good"})
                if split == "test":
                    maps.append(amap[0, 0].numpy())
                    masks.append(load_mask(mask_path) if mask_path else np.zeros((SIZE, SIZE), bool))
            raw[split] = {"scores": np.array(scores), "labels": np.array(labels), "records": records, "maps": maps, "masks": masks}

    # Normalisation and thresholds from VALIDATION only.
    vs, vy = raw["val"]["scores"], raw["val"]["labels"]
    raw_t = float(np.quantile(vs[vy == 0], 1 - FAIL_MAX_GOOD_REJECT, method="higher"))
    lo, hi = float(vs.min()), float(vs.max())
    span = max(hi - lo, 1e-6)

    def normalise(v):
        return np.clip((np.asarray(v) - raw_t) / span + 0.5, 0, 1)

    val_bad = np.sort(normalise(vs[vy == 1]))
    k = int(np.floor((1 - REVIEW_RECALL_TARGET) * len(val_bad)))
    review_t = float(min(val_bad[k], FAIL_T)) if len(val_bad) else FAIL_T

    run = {
        "category": category,
        "model": "patchcore",
        "implementation": "patchcore_lite (ResNet18 layer2+layer3, k-center coreset, CPU)",
        "train_images": len(train_files),
        "train_seconds": round(train_seconds, 1),
        "memory_bank": list(bank.shape),
        "coreset_sampling_ratio": coreset_ratio,
        "backbone": "resnet18 (ImageNet, torchvision)",
        "normalisation": {"raw_threshold": round(raw_t, 5), "raw_min": round(lo, 5), "raw_max": round(hi, 5)},
    }
    (out / "train_run.json").write_text(json.dumps(run, indent=2))

    report: dict = {
        "category": category,
        "model": "patchcore",
        "dataset": "VisA (CC BY 4.0)",
        "image_size": SIZE,
        "thresholds": {
            "fail": FAIL_T,
            "review": round(review_t, 5),
            "review_recall_target": REVIEW_RECALL_TARGET,
            "fail_max_good_reject": FAIL_MAX_GOOD_REJECT,
        },
        "train": run,
        "splits": {},
    }
    for split, r in raw.items():
        s, y = normalise(r["scores"]), r["labels"]
        for rec, value in zip(r["records"], s):
            rec["score"] = round(float(value), 5)
        # Ranking metrics use the raw scores: clipping to [0, 1] would create ties.
        unclipped = (r["scores"] - raw_t) / span + 0.5
        fpr, tpr, roc_t = roc_curve(y, unclipped)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).round().astype(int))
        entry = {
            "n_good": int((y == 0).sum()),
            "n_bad": int((y == 1).sum()),
            "image_auroc": round(float(roc_auc_score(y, unclipped)), 4),
            "image_ap": round(float(average_precision_score(y, unclipped)), 4),
            "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4), round(float(min(roc_t[i], 1e3)), 4)] for i in keep],
            **decision_counts(s, y, review_t),
            "images": r["records"],
        }
        if split == "test":
            raw_maps = np.stack(r["maps"])
            masks = np.stack(r["masks"])
            entry["pixel_auroc"] = round(float(roc_auc_score(masks.ravel(), raw_maps.ravel())), 4)
            entry["pixel_aupro_0.3"] = round(aupro(raw_maps, masks), 4)
            maps = normalise(raw_maps)
            for rec, amap in zip(r["records"], maps):
                stem = Path(rec["file"]).stem
                Image.fromarray((amap * 255).astype(np.uint8)).save(heat_dir / f"{rec['label']}-{stem}.png")
        report["splits"][split] = entry

    lat = np.array(latencies[5:]) * 1000
    report["latency_ms"] = {
        "device": "CPU (PyTorch, laptop i7-10750H)",
        "batch": 1,
        "mean": round(float(lat.mean()), 2),
        "p95": round(float(np.percentile(lat, 95)), 2),
        "note": "Feature extraction + nearest-neighbour scoring + smoothing on a 256x256 input; excludes image decoding.",
    }
    (out / "eval.json").write_text(json.dumps(report, indent=1))

    # Browser assets: float32 bank + the normalisation the browser must apply.
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    bank.numpy().astype("<f4").tofile(WEB_MODELS / f"{category}.bank.bin")
    (WEB_MODELS / f"{category}.bank.json").write_text(
        json.dumps(
            {
                "category": category,
                "rows": int(bank.shape[0]),
                "dim": DIM,
                "raw_threshold": raw_t,
                "raw_min": lo,
                "raw_max": hi,
                "fail": FAIL_T,
                "review": review_t,
            },
            indent=1,
        )
    )

    t = report["splits"]["test"]
    print(
        f"[{category}] test: image AUROC {t['image_auroc']}  AP {t['image_ap']}  pixel AUROC {t['pixel_auroc']}  "
        f"AUPRO {t['pixel_aupro_0.3']}  @0.5 recall {t['at_fail_threshold']['recall']} "
        f"FRR {t['at_fail_threshold']['false_reject_rate']}  review-band escape {t['with_review_band']['escape_rate']}  "
        f"latency {report['latency_ms']['mean']} ms",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=["pcb1", "capsules", "cashew"])
    parser.add_argument("--max-train", type=int, default=300, help="good training images used for the memory bank")
    parser.add_argument("--coreset", type=float, default=0.01, help="fraction of patch embeddings kept")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--reuse-bank", action="store_true", help="re-evaluate with saved memory banks instead of retraining")
    args = parser.parse_args()

    torch.manual_seed(0)
    scorer = Scorer(FeatureNet()).eval()
    if not args.no_export:
        print(f"ONNX graph -> {export_onnx(scorer)}", flush=True)
    for category in args.categories:
        run_category(category, scorer, args.max_train, args.coreset, args.reuse_bank)


if __name__ == "__main__":
    main()
