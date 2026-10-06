"""Evaluate a trained model on the validation and held-out test splits.

Scores are the model's normalised outputs (0..1, 0.5 = anomalib's F1-optimal threshold
fitted on validation). Every image is resized to 256x256 before inference, which is what
the browser demo does too, so Python and browser results come from the same pipeline.

Decision bands (all fitted on VALIDATION only):
    score >= FAIL threshold (0.5)            -> FAIL
    REVIEW threshold <= score < 0.5          -> REVIEW (human check)
    score < REVIEW threshold                 -> PASS
The REVIEW threshold is the highest value that still sends >= 98% of validation defects
to FAIL or REVIEW.

Writes models/<category>/<model>/eval.json and heatmaps/<test image>.png (8-bit maps).

Usage:
    python ml/evaluate.py --category pcb1 --model efficientad
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import _env  # noqa: F401  (must precede torch/anomalib: moves caches off the system drive)
import torch
from anomalib.metrics.aupro import _AUPRO
from anomalib.models import EfficientAd, Patchcore
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
SIZE = 256
FAIL_T = 0.5
REVIEW_RECALL_TARGET = 0.98
MODELS = {"patchcore": Patchcore, "efficientad": EfficientAd}


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127


def collect(split_dir: Path) -> list[tuple[Path, int, Path | None]]:
    items: list[tuple[Path, int, Path | None]] = [(p, 0, None) for p in sorted((split_dir / "good").iterdir())]
    for p in sorted((split_dir / "bad").iterdir()):
        items.append((p, 1, split_dir / "masks" / f"{p.stem}.png"))
    return items


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
            # Defects that were auto-passed without a human seeing them.
            "escape_rate": round(float((passed & bad).sum() / bad.sum()), 4) if bad.any() else 0.0,
            "review_share": round(float(review.mean()), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()

    out = ROOT / "models" / args.category / args.model
    run = json.loads((out / "train_run.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODELS[args.model].load_from_checkpoint(ROOT / run["checkpoint"], map_location=device).eval().to(device)

    results: dict[str, dict] = {}
    heat_dir = out / "heatmaps"
    heat_dir.mkdir(exist_ok=True)
    latencies: list[float] = []

    for split in ("val", "test"):
        items = collect(DATA / args.category / split)
        scores, labels, records = [], [], []
        maps, masks = [], []
        with torch.inference_mode():
            for path, label, mask_path in items:
                x = load_image(path)[None].to(device)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)

                score = float(pred.pred_score.reshape(-1)[0])
                amap = pred.anomaly_map.reshape(SIZE, SIZE).float().cpu().numpy()
                scores.append(score)
                labels.append(label)
                records.append({"file": path.name, "label": "bad" if label else "good", "score": round(score, 5)})

                if split == "test":
                    maps.append(amap)
                    masks.append(load_mask(mask_path) if mask_path else np.zeros((SIZE, SIZE), bool))
                    Image.fromarray((np.clip(amap, 0, 1) * 255).astype(np.uint8)).save(heat_dir / f"{'bad' if label else 'good'}-{path.stem}.png")

        s, y = np.array(scores), np.array(labels)
        results[split] = {"scores": s, "labels": y, "records": records, "maps": maps, "masks": masks}

    # REVIEW threshold from validation only.
    val_bad = np.sort(results["val"]["scores"][results["val"]["labels"] == 1])
    k = int(np.floor((1 - REVIEW_RECALL_TARGET) * len(val_bad)))
    review_t = float(min(val_bad[k], FAIL_T)) if len(val_bad) else FAIL_T

    report: dict = {
        "category": args.category,
        "model": args.model,
        "dataset": "VisA (CC BY 4.0)",
        "image_size": SIZE,
        "thresholds": {"fail": FAIL_T, "review": round(review_t, 5), "review_recall_target": REVIEW_RECALL_TARGET},
        "train": run,
        "splits": {},
    }
    for split, r in results.items():
        s, y = r["scores"], r["labels"]
        fpr, tpr, thr = roc_curve(y, s)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).round().astype(int))
        entry = {
            "n_good": int((y == 0).sum()),
            "n_bad": int((y == 1).sum()),
            "image_auroc": round(float(roc_auc_score(y, s)), 4),
            "image_ap": round(float(average_precision_score(y, s)), 4),
            "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4), round(float(min(thr[i], 1e3)), 4)] for i in keep],
            **decision_counts(s, y, review_t),
            "images": r["records"],
        }
        if split == "test" and r["maps"]:
            maps = torch.from_numpy(np.stack(r["maps"]))
            masks = torch.from_numpy(np.stack(r["masks"]))
            entry["pixel_auroc"] = round(float(roc_auc_score(masks.flatten().numpy(), maps.flatten().numpy())), 4)
            aupro = _AUPRO(fpr_limit=0.3)
            aupro.update(maps, masks.int())
            entry["pixel_aupro_0.3"] = round(float(aupro.compute()), 4)
        report["splits"][split] = entry

    lat = np.array(latencies[5:]) * 1000  # skip warm-up
    report["latency_ms"] = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "batch": 1,
        "mean": round(float(lat.mean()), 2),
        "p95": round(float(np.percentile(lat, 95)), 2),
        "note": "Model forward incl. pre/post-processing on a 256x256 input; excludes image decoding.",
    }

    (out / "eval.json").write_text(json.dumps(report, indent=1))
    t = report["splits"]["test"]
    print(
        f"[{args.model}/{args.category}] test: image AUROC {t['image_auroc']}  AP {t['image_ap']}  "
        f"pixel AUROC {t.get('pixel_auroc')}  AUPRO {t.get('pixel_aupro_0.3')}  "
        f"@0.5 recall {t['at_fail_threshold']['recall']} FRR {t['at_fail_threshold']['false_reject_rate']}  "
        f"review-band escape {t['with_review_band']['escape_rate']} review share {t['with_review_band']['review_share']}  "
        f"latency {report['latency_ms']['mean']} ms"
    )


if __name__ == "__main__":
    main()
