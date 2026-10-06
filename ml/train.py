"""Train anomaly-detection models on prepared VisA categories (good parts only).

Anomalib's own validation/threshold step runs on our `val` split; the `test` split is
never touched here. Run `ml/evaluate.py` afterwards for the held-out numbers.

Usage:
    python ml/train.py --category pcb1 --model efficientad --max-steps 6000
    python ml/train.py --category pcb1 --model patchcore
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import _env  # noqa: F401  (must precede torch/anomalib: moves caches off the system drive)
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
RUNS = ROOT / "results"

# PatchCore keeps every patch embedding of every training image in memory before coreset
# sampling (~6 MB per image with wide_resnet50_2 at 256 px). A seeded subset keeps it
# within a 4 GB GPU / 8 GB RAM laptop. EfficientAD trains on the full train split.
PATCHCORE_MAX_TRAIN_IMAGES = 200


def patchcore_subset(category: str) -> Path:
    """Copy a seeded subset of train/good into train_subset/good (idempotent)."""
    src = DATA / category / "train" / "good"
    dst = DATA / category / f"train_subset{PATCHCORE_MAX_TRAIN_IMAGES}" / "good"
    files = sorted(src.iterdir())
    random.Random(f"patchcore-{category}").shuffle(files)
    chosen = files[:PATCHCORE_MAX_TRAIN_IMAGES]
    if not dst.exists() or len(list(dst.iterdir())) != len(chosen):
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for f in chosen:
            shutil.copy2(f, dst / f.name)
    return dst.relative_to(DATA / category)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    parser.add_argument("--max-steps", type=int, default=6000, help="EfficientAD training steps (batch size 1)")
    parser.add_argument("--backbone", default="wide_resnet50_2", help="PatchCore backbone (resnet18 needs far less memory)")
    parser.add_argument("--imagenet-dir", type=Path, default=ROOT / "data" / "imagenette")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    root = DATA / args.category

    if args.model == "patchcore":
        normal_dir = patchcore_subset(args.category)
        model = Patchcore(backbone=args.backbone, layers=("layer2", "layer3"), coreset_sampling_ratio=0.1)
        train_bs, engine_kwargs = 8, {"max_epochs": 1}
    else:
        normal_dir = Path("train/good")
        model = EfficientAd(imagenet_dir=args.imagenet_dir, model_size="small")
        train_bs, engine_kwargs = 1, {"max_steps": args.max_steps}

    datamodule = Folder(
        name=args.category,
        root=root,
        normal_dir=normal_dir,
        # Anomalib validates/thresholds on OUR validation split; test stays held out.
        normal_test_dir="val/good",
        abnormal_dir="val/bad",
        mask_dir="val/masks",
        val_split_mode="same_as_test",
        train_batch_size=train_bs,
        eval_batch_size=8,
        num_workers=0,  # Windows: avoid worker spawn overhead / pickling issues
        seed=42,
    )

    run_dir = RUNS / args.model / args.category
    engine = Engine(default_root_dir=run_dir, accelerator="gpu", devices=1, logger=False, **engine_kwargs)

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_seconds = time.perf_counter() - t0
    val_metrics = engine.test(model=model, datamodule=datamodule)[0]

    ckpt = Path(engine.trainer.checkpoint_callback.best_model_path or engine.trainer.checkpoint_callback.last_model_path)
    out = ROOT / "models" / args.category / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_run.json").write_text(
        json.dumps(
            {
                "category": args.category,
                "model": args.model,
                "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
                "train_images": len(list((root / normal_dir).iterdir())),
                "train_seconds": round(train_seconds, 1),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "anomalib_val_metrics": {k: round(float(v), 4) for k, v in val_metrics.items()},
                **({"max_steps": args.max_steps} if args.model == "efficientad" else {"coreset_sampling_ratio": 0.1, "backbone": args.backbone}),
            },
            indent=2,
        )
    )
    print(f"[done] {args.model}/{args.category} in {train_seconds:.0f}s  val={val_metrics}  ckpt={ckpt}")


if __name__ == "__main__":
    main()
