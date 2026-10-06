"""Prepare VisA categories in an MVTec-style folder layout with a validation split.

VisA (Amazon Science, CC BY 4.0) ships an official split in `split_csv/1cls.csv`:
train = normal images only, test = normal + anomalous. It has no validation split,
so we carve one out of the official test set (stratified, seeded). Thresholds are
tuned on `val` and every reported number comes from the untouched `test` split.

Output layout per category:
    data/visa/<category>/
        train/good/*.JPG
        val/good/*.JPG     val/bad/*.JPG     val/masks/*.png
        test/good/*.JPG    test/bad/*.JPG    test/masks/*.png
        split.json         (exact file lists, for reproducibility)

Usage:
    python ml/prepare_visa.py --tar data/VisA_20220922.tar --categories pcb1 capsules cashew
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

VAL_FRACTION = 0.4  # share of the official test set moved to validation
SEED = 42


def read_split_csv(tar: tarfile.TarFile) -> list[dict[str, str]]:
    member = next(m for m in tar.getmembers() if m.name.endswith("split_csv/1cls.csv"))
    data = tar.extractfile(member).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(data)))


def stratified_val(rows: list[dict[str, str]], rng: random.Random) -> set[str]:
    """Pick VAL_FRACTION of the test images per label (normal / anomaly)."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r["image"])
    val: set[str] = set()
    for images in by_label.values():
        images = sorted(images)
        rng.shuffle(images)
        val.update(images[: round(len(images) * VAL_FRACTION)])
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tar", type=Path, default=Path("data/VisA_20220922.tar"))
    parser.add_argument("--out", type=Path, default=Path("data/visa"))
    parser.add_argument("--categories", nargs="+", default=["pcb1", "capsules", "cashew"])
    args = parser.parse_args()

    with tarfile.open(args.tar) as tar:
        rows = read_split_csv(tar)
        # CSV paths look like "pcb1/Data/Images/Normal/0000.JPG"; the tar may add a leading folder.
        members: dict[str, tarfile.TarInfo] = {}
        for m in tar.getmembers():
            if m.isfile():
                members[m.name] = m
                members.setdefault(m.name.split("/", 1)[-1], m)

        def find(path: str) -> tarfile.TarInfo:
            if path not in members:
                raise FileNotFoundError(f"{path} not found in {args.tar}")
            return members[path]

        for category in args.categories:
            cat_rows = [r for r in rows if r["object"] == category]
            if not cat_rows:
                raise SystemExit(f"Unknown VisA category: {category}")
            rng = random.Random(f"{SEED}-{category}")
            test_rows = [r for r in cat_rows if r["split"] == "test"]
            val_images = stratified_val(test_rows, rng)

            root = args.out / category
            split: dict[str, list[str]] = defaultdict(list)
            for r in cat_rows:
                part = "train" if r["split"] == "train" else ("val" if r["image"] in val_images else "test")
                kind = "good" if r["label"] == "normal" else "bad"
                src = find(r["image"])
                name = Path(r["image"]).name
                dest = root / part / kind / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(src).read())
                split[f"{part}/{kind}"].append(name)

                if kind == "bad" and r.get("mask"):
                    mask = np.array(Image.open(io.BytesIO(tar.extractfile(find(r["mask"])).read())))
                    binary = ((mask > 0).any(axis=-1) if mask.ndim == 3 else mask > 0).astype(np.uint8) * 255
                    mask_dest = root / part / "masks" / (Path(name).stem + ".png")
                    mask_dest.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(binary).save(mask_dest)

            (root / "split.json").write_text(
                json.dumps({"seed": SEED, "val_fraction": VAL_FRACTION, "files": {k: sorted(v) for k, v in split.items()}}, indent=1)
            )
            counts = {k: len(v) for k, v in sorted(split.items())}
            print(f"{category}: {counts}")


if __name__ == "__main__":
    main()
