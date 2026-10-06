"""Export the held-out test split (and model results, when present) for the web workbench.

Writes web/public/demo/:
    index.json                       categories + whether model results exist
    <category>/manifest.json         items, dataset counts, model metrics/thresholds/ROC
    <category>/img/<id>.jpg          display image (640 px wide)
    <category>/mask/<id>.png         ground-truth defect mask (transparent + red), defects only
    <category>/heat/<id>.png         model anomaly heatmap (transparent colormap), when evaluated
    <category>/input/<id>.webp       exact 256x256 model input (box resize, lossless), so the
                                     in-browser model sees the same pixels as the evaluation

Needs only Pillow + numpy (no torch), so it runs even when the machine is short on memory.
Run again after ml/evaluate.py to merge real scores and heatmaps.

Usage:
    python ml/export_web.py --categories pcb1 capsules cashew
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
OUT = ROOT / "web" / "public" / "demo"
WIDTH = 640
MODEL_PREFERENCE = ("patchcore", "efficientad")  # first one with an eval.json is shown in the demo

TITLES = {
    "pcb1": ("PCB1: ultrasonic sensor board", "Electronics assembly. Defects: bent pins, melted plastic, scratches, missing parts."),
    "capsules": ("Capsules: soft-gel tray", "Pharma / packaging, many parts per image. Defects: bubbles, discoloration, scratches, leaks, misshapes."),
    "cashew": ("Cashew: single nut", "Food grading. Defects: burnt, breakage, holes, spots, scratches, stuck together."),
}


def load_annotations(tar_path: Path, category: str) -> dict[str, list[str]]:
    """Map 'Anomaly/<file>' -> defect types, cached next to the prepared data."""
    cache = DATA / category / "anno.json"
    if cache.exists():
        return json.loads(cache.read_text())
    anno: dict[str, list[str]] = {}
    with tarfile.open(tar_path) as tar:
        member = next(m for m in tar if m.name.endswith(f"{category}/image_anno.csv"))
        for row in csv.DictReader(io.StringIO(tar.extractfile(member).read().decode("utf-8"))):
            if row["label"] != "normal":
                folder, name = Path(row["image"]).parent.name, Path(row["image"]).name
                anno[f"{folder}/{name}"] = [d.strip() for d in row["label"].split(",")]
    cache.write_text(json.dumps(anno))
    return anno


def colormap(amap: np.ndarray) -> np.ndarray:
    """Transparent-to-hot overlay: low scores invisible, high scores yellow/red."""
    a = np.clip(amap, 0, 1)
    stops = np.array([[0.0, 0, 0, 0, 0], [0.35, 56, 198, 236, 0], [0.55, 56, 198, 236, 110],
                      [0.75, 246, 168, 37, 170], [1.0, 229, 72, 77, 215]], dtype=np.float32)
    out = np.zeros((*a.shape, 4), dtype=np.float32)
    for c in range(4):
        out[..., c] = np.interp(a, stops[:, 0], stops[:, c + 1])
    return out.astype(np.uint8)


def load_eval(category: str) -> tuple[str | None, dict | None]:
    for name in MODEL_PREFERENCE:
        path = ROOT / "models" / category / name / "eval.json"
        if path.exists():
            return name, json.loads(path.read_text())
    return None, None


def export_category(category: str, tar_path: Path) -> dict:
    anno = load_annotations(tar_path, category)
    split = json.loads((DATA / category / "split.json").read_text())["files"]
    model_name, ev = load_eval(category)
    scores = {}
    if ev:
        scores = {(r["label"], r["file"]): r["score"] for r in ev["splits"]["test"]["images"]}

    out = OUT / category
    for sub in ("img", "mask", "heat", "input"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    items = []
    for kind in ("good", "bad"):
        for name in split.get(f"test/{kind}", []):
            stem = Path(name).stem
            item_id = f"{kind}-{stem}"
            src = DATA / category / "test" / kind / name
            img = Image.open(src).convert("RGB")
            h = round(img.height * WIDTH / img.width)
            # Display image and model input depend only on the dataset: skip them on re-runs.
            display, model_input = out / "img" / f"{item_id}.jpg", out / "input" / f"{item_id}.webp"
            if not display.exists():
                img.resize((WIDTH, h), Image.LANCZOS).save(display, quality=82, optimize=True)
            if not model_input.exists():
                # Same resize as ml/patchcore_lite.py model_input(); lossless so browser scores match.
                img.resize((256, 256), Image.BOX).save(model_input, lossless=True, quality=100, method=6)

            item: dict = {"id": item_id, "label": kind, "defects": anno.get(f"Anomaly/{name}", []) if kind == "bad" else [], "input": True}
            if kind == "bad":
                m = np.asarray(Image.open(DATA / category / "test" / "masks" / f"{stem}.png").convert("L").resize((WIDTH, h), Image.NEAREST)) > 127
                rgba = np.zeros((h, WIDTH, 4), np.uint8)
                rgba[m] = (229, 72, 77, 255)
                Image.fromarray(rgba).save(out / "mask" / f"{item_id}.png", optimize=True)
                item["mask"] = True

            heat_src = ROOT / "models" / category / str(model_name) / "heatmaps" / f"{item_id}.png"
            if ev and (kind, name) in scores:
                item["score"] = scores[(kind, name)]
                if heat_src.exists():
                    amap = np.asarray(Image.open(heat_src).convert("L").resize((WIDTH, h), Image.BILINEAR), dtype=np.float32) / 255
                    Image.fromarray(colormap(amap)).save(out / "heat" / f"{item_id}.png", optimize=True)
                    item["heat"] = True
            items.append(item)

    title, blurb = TITLES.get(category, (category, ""))
    manifest: dict = {
        "category": category,
        "title": title,
        "blurb": blurb,
        "dataset": "VisA (Amazon Science, CC BY 4.0), held-out test split",
        "counts": {
            "train_good": len(split.get("train/good", [])),
            "val": len(split.get("val/good", [])) + len(split.get("val/bad", [])),
            "test_good": len(split.get("test/good", [])),
            "test_bad": len(split.get("test/bad", [])),
        },
        "model": None,
        "items": items,
    }
    if ev:
        t = ev["splits"]["test"]
        manifest["model"] = {
            "name": model_name,
            "thresholds": ev["thresholds"],
            "latency_ms": ev["latency_ms"],
            "train": {k: ev["train"].get(k) for k in ("train_images", "train_seconds", "gpu", "backbone", "max_steps", "memory_bank", "implementation")},
            "test": {k: t.get(k) for k in ("image_auroc", "image_ap", "pixel_auroc", "pixel_aupro_0.3", "roc",
                                            "at_fail_threshold", "with_review_band", "n_good", "n_bad")},
        }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return {"category": category, "title": title, "hasModel": bool(ev), "model": model_name, "n": len(items)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=["pcb1", "capsules", "cashew"])
    parser.add_argument("--tar", type=Path, default=ROOT / "data" / "VisA_20220922.tar")
    args = parser.parse_args()
    index = [export_category(c, args.tar) for c in args.categories]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps({"categories": index}, indent=1))
    for c in index:
        print(c)


if __name__ == "__main__":
    main()
