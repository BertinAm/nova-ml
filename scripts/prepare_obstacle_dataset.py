"""Merge Detectra + VisDrone into one YOLO-format dataset for MOD-01,
auto-discovering each dataset's folder layout and auto-generating the
final training YAML (correct nc/names) — no manual config editing.

Key correctness detail: Detectra and VisDrone label files both use class
indices starting at 0, but the indices mean different classes. VisDrone
label files are therefore REWRITTEN during the merge, remapping each
class index into the combined class list (deduplicating classes whose
names already exist in Detectra).

    python scripts/prepare_obstacle_dataset.py \
        --detectra /kaggle/input/dectectra-dataset \
        --visdrone /kaggle/input/visdrone-dataset \
        --out /kaggle/working/datasets/obstacle_combined \
        --yaml-out /kaggle/working/obstacle_data.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# VisDrone2019-DET class list (fixed, indices 0-9 in its YOLO label files)
VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
# VisDrone -> canonical name so duplicates dedupe against Detectra
VISDRONE_CANONICAL = {"pedestrian": "person", "people": "person", "motor": "motorcycle"}


def find_class_names(root: Path) -> list[str] | None:
    """Locate a data.yaml (any depth) and read its `names`."""
    for candidate in sorted(root.rglob("*.yaml")):
        try:
            data = yaml.safe_load(candidate.read_text())
        except Exception:
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if not names:
            continue
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        print(f"Found class names in {candidate} ({len(names)} classes)")
        return [str(n) for n in names]
    return None


def find_image_dirs(root: Path) -> list[tuple[Path, str]]:
    """Return (dir, split) for every directory that directly contains images.
    Split is inferred from path parts; defaults to 'train'."""
    results = []
    for d in sorted({p.parent for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS}):
        parts = [part.lower() for part in d.parts]
        if any("val" in part for part in parts):
            split = "val"
        elif any("test" in part for part in parts):
            split = "test"
        else:
            split = "train"
        results.append((d, split))
    return results


def label_file_for(img: Path) -> Path | None:
    """Find the YOLO .txt label for an image, trying common conventions."""
    candidates = [
        img.with_suffix(".txt"),  # labels beside images
        Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt"),
        Path(str(img.parent).replace("Images", "labels")) / (img.stem + ".txt"),
    ]
    for c in candidates:
        if c != img and c.exists():
            return c
    return None


def remap_label(src: Path, dest: Path, index_map: dict[int, int]) -> None:
    out_lines = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls not in index_map:
            continue  # class not in mapping (e.g. VisDrone 'ignored regions')
        out_lines.append(" ".join([str(index_map[cls])] + parts[1:]))
    dest.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def ingest(root: Path, prefix: str, out: Path, index_map: dict[int, int] | None,
           val_fraction: float = 0.1) -> dict[str, int]:
    """Copy images+labels into the combined tree. If a dataset has no val
    split, hold out `val_fraction` of train images deterministically."""
    counts = {"train": 0, "val": 0}
    image_dirs = find_image_dirs(root)
    if not image_dirs:
        print(f"WARNING: no images found anywhere under {root}")
        return counts

    has_val = any(split == "val" for _, split in image_dirs)
    rng = random.Random(42)

    for d, split in image_dirs:
        if split == "test":
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = label_file_for(img)
            if lbl is None:
                continue  # unlabelled image is useless for detection training
            eff_split = split
            if not has_val and split == "train" and rng.random() < val_fraction:
                eff_split = "val"
            img_dest = out / "images" / eff_split / f"{prefix}_{img.name}"
            lbl_dest = out / "labels" / eff_split / f"{prefix}_{img.stem}.txt"
            shutil.copy(img, img_dest)
            if index_map is None:
                shutil.copy(lbl, lbl_dest)
            else:
                remap_label(lbl, lbl_dest, index_map)
            counts[eff_split] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detectra", required=True)
    parser.add_argument("--visdrone", default=None)
    parser.add_argument("--out", default="datasets/obstacle_combined")
    parser.add_argument("--yaml-out", default="configs/obstacle_data_generated.yaml")
    args = parser.parse_args()

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ── Detectra: classes come from its own data.yaml ──────────────────
    detectra_root = Path(args.detectra)
    names = find_class_names(detectra_root)
    if names is None:
        raise SystemExit(
            f"Could not find a data.yaml with class names under {detectra_root}. "
            "Run `find <dir> -name '*.yaml'` and inspect the dataset layout."
        )
    c1 = ingest(detectra_root, "detectra", out, index_map=None)
    print(f"Detectra: {c1}")

    # ── VisDrone: remap its indices into the merged class list ─────────
    if args.visdrone:
        name_to_idx = {n.lower(): i for i, n in enumerate(names)}
        vis_map: dict[int, int] = {}
        for vi, vname in enumerate(VISDRONE_NAMES):
            canonical = VISDRONE_CANONICAL.get(vname, vname)
            if canonical.lower() in name_to_idx:
                vis_map[vi] = name_to_idx[canonical.lower()]
            else:
                vis_map[vi] = len(names)
                names.append(canonical)
                name_to_idx[canonical.lower()] = vis_map[vi]
        c2 = ingest(Path(args.visdrone), "visdrone", out, index_map=vis_map)
        print(f"VisDrone: {c2} (index remap: {vis_map})")

    # ── Generate the training YAML — correct nc/names, no manual step ──
    cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
        # Chest-mounted camera augmentation profile
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 5.0, "translate": 0.1, "scale": 0.5,
        "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1,
    }
    yaml_out = Path(args.yaml_out)
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(yaml.dump(cfg, sort_keys=False))

    n_train = len(list((out / "images" / "train").iterdir()))
    n_val = len(list((out / "images" / "val").iterdir()))
    print(f"\nCombined: {n_train} train / {n_val} val images, {len(names)} classes")
    print(f"Training config written to {yaml_out} — pass it as --data to train_obstacle.py")
    if n_train == 0:
        raise SystemExit("ERROR: 0 training images — dataset layout not recognised. "
                         "Inspect with `find <input-dir> -maxdepth 3 -type d`.")


if __name__ == "__main__":
    main()
