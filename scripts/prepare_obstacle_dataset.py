"""Merge Detectra + VisDrone into one YOLO-format dataset for MOD-01.

On Kaggle, attach these datasets to the notebook:
  - jhontroya/dectectra-dataset      (primary — obstacles for BVI users)
  - kushagrapandya/visdrone-dataset  (dense pedestrians/vehicles)

then run:
    python scripts/prepare_obstacle_dataset.py \
        --detectra /kaggle/input/dectectra-dataset \
        --visdrone /kaggle/input/visdrone-dataset \
        --out /kaggle/working/datasets/obstacle_combined
"""
import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_split_dirs(root: Path, split: str) -> tuple[Path | None, Path | None]:
    """Locate images/<split> and labels/<split> allowing for the slightly
    different folder layouts Kaggle datasets ship with."""
    candidates = [
        (root / "images" / split, root / "labels" / split),
        (root / split / "images", root / split / "labels"),
        (root / f"VisDrone2019-DET-{split}" / "images",
         root / f"VisDrone2019-DET-{split}" / "labels"),
    ]
    for imgs, lbls in candidates:
        if imgs.is_dir():
            return imgs, (lbls if lbls.is_dir() else None)
    return None, None


def copy_split(src_images: Path, src_labels: Path | None, out: Path, split: str, prefix: str) -> int:
    count = 0
    for img in src_images.iterdir():
        if img.suffix.lower() not in IMG_EXTS:
            continue
        shutil.copy(img, out / "images" / split / f"{prefix}_{img.name}")
        if src_labels:
            lbl = src_labels / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, out / "labels" / split / f"{prefix}_{lbl.name}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detectra", required=True)
    parser.add_argument("--visdrone", default=None)
    parser.add_argument("--out", default="datasets/obstacle_combined")
    args = parser.parse_args()

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    total = {"train": 0, "val": 0}
    for name, root in [("detectra", Path(args.detectra))] + (
        [("visdrone", Path(args.visdrone))] if args.visdrone else []
    ):
        for split in ("train", "val"):
            imgs, lbls = find_split_dirs(root, split)
            if imgs is None:
                print(f"WARNING: no {split} split found in {root} — check folder layout")
                continue
            n = copy_split(imgs, lbls, out, split, name)
            total[split] += n
            print(f"{name}/{split}: copied {n} images")

    print(f"Combined dataset at {out}: {total['train']} train / {total['val']} val images")
    print("Now update configs/obstacle_data.yaml `path:` to point here and verify class list.")


if __name__ == "__main__":
    main()
