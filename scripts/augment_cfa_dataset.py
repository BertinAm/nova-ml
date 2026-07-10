"""Expand raw CFA currency images into a training-ready ImageFolder dataset.

Takes the scraped/photographed raw images (<raw>/raw/<class>/*.jpg) and
generates N augmented variants per source image, simulating real capture
conditions a blind user's phone will produce:

- random rotation (money can be at any angle)
- perspective warp (camera not parallel to the note)
- brightness/contrast/colour shifts (indoor lighting, torch, daylight)
- Gaussian blur + noise (cheap phone cameras, motion)
- random occlusion patches (fingers holding the note)
- random background composition (table, hand-coloured, dark)

Split: 80% train / 10% val / 10% test BY SOURCE IMAGE (augmented variants
of one source never leak across splits).

    python scripts/augment_cfa_dataset.py \
        --raw datasets/cfa_currency_scraped \
        --out datasets/cfa_currency \
        --per-image 40
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png"}
BACKGROUNDS = [
    (139, 105, 75), (90, 90, 95), (200, 195, 185), (60, 45, 35),
    (170, 140, 110), (30, 30, 30), (210, 180, 160), (120, 120, 125),
]


def random_background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    base = rng.choice(BACKGROUNDS)
    jitter = [min(255, max(0, c + rng.randint(-25, 25))) for c in base]
    bg = Image.new("RGB", size, tuple(jitter))
    # subtle texture noise
    noise = np.random.default_rng(rng.randint(0, 2**31)).integers(
        -12, 12, (size[1], size[0], 3), dtype=np.int16)
    arr = np.clip(np.asarray(bg, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_once(src: Image.Image, rng: random.Random) -> Image.Image:
    img = src.copy()

    # 1. scale the money to occupy 45-90% of a square canvas
    canvas_size = 480
    target = int(canvas_size * rng.uniform(0.45, 0.9))
    scale = target / max(img.size)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))

    # 2. rotation (any angle) with transparent expand
    angle = rng.uniform(0, 360)
    img = img.convert("RGBA").rotate(angle, expand=True, resample=Image.BICUBIC)

    # 3. perspective-ish shear via affine
    shear = rng.uniform(-0.15, 0.15)
    img = img.transform(
        img.size, Image.AFFINE, (1, shear, 0, rng.uniform(-0.1, 0.1), 1, 0),
        resample=Image.BICUBIC)

    # 4. compose onto random background at random position
    bg = random_background((canvas_size, canvas_size), rng)
    max_x = max(1, canvas_size - img.width)
    max_y = max(1, canvas_size - img.height)
    bg.paste(img, (rng.randint(0, max_x), rng.randint(0, max_y)), img)
    out = bg

    # 5. photometric jitter
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.5, 1.5))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.7, 1.3))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.7, 1.3))

    # 6. blur / noise
    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))
    if rng.random() < 0.4:
        arr = np.asarray(out, dtype=np.int16)
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
            0, rng.uniform(3, 10), arr.shape)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    # 7. occlusion patches (fingers holding the note/coin)
    if rng.random() < 0.5:
        draw = ImageDraw.Draw(out)
        for _ in range(rng.randint(1, 3)):
            w, h = rng.randint(30, 90), rng.randint(30, 90)
            x, y = rng.randint(0, canvas_size - w), rng.randint(0, canvas_size - h)
            skin = (rng.randint(120, 210), rng.randint(80, 150), rng.randint(60, 120))
            draw.ellipse([x, y, x + w, y + h], fill=skin)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True,
                        help="Folder containing raw/<class>/*.jpg")
    parser.add_argument("--out", default="datasets/cfa_currency")
    parser.add_argument("--per-image", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw) / "raw"
    if not raw_root.exists():
        raw_root = Path(args.raw)  # allow passing the raw dir directly
    out_root = Path(args.out)
    rng = random.Random(args.seed)

    class_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders under {raw_root}")

    summary: dict[str, dict[str, int]] = {}
    for class_dir in class_dirs:
        cls = class_dir.name
        sources = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not sources:
            print(f"{cls}: no source images, skipping")
            continue
        rng.shuffle(sources)
        # Split BY SOURCE so augmented twins never leak across splits.
        # NOTE: classes with <6 sources fall back to leaky val/test (variants
        # of train sources) — unavoidable at bootstrap scale; replace with
        # real photos before trusting the test metric for those classes.
        n = len(sources)
        n_val = max(1, round(n * 0.1)) if n >= 6 else 0
        n_test = max(1, round(n * 0.1)) if n >= 6 else 0
        split_of: dict[Path, str] = {}
        for i, src in enumerate(sources):
            if i < n_test:
                split_of[src] = "test"
            elif i < n_test + n_val:
                split_of[src] = "val"
            else:
                split_of[src] = "train"

        counts = {"train": 0, "val": 0, "test": 0}
        for src_path in sources:
            split = split_of[src_path]
            dest_dir = out_root / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                src_img = Image.open(src_path).convert("RGB")
            except Exception:
                continue
            # keep the clean original too
            src_img.save(dest_dir / f"{src_path.stem}_orig.jpg", quality=90)
            counts[split] += 1
            # fewer variants for val/test — they should stay close to real
            n_aug = args.per_image if split == "train" else max(3, args.per_image // 8)
            for k in range(n_aug):
                aug = augment_once(src_img, rng)
                aug.save(dest_dir / f"{src_path.stem}_aug{k:03d}.jpg", quality=88)
                counts[split] += 1
            # Leaky fallback for tiny classes: derive val/test variants from
            # train sources so no split is ever empty.
            if split == "train" and n_val == 0:
                for fallback_split in ("val", "test"):
                    fb_dir = out_root / fallback_split / cls
                    fb_dir.mkdir(parents=True, exist_ok=True)
                    for k in range(4):
                        aug = augment_once(src_img, rng)
                        aug.save(fb_dir / f"{src_path.stem}_fb{k}.jpg", quality=88)
                        counts[fallback_split] += 1
        summary[cls] = counts
        print(f"{cls}: {counts}")

    print("\n=== Dataset summary ===")
    grand = {"train": 0, "val": 0, "test": 0}
    for cls, counts in summary.items():
        for k in grand:
            grand[k] += counts[k]
    print(f"  classes: {len(summary)}")
    print(f"  train: {grand['train']}  val: {grand['val']}  test: {grand['test']}")
    print(f"  -> {out_root}")
    print("\nUpload to Kaggle as a private dataset, then run notebook 02.")


if __name__ == "__main__":
    main()
