"""LFW face-verification evaluation for the MobileFaceNet student (MOD-05).

Works against the Kaggle mirror `jessicali9530/lfw-dataset`, which ships:
  lfw-deepfunneled/lfw-deepfunneled/<Person>/<Person>_NNNN.jpg
  matchpairsDevTest.csv     (name, imagenum1, imagenum2)
  mismatchpairsDevTest.csv  (name, imagenum1, name.1, imagenum2)
  pairs.csv                 (combined; column layout varies)

Loads whichever pair files exist, computes cosine similarities with the
trained checkpoint, and reports verification accuracy at the configured
threshold plus the best-threshold accuracy and TAR@FAR=1e-3.

    python scripts/evaluate_lfw.py \
        --checkpoint /kaggle/working/checkpoints/face_embedding_best.pth \
        --lfw-root /kaggle/input/datasets/jessicali9530/lfw-dataset \
        --out /kaggle/working/evaluation/lfw_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TFM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def find_images_root(lfw_root: Path) -> Path:
    """Locate the folder whose children are person folders."""
    candidates = [
        lfw_root / "lfw-deepfunneled" / "lfw-deepfunneled",
        lfw_root / "lfw-deepfunneled",
        lfw_root / "lfw",
    ]
    for c in candidates:
        if c.is_dir() and any(p.is_dir() for p in list(c.iterdir())[:5]):
            return c
    # Last resort: search for a well-known LFW identity
    hits = list(lfw_root.rglob("George_W_Bush"))
    if hits:
        return hits[0].parent
    raise SystemExit(f"Could not find LFW images under {lfw_root}")


def img_path(images_root: Path, name: str, num: int) -> Path:
    return images_root / name / f"{name}_{int(num):04d}.jpg"


def load_pairs(lfw_root: Path, images_root: Path) -> list[tuple[Path, Path, int]]:
    """Return [(img1, img2, is_same)] from whichever CSVs exist."""
    import csv

    pairs: list[tuple[Path, Path, int]] = []

    def add(p1: Path, p2: Path, same: int):
        if p1.exists() and p2.exists():
            pairs.append((p1, p2, same))

    match_files = list(lfw_root.glob("matchpairs*.csv"))
    mismatch_files = list(lfw_root.glob("mismatchpairs*.csv"))

    for f in match_files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 3 or not row[1].strip().isdigit():
                    continue  # header or malformed
                name, n1, n2 = row[0].strip(), row[1], row[2]
                add(img_path(images_root, name, int(n1)),
                    img_path(images_root, name, int(n2)), 1)

    for f in mismatch_files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 4 or not row[1].strip().isdigit():
                    continue
                name1, n1, name2, n2 = (row[0].strip(), row[1],
                                        row[2].strip(), row[3])
                add(img_path(images_root, name1, int(n1)),
                    img_path(images_root, name2, int(n2)), 0)

    if not pairs:
        # Fallback: the combined pairs.csv (match rows have 3 useful cols,
        # mismatch rows 4).
        for f in lfw_root.glob("pairs*.csv"):
            with open(f, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    row = [c.strip() for c in row if c.strip()]
                    if len(row) == 3 and row[1].isdigit() and row[2].isdigit():
                        add(img_path(images_root, row[0], int(row[1])),
                            img_path(images_root, row[0], int(row[2])), 1)
                    elif len(row) == 4 and row[1].isdigit() and row[3].isdigit():
                        add(img_path(images_root, row[0], int(row[1])),
                            img_path(images_root, row[2], int(row[3])), 0)

    if not pairs:
        raise SystemExit(f"No usable pair CSVs found under {lfw_root}: "
                         f"{[p.name for p in lfw_root.glob('*.csv')]}")
    return pairs


@torch.no_grad()
def embed_batch(model, paths: list[Path], batch_size: int = 128) -> np.ndarray:
    from PIL import Image

    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([TFM(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch_size]]).to(DEVICE)
        emb = F.normalize(model(batch), p=2, dim=1)
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lfw-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--out", default="evaluation/lfw_results.json")
    args = parser.parse_args()

    from train_face_embedding import MobileFaceNet

    model = MobileFaceNet().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    lfw_root = Path(args.lfw_root)
    images_root = find_images_root(lfw_root)
    pairs = load_pairs(lfw_root, images_root)
    n_same = sum(s for _, _, s in pairs)
    print(f"LFW pairs: {len(pairs)} ({n_same} same / {len(pairs) - n_same} different)")

    # Embed unique images once
    unique = sorted({p for a, b, _ in pairs for p in (a, b)})
    idx = {p: i for i, p in enumerate(unique)}
    embs = embed_batch(model, unique)

    sims = np.array([float(embs[idx[a]] @ embs[idx[b]]) for a, b, _ in pairs])
    gt = np.array([s for _, _, s in pairs])

    acc_at_threshold = float(((sims >= args.threshold) == gt).mean())
    # Best threshold sweep (standard LFW protocol reports best-threshold acc)
    ths = np.linspace(-1, 1, 401)
    accs = [((sims >= t) == gt).mean() for t in ths]
    best_i = int(np.argmax(accs))
    # TAR @ FAR=1e-3
    neg = np.sort(sims[gt == 0])[::-1]
    far_thresh = neg[max(0, int(len(neg) * 1e-3) - 1)] if len(neg) else 1.0
    tar = float((sims[gt == 1] >= far_thresh).mean()) if (gt == 1).any() else 0.0

    results = {
        "lfw_pairs": len(pairs),
        "accuracy_at_threshold_0.75": acc_at_threshold,
        "best_accuracy": float(accs[best_i]),
        "best_threshold": float(ths[best_i]),
        "tar_at_far_1e-3": tar,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
