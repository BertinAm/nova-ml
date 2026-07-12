"""MOD-05 (embed stage): MobileFaceNet training with ArcFace loss +
embedding-level distillation from a pretrained ArcFace R100 teacher.

The teacher comes from InsightFace's model zoo (buffalo_l bundle) — we use
its recognition model to produce target embeddings; the student learns to
mimic them (MSE on L2-normalised embeddings) while also minimising its own
ArcFace classification loss.

Dataset: identity-labelled ImageFolder (one folder per identity), e.g. a
VGGFace2 subset. Evaluation: LFW pairs.

    python scripts/train_face_embedding.py \
        --data-dir /kaggle/input/vggface2-subset \
        --epochs 50
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from nova_common import ensure_dirs
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 512


# ── MobileFaceNet (compact definition, ~1.0M params) ─────────────────
class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k=1, s=1, p=0, dw=False, linear=False):
        super().__init__()
        groups = inp if dw else 1
        self.conv = nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = None if linear else nn.PReLU(oup)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x if self.act is None else self.act(x)


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        hid = inp * expansion
        self.conv = nn.Sequential(
            ConvBlock(inp, hid),
            ConvBlock(hid, hid, 3, stride, 1, dw=True),
            ConvBlock(hid, oup, linear=True),
        )

    def forward(self, x):
        return x + self.conv(x) if self.connect else self.conv(x)


class MobileFaceNet(nn.Module):
    # (expansion, out_channels, num_blocks, stride)
    cfg = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]

    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        layers = []
        inp = 64
        for expansion, oup, n, stride in self.cfg:
            for i in range(n):
                layers.append(Bottleneck(inp, oup, stride if i == 0 else 1, expansion))
                inp = oup
        self.blocks = nn.Sequential(*layers)
        self.conv2 = ConvBlock(128, 512)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, embedding_dim, linear=True)

    def forward(self, x):
        x = self.dw_conv1(self.conv1(x))
        x = self.blocks(x)
        x = self.linear7(self.conv2(x))
        x = self.linear1(x)
        return x.flatten(1)


# ── Losses ────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss WITH the standard "easy margin"
    safeguard.

    Without it, cos(theta + margin) is only monotonically decreasing while
    theta + margin <= pi. Once theta + margin exceeds pi (routine early in
    training — random Xavier-initialized class weights across thousands of
    classes put most samples far from their target direction at step 0),
    cosine turns non-monotonic and the loss starts REWARDING misalignment
    for the hardest samples. That is a structural property of the naive
    formula, independent of random seed or any distillation teacher —
    training collapses onto the same degenerate fixed point regardless.
    The fix (used in every reference ArcFace implementation): fall back to
    a linear, monotonic approximation for samples past the threshold.
    """

    def __init__(self, num_classes, embedding_dim=EMBEDDING_DIM, scale=64.0, margin=0.5):
        super().__init__()
        self.scale, self.margin = scale, margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        # Precompute the easy-margin constants.
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, w).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(min=1e-7))
        target = cosine * self.cos_m - sine * self.sin_m  # cos(theta + margin), stable form
        # Easy margin: where theta+margin would exceed pi, use a linear
        # monotonic fallback instead of the (now decreasing) cosine.
        target = torch.where(cosine > self.threshold, target, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = cosine * (1 - one_hot) + target * one_hot
        return F.cross_entropy(logits * self.scale, labels)


def load_teacher():
    """InsightFace ArcFace teacher via ONNX Runtime (CPU or CUDA)."""
    import insightface

    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)
    rec = app.models["recognition"]

    @torch.no_grad()
    def embed(batch: torch.Tensor) -> torch.Tensor:
        # batch: (N,3,112,112) normalised to [-1,1]; teacher expects the same
        arr = batch.cpu().numpy()
        out = rec.session.run(None, {rec.session.get_inputs()[0].name: arr})[0]
        return torch.from_numpy(out).to(batch.device)

    return embed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kd-alpha", type=float, default=0.6)
    parser.add_argument("--no-teacher", action="store_true",
                        help="Skip distillation, train with ArcFace loss only")
    parser.add_argument("--max-identities", type=int, default=2000,
                        help="Subsample to this many identities so training fits a "
                             "12h Kaggle session (0 = use all)")
    parser.add_argument("--max-per-identity", type=int, default=40,
                        help="Cap images per identity (VGGFace2 averages ~360 — "
                             "uncapped, one epoch takes hours)")
    args = parser.parse_args()

    dirs = ensure_dirs()

    tfm = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    # Fast dataset build: select identities FIRST, then scan only their
    # folders (ImageFolder walks all 3.1M files — took 2.6h on Kaggle).
    # Cap images per identity so an epoch fits the session budget:
    # 2000 ids x 40 imgs = 80k imgs/epoch (~10 min/epoch on a T4).
    import os
    import random

    rng = random.Random(42)
    all_idents = sorted(e.name for e in os.scandir(args.data_dir) if e.is_dir())
    if args.max_identities and len(all_idents) > args.max_identities:
        idents = sorted(rng.sample(all_idents, args.max_identities))
    else:
        idents = all_idents

    samples: list[tuple[str, int]] = []
    for cls_idx, ident in enumerate(idents):
        d = os.path.join(args.data_dir, ident)
        files = [f.path for f in os.scandir(d)
                 if f.name.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(files) > args.max_per_identity:
            files = rng.sample(files, args.max_per_identity)
        samples += [(p, cls_idx) for p in files]
    print(f"Identities: {len(idents)}, images: {len(samples)} "
          f"(capped at {args.max_per_identity}/identity)")

    from PIL import Image
    from torch.utils.data import Dataset

    class FaceDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            path, label = samples[i]
            return tfm(Image.open(path).convert("RGB")), label

    ds = FaceDataset()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)
    num_identities = len(idents)

    student = MobileFaceNet().to(DEVICE)
    arcface = ArcFaceLoss(num_classes=num_identities).to(DEVICE)
    teacher_embed = None if args.no_teacher else load_teacher()

    base_lr = 0.1
    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(arcface.parameters()),
        lr=base_lr, momentum=0.9, weight_decay=5e-4,
    )
    ms = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.1)
    mse = nn.MSELoss()

    # Linear LR warmup over the first epoch: training an angular-margin
    # loss from a random init at full LR risks exactly the kind of early
    # instability that caused the collapse this script now guards against
    # structurally (see ArcFaceLoss) — warmup is a second, cheap safety net.
    warmup_steps = len(loader)
    global_step = 0

    best_path = dirs["checkpoints"] / "face_embedding_best.pth"
    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        for step, (imgs, labels) in enumerate(loader):
            if global_step < warmup_steps:
                warmup_lr = base_lr * (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = warmup_lr
            global_step += 1

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            s_emb = student(imgs)
            loss = arcface(s_emb, labels)
            if teacher_embed is not None:
                t_emb = teacher_embed(imgs)
                kd = mse(F.normalize(s_emb, p=2, dim=1), F.normalize(t_emb, p=2, dim=1))
                loss = args.kd_alpha * kd + (1 - args.kd_alpha) * loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(arcface.parameters()), max_norm=5.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(loader)} "
                      f"loss {loss.item():.4f}", flush=True)
        if global_step >= warmup_steps:
            scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs} | loss {epoch_loss / len(loader):.4f}", flush=True)
        torch.save(student.state_dict(), best_path)  # checkpoint every epoch

    (dirs["evaluation"] / "face_embedding_results.json").write_text(
        json.dumps({"epochs": args.epochs, "identities": num_identities,
                    "images": len(samples), "teacher_kd": not args.no_teacher},
                   indent=2)
    )
    print(f"Checkpoint: {best_path}")
    print("Evaluate on LFW with scripts/evaluate_lfw.py before publishing.")


if __name__ == "__main__":
    main()
