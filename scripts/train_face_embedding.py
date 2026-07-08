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
    def __init__(self, num_classes, embedding_dim=EMBEDDING_DIM, scale=64.0, margin=0.5):
        super().__init__()
        self.scale, self.margin = scale, margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, w)
        theta = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.margin)
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kd-alpha", type=float, default=0.6)
    parser.add_argument("--no-teacher", action="store_true",
                        help="Skip distillation, train with ArcFace loss only")
    parser.add_argument("--max-identities", type=int, default=2000,
                        help="Subsample to this many identities so training fits a "
                             "12h Kaggle session (0 = use all)")
    args = parser.parse_args()

    dirs = ensure_dirs()

    tfm = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    if args.max_identities and len(ds.classes) > args.max_identities:
        import random

        rng = random.Random(42)
        kept_classes = set(rng.sample(range(len(ds.classes)), args.max_identities))
        # Remap kept class indices to a dense 0..N-1 range for ArcFace.
        remap = {old: new for new, old in enumerate(sorted(kept_classes))}
        indices = [i for i, (_, c) in enumerate(ds.samples) if c in kept_classes]
        ds.samples = [(p, remap[c]) for p, c in (ds.samples[i] for i in indices)]
        ds.targets = [c for _, c in ds.samples]
        ds.classes = [ds.classes[i] for i in sorted(kept_classes)]
        ds.imgs = ds.samples
        print(f"Subsampled to {args.max_identities} identities, {len(ds.samples)} images")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    num_identities = len(ds.classes)
    print(f"Identities: {num_identities}, images: {len(ds)}")

    student = MobileFaceNet().to(DEVICE)
    arcface = ArcFaceLoss(num_classes=num_identities).to(DEVICE)
    teacher_embed = None if args.no_teacher else load_teacher()

    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(arcface.parameters()),
        lr=0.1, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.1)
    mse = nn.MSELoss()

    best_path = dirs["checkpoints"] / "face_embedding_best.pth"
    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            s_emb = student(imgs)
            loss = arcface(s_emb, labels)
            if teacher_embed is not None:
                t_emb = teacher_embed(imgs)
                kd = mse(F.normalize(s_emb, p=2, dim=1), F.normalize(t_emb, p=2, dim=1))
                loss = args.kd_alpha * kd + (1 - args.kd_alpha) * loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs} | loss {epoch_loss / len(loader):.4f}")
        torch.save(student.state_dict(), best_path)

    (dirs["evaluation"] / "face_embedding_results.json").write_text(
        json.dumps({"epochs": args.epochs, "identities": num_identities}, indent=2)
    )
    print(f"Checkpoint: {best_path}")
    print("Evaluate on LFW with scripts/evaluate_models.py before publishing.")


if __name__ == "__main__":
    main()
