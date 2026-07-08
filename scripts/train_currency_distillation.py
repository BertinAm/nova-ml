"""MOD-04 currency classification: two-phase distillation training.

Phase 1: fine-tune EfficientNet-B4 teacher on the CFA dataset.
Phase 2: distill into MobileNetV3-Small student (KD + CE loss).

Expects an ImageFolder dataset:
    <data-dir>/train/fcfa_500 ... fcfa_10000
    <data-dir>/val/...

On Kaggle, upload your collected CFA images as a private Kaggle dataset
and pass --data-dir /kaggle/input/<your-dataset-slug>.

    python scripts/train_currency_distillation.py --data-dir datasets/cfa_currency
"""
import argparse
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from nova_common import ensure_dirs
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEACHER_MODEL = "efficientnet_b4"
STUDENT_MODEL = "mobilenetv3_small_100"
NUM_CLASSES = 5
TEMPERATURE = 5.0
ALPHA = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")


def distillation_loss(student_logits, teacher_logits, labels):
    soft_t = F.softmax(teacher_logits / TEMPERATURE, dim=1)
    soft_s = F.log_softmax(student_logits / TEMPERATURE, dim=1)
    kd = kl_loss(soft_s, soft_t) * (TEMPERATURE**2)
    hard = ce_loss(student_logits, labels)
    return ALPHA * kd + (1 - ALPHA) * hard


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def finetune_teacher(teacher, train_loader, epochs):
    print(f"Phase 1: fine-tuning teacher ({TEACHER_MODEL}) for {epochs} epochs...")
    opt = torch.optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        teacher.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            ce_loss(teacher(imgs), labels).backward()
            opt.step()
        sch.step()
        print(f"  teacher epoch {epoch + 1}/{epochs}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False


def distill_student(teacher, student, train_loader, val_loader, epochs, lr, ckpt_dir, use_wandb):
    print(f"Phase 2: distilling into student ({STUDENT_MODEL}) for {epochs} epochs...")
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if use_wandb:
        import wandb

        wandb.init(project="nova-currency-distillation")
    best_acc = 0.0
    best_path = ckpt_dir / "currency_student_best.pth"
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(imgs)
            loss = distillation_loss(student(imgs), t_logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sch.step()
        acc = evaluate(student, val_loader)
        if use_wandb:
            import wandb

            wandb.log({"epoch": epoch, "val_acc": acc, "loss": total_loss})
        print(f"  epoch {epoch + 1}/{epochs} | loss {total_loss:.4f} | val_acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), best_path)
    if use_wandb:
        import wandb

        wandb.finish()
    print(f"Best val accuracy: {best_acc:.4f} -> {best_path}")
    return best_path, best_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    dirs = ensure_dirs()
    data_dir = Path(args.data_dir)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_transforms)
    print(f"Classes: {train_ds.classes}")
    assert len(train_ds.classes) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} class folders, found {train_ds.classes}"
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    teacher = timm.create_model(TEACHER_MODEL, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    student = timm.create_model(STUDENT_MODEL, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    finetune_teacher(teacher, train_loader, args.teacher_epochs)
    teacher_acc = evaluate(teacher, val_loader)
    print(f"Teacher val accuracy: {teacher_acc:.4f}")

    best_path, best_acc = distill_student(
        teacher, student, train_loader, val_loader,
        args.student_epochs, args.lr, dirs["checkpoints"], args.wandb,
    )

    results = {
        "teacher_val_acc": teacher_acc,
        "student_val_acc": best_acc,
        "classes": train_ds.classes,
    }
    (dirs["evaluation"] / "currency_results.json").write_text(json.dumps(results, indent=2))
    print(f"Done. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
