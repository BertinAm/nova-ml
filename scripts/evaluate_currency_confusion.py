"""MOD-04 confusion matrix: evaluate the published student checkpoint on
the held-out test split and save a confusion matrix figure + per-class
report.

Needs the CFA test split (<data-dir>/test/fcfa_500 ... fcfa_10000) and the
published checkpoint (downloaded from HF, no retraining). Run on Kaggle
where the dataset is mounted:

    python scripts/evaluate_currency_confusion.py --data-dir /kaggle/input/<cfa-dataset-slug>
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from nova_common import HF_REPOS, ensure_dirs
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

STUDENT_MODEL = "mobilenetv3_small_100"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    dirs = ensure_dirs()
    test_ds = datasets.ImageFolder(f"{args.data_dir}/test", transform=test_transforms)
    classes = test_ds.classes
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    ckpt_path = hf_hub_download(HF_REPOS["MOD-04"], "pytorch/currency_student_best.pth")
    model = timm.create_model(STUDENT_MODEL, pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print(report)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MOD-04 Currency Detection — Confusion Matrix (test split)")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = dirs["evaluation"] / "currency_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    (dirs["evaluation"] / "currency_classification_report.txt").write_text(report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
