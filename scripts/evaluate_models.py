"""Evaluation utilities for NOVA student models.

- Currency: top-1 accuracy overall and at the 0.85 confidence gate the
  mobile app uses (FR-04-03) plus coverage at that gate.
- Face embedding: LFW verification accuracy + TAR@FAR=1e-3.

    python scripts/evaluate_models.py currency \
        --checkpoint checkpoints/currency_student_best.pth \
        --data-dir datasets/cfa_currency/test
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from nova_common import ensure_dirs
from sklearn.metrics import accuracy_score, classification_report, roc_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_currency(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    for imgs, labels in test_loader:
        probs = torch.softmax(model(imgs.to(DEVICE)), dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confs.extend(confs.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
    labels_arr, preds_arr, confs_arr = map(np.array, (all_labels, all_preds, all_confs))
    mask = confs_arr >= 0.85
    gated_acc = accuracy_score(labels_arr[mask], preds_arr[mask]) if mask.any() else 0.0
    results = {
        "overall_accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "accuracy_at_conf_0.85": float(gated_acc),
        "coverage_at_conf_0.85": float(mask.mean()),
    }
    print(json.dumps(results, indent=2))
    return results


@torch.no_grad()
def evaluate_face_verification(model, pair_loader, threshold=0.75):
    model.eval()
    similarities, is_same = [], []
    for img1, img2, same in pair_loader:
        e1 = F.normalize(model(img1.to(DEVICE)), p=2, dim=1)
        e2 = F.normalize(model(img2.to(DEVICE)), p=2, dim=1)
        similarities.extend((e1 * e2).sum(dim=1).cpu().numpy())
        is_same.extend(same.numpy())
    sims, gt = np.array(similarities), np.array(is_same)
    acc = accuracy_score(gt, sims >= threshold)
    fpr, tpr, _ = roc_curve(gt, sims)
    tar = float(tpr[np.searchsorted(fpr, 1e-3)])
    results = {"lfw_accuracy": float(acc), "tar_at_far_1e-3": tar, "threshold": threshold}
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    p_cur = sub.add_parser("currency")
    p_cur.add_argument("--checkpoint", required=True)
    p_cur.add_argument("--data-dir", required=True)
    p_cur.add_argument("--arch", default="mobilenetv3_small_100")

    args = parser.parse_args()
    dirs = ensure_dirs()

    if args.task == "currency":
        import timm
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ds = datasets.ImageFolder(args.data_dir, transform=tfm)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        model = timm.create_model(args.arch, pretrained=False, num_classes=len(ds.classes)).to(DEVICE)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        results = evaluate_currency(model, loader, ds.classes)
        (dirs["evaluation"] / "currency_test_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
