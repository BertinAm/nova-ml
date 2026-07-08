"""Publish a trained model (PyTorch checkpoint + TFLite INT8) to the
HuggingFace Hub under the personal account (default: unixio).

Usage (after training + conversion):

    python scripts/push_to_huggingface.py \
        --module MOD-04 \
        --pytorch checkpoints/currency_student_best.pth \
        --tflite exports/currency_detection_v1.tflite \
        --labels labels/cfa_labels.txt \
        --version 1.0.0

Prints the TFLite SHA-256 checksum at the end — that value goes into the
nova-backend model registry (scripts/register_model_in_backend.py).
"""
import argparse
import datetime
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from nova_common import HF_REPOS, get_hf_token, sha256_of

# Static per-module config merged into config.json on the Hub.
MODULE_CONFIGS: dict[str, dict] = {
    "MOD-01": {
        "module_id": "MOD-01",
        "architecture": "YOLOv8n",
        "input_size": [320, 320],
        "distilled_from": "YOLOv8m",
        "datasets": ["detectra", "visdrone"],
        "near_threshold_m": 1.5,
        "warning_threshold_m": 3.0,
        "confidence_threshold": 0.55,
        "suppression_seconds": 2.0,
    },
    "MOD-04": {
        "module_id": "MOD-04",
        "architecture": "MobileNetV3-Small",
        "input_size": [224, 224],
        "num_classes": 5,
        "class_names": ["fcfa_500", "fcfa_1000", "fcfa_2000", "fcfa_5000", "fcfa_10000"],
        "spoken_labels": {
            "fcfa_500": "Five hundred francs CFA",
            "fcfa_1000": "One thousand francs CFA",
            "fcfa_2000": "Two thousand francs CFA",
            "fcfa_5000": "Five thousand francs CFA",
            "fcfa_10000": "Ten thousand francs CFA",
        },
        "distilled_from": "EfficientNet-B4",
        "confidence_threshold": 0.85,
    },
    "MOD-05-detect": {
        "module_id": "MOD-05-detect",
        "architecture": "BlazeFace",
        "input_size": [128, 128],
        "distilled_from": "RetinaFace-ResNet50",
    },
    "MOD-05-embed": {
        "module_id": "MOD-05-embed",
        "architecture": "MobileFaceNet",
        "input_size": [112, 112],
        "embedding_dim": 512,
        "distilled_from": "ArcFace-R100",
        "match_threshold": 0.75,
        "distance_metric": "cosine",
        "training_data": "VGGFace2-subset",
    },
}


def publish_model(
    module_id: str,
    pytorch_ckpt: str | None,
    tflite_model: str,
    labels_file: str | None,
    eval_results: dict,
    version: str,
    token: str | None = None,
) -> str:
    token = get_hf_token(token)
    api = HfApi(token=token)
    repo_id = HF_REPOS[module_id]
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)

    checksum = sha256_of(tflite_model)
    config = dict(MODULE_CONFIGS[module_id])
    config.update(
        {
            "tflite_checksum": checksum,
            "version": version,
            "published_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "download_url_hint": f"tflite/{Path(tflite_model).name}",
        }
    )

    uploads = [(tflite_model, f"tflite/{Path(tflite_model).name}")]
    if pytorch_ckpt:
        uploads.append((pytorch_ckpt, f"pytorch/{Path(pytorch_ckpt).name}"))
    if labels_file:
        uploads.append((labels_file, f"labels/{Path(labels_file).name}"))

    for src, dest in uploads:
        api.upload_file(
            path_or_fileobj=src, path_in_repo=dest, repo_id=repo_id,
            commit_message=f"v{version}: {dest}",
        )

    api.upload_file(
        path_or_fileobj=json.dumps(config, indent=2).encode(),
        path_in_repo="config.json", repo_id=repo_id,
        commit_message=f"config v{version}",
    )
    api.upload_file(
        path_or_fileobj=json.dumps(eval_results, indent=2).encode(),
        path_in_repo="evaluation/results.json", repo_id=repo_id,
        commit_message=f"eval v{version}",
    )

    print(f"Published: https://huggingface.co/{repo_id}")
    print(f"TFLite SHA-256: {checksum}")
    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, choices=list(HF_REPOS))
    parser.add_argument("--pytorch", default=None, help="PyTorch checkpoint path")
    parser.add_argument("--tflite", required=True, help="TFLite INT8 model path")
    parser.add_argument("--labels", default=None, help="Label file path")
    parser.add_argument("--eval-json", default=None, help="Path to evaluation results JSON")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    eval_results = {}
    if args.eval_json and Path(args.eval_json).exists():
        eval_results = json.loads(Path(args.eval_json).read_text())

    publish_model(args.module, args.pytorch, args.tflite, args.labels, eval_results, args.version)
