"""Register a published model in the nova-backend model registry.

The backend's /models/register endpoint accepts a multipart upload of the
actual .tflite file (it recomputes the SHA-256 server-side and stores the
file for OTA download). Requires an operator account (User.is_operator).

    export NOVA_BACKEND_URL=http://localhost:8000
    export NOVA_OPERATOR_EMAIL=ops@example.com
    export NOVA_OPERATOR_PASSWORD=...
    python scripts/register_model_in_backend.py \
        --module MOD-04 --version 1.0.0 \
        --tflite exports/currency_detection_v1.tflite \
        --notes "MobileNetV3-S distilled from EfficientNet-B4"
"""
import argparse
import os
import sys

import requests
from nova_common import HF_REPOS

BACKEND_URL = os.environ.get("NOVA_BACKEND_URL", "http://localhost:8000")


def login() -> str:
    email = os.environ.get("NOVA_OPERATOR_EMAIL")
    password = os.environ.get("NOVA_OPERATOR_PASSWORD")
    if not email or not password:
        sys.exit("Set NOVA_OPERATOR_EMAIL and NOVA_OPERATOR_PASSWORD env vars (never hardcode).")
    r = requests.post(
        f"{BACKEND_URL}/auth/login",
        data={"username": email, "password": password},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def register_model(module_id: str, version: str, tflite_path: str, notes: str, token: str):
    hf_repo_url = f"https://huggingface.co/{HF_REPOS[module_id]}"
    with open(tflite_path, "rb") as f:
        r = requests.post(
            f"{BACKEND_URL}/models/register",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "module_id": module_id,
                "version": version,
                "hf_repo_url": hf_repo_url,
                "notes": notes,
                "activate": "true",
            },
            files={"file": (os.path.basename(tflite_path), f, "application/octet-stream")},
            timeout=120,
        )
    if r.status_code == 201:
        body = r.json()
        print(f"Registered {module_id} v{version} — checksum {body['checksum']}")
    else:
        sys.exit(f"FAILED: {r.status_code} {r.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, choices=list(HF_REPOS))
    parser.add_argument("--version", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    token = login()
    register_model(args.module, args.version, args.tflite, args.notes, token)
