"""Generates the Kaggle notebooks in this folder. Run once after editing:
    python notebooks/make_notebooks.py
Kept as the source of truth so notebook diffs stay reviewable.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

BOOTSTRAP = """\
# ── NOVA bootstrap: clone repo + resolve HF token from Kaggle secret ──
# Before running: Add-ons > Secrets > attach a secret named HF_TOKEN
# (your HuggingFace write token). Settings > Accelerator > GPU T4/P100.
import os, sys, subprocess

REPO = 'https://github.com/BertinAm/nova-ml.git'
if not os.path.exists('/kaggle/working/nova-ml'):
    subprocess.run(['git', 'clone', REPO, '/kaggle/working/nova-ml'], check=True)
else:  # already cloned in this session — pull latest fixes
    subprocess.run(['git', '-C', '/kaggle/working/nova-ml', 'pull'], check=True)
os.chdir('/kaggle/working/nova-ml')
sys.path.insert(0, '/kaggle/working/nova-ml/scripts')

from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
os.environ['NOVA_HF_USERNAME'] = 'unixio'

# GPU compatibility guard: Kaggle's PyTorch 2.10 image dropped sm_60 (P100).
# If you get a P100, switch Settings > Accelerator to 'GPU T4 x2'.
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    assert cap >= (7, 0), (
        f'{name} (sm_{cap[0]}{cap[1]}) is NOT supported by this PyTorch build. '
        'Switch Settings > Accelerator to GPU T4 x2 and restart.')
    print(f'GPU OK: {name}')
else:
    raise RuntimeError('No GPU — set Settings > Accelerator to GPU T4 x2.')
print('Bootstrap OK — repo cloned, HF token loaded.')"""


def nb(cells):
    return {
        "cells": [
            {
                "cell_type": "code" if kind == "code" else "markdown",
                "metadata": {},
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
                "source": src.splitlines(keepends=True),
            }
            for kind, src in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── 00: setup check ───────────────────────────────────────────────────
setup = nb([
    ("md", "# NOVA 00 — Setup Check\n"
           "Verifies GPU, HF token (secret `HF_TOKEN`), and repo access in ~1 minute.\n"
           "Run this first, once."),
    ("code", BOOTSTRAP),
    ("code", """\
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))"""),
    ("code", """\
# Verify HF token works and repos can be created under unixio
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
me = api.whoami()
print('Logged in as:', me['name'])
assert me['name'] == 'unixio', f"Expected unixio, got {me['name']}"
"""),
    ("code", """\
# Create the four model repos (idempotent)
from huggingface_hub import create_repo
for repo in ['nova-obstacle-detection', 'nova-currency-detection',
             'nova-face-detection', 'nova-face-embedding']:
    create_repo(f'unixio/{repo}', repo_type='model', exist_ok=True,
                token=os.environ['HF_TOKEN'])
    print(f'ready: https://huggingface.co/unixio/{repo}')"""),
    ("code", """\
# Smoke-test the obstacle pipeline end-to-end on COCO128 (~3 min on T4).
# Proves: ultralytics install, GPU training, TFLite INT8 export.
!pip install -q ultralytics onnx2tf onnx
!python scripts/train_obstacle.py --data coco128.yaml --epochs 1 --imgsz 320 --batch 16"""),
])

# ── 01: obstacle ──────────────────────────────────────────────────────
obstacle = nb([
    ("md", "# NOVA 01 — MOD-01 Obstacle Detection (YOLOv8n)\n"
           "**Attach datasets** (Add Data, right sidebar):\n"
           "- `jhontroya/dectectra-dataset`\n"
           "- `kushagrapandya/visdrone-dataset`\n\n"
           "**Accelerator:** GPU T4 x2 or P100. Full training ~6-9h — fits one "
           "Kaggle session (12h limit). Enable *Persistence: Files* to survive restarts."),
    ("code", BOOTSTRAP),
    ("code", "!pip install -q ultralytics onnx2tf onnx"),
    ("code", """\
# Resolve the ACTUAL mount paths — Kaggle sometimes mounts datasets
# nested as /kaggle/input/datasets/<owner>/<slug>, so search 3 levels.
import glob
inputs = (glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/*/*')
          + glob.glob('/kaggle/input/*/*/*'))
DETECTRA = next(p for p in inputs if 'dect' in p.split('/')[-1].lower())
VISDRONE = next((p for p in inputs if 'visdrone' in p.split('/')[-1].lower()), None)
print('Detectra:', DETECTRA)
print('VisDrone:', VISDRONE)
!find {DETECTRA} -maxdepth 3 -type d | head -20"""),
    ("code", """\
# Free disk first: /kaggle/working is only 19.5 GB and stale copies from
# earlier runs fill it. Images are SYMLINKED (not copied) into the merge,
# so the combined dataset itself costs ~100 MB of labels only.
!rm -rf /kaggle/working/datasets
!df -h /kaggle/working | tail -1
# Merges both datasets (remapping VisDrone class indices into the merged
# class list) and GENERATES the training YAML with correct nc/names.
# Aborts with a clear error if 0 images are found.
!python scripts/prepare_obstacle_dataset.py \\
    --detectra {DETECTRA} \\
    --visdrone {VISDRONE} \\
    --out /kaggle/working/datasets/obstacle_combined \\
    --yaml-out /kaggle/working/obstacle_data.yaml
!head -40 /kaggle/working/obstacle_data.yaml"""),
    ("code", """\
# Full training + TFLite INT8 export (Ultralytics native export).
# STOP if the previous cell errored — do not burn GPU hours on 0 images.
# Fast profile: 40 epochs + early stopping (patience 10) + batch 64 +
# 4 dataloader workers ≈ 2-3h on a T4 and typically lands within ~1-2
# mAP points of a 100-epoch run (YOLOv8n starts COCO-pretrained).
# Add --fraction 0.5 to halve it again if you're in a real hurry.
# If a session dies mid-run, re-run this cell with --resume.
!python scripts/train_obstacle.py --data /kaggle/working/obstacle_data.yaml \\
    --epochs 40 --imgsz 320 --batch 64 --workers 4 --patience 10"""),
    ("code", """\
# Publish to HuggingFace: unixio/nova-obstacle-detection
import glob
candidates = glob.glob('/kaggle/working/runs/obstacle_student/weights/**/*_int8.tflite',
                       recursive=True)
print('TFLite candidates:', candidates)
assert candidates, 'No INT8 TFLite found — training/export must succeed first.'
tflite_path = candidates[0]
best_pt = '/kaggle/working/runs/obstacle_student/weights/best.pt'
!python scripts/push_to_huggingface.py --module MOD-01 \\
    --pytorch {best_pt} --tflite {tflite_path} \\
    --eval-json /kaggle/working/evaluation/obstacle_results.json --version 1.0.0"""),
])

# ── 02: currency ──────────────────────────────────────────────────────
currency = nb([
    ("md", "# NOVA 02 — MOD-04 Currency Detection (MobileNetV3-Small)\n"
           "**Prerequisite:** upload your team-collected CFA images as a **private "
           "Kaggle dataset** with the ImageFolder layout "
           "(`train|val|test / fcfa_500 ... fcfa_10000`), then attach it here.\n\n"
           "No public CFA dataset exists — see section 4.3 of the training guide "
           "for the collection protocol (300+ images per denomination per side)."),
    ("code", BOOTSTRAP),
    ("code", "!pip install -q timm onnx2tf onnx"),
    ("code", """\
# Point at your attached dataset — EDIT the slug to match yours
CFA_DATA = '/kaggle/input/cfa-currency-dataset'   # <-- your dataset slug
!ls {CFA_DATA}/train"""),
    ("code", """\
# Two-phase: fine-tune EfficientNet-B4 teacher, distill into MobileNetV3-Small
!python scripts/train_currency_distillation.py --data-dir {CFA_DATA} \\
    --teacher-epochs 30 --student-epochs 80 --batch-size 32"""),
    ("code", """\
# Held-out test evaluation at the 0.85 confidence gate (FR-04-03)
!python scripts/evaluate_models.py currency \\
    --checkpoint /kaggle/working/checkpoints/currency_student_best.pth \\
    --data-dir {CFA_DATA}/test"""),
    ("code", """\
# Convert to TFLite INT8 (calibrate on val images) + benchmark
!python scripts/convert_to_tflite.py \\
    --checkpoint /kaggle/working/checkpoints/currency_student_best.pth \\
    --arch mobilenetv3_small_100 --num-classes 5 --input-size 224 \\
    --out /kaggle/working/exports/currency_detection_v1.tflite \\
    --calib-dir {CFA_DATA}/val --benchmark"""),
    ("code", """\
# Publish to HuggingFace: unixio/nova-currency-detection
!python scripts/push_to_huggingface.py --module MOD-04 \\
    --pytorch /kaggle/working/checkpoints/currency_student_best.pth \\
    --tflite /kaggle/working/exports/currency_detection_v1.tflite \\
    --labels labels/cfa_labels.txt \\
    --eval-json /kaggle/working/evaluation/currency_test_results.json --version 1.0.0"""),
])

# ── 03: face embedding ────────────────────────────────────────────────
face = nb([
    ("md", "# NOVA 03 — MOD-05 Face Embedding (MobileFaceNet)\n"
           "**Attach datasets:**\n"
           "- `yakhyokhuja/vggface2-112x112` — VGGFace2 pre-aligned to 112x112 "
           "(exactly MobileFaceNet's input size; no alignment preprocessing needed)\n"
           "- `jessicali9530/lfw-dataset` (evaluation only)\n\n"
           "**Accelerator:** GPU. Training on a subset of identities ~6-10h.\n"
           "Teacher (InsightFace ArcFace) downloads its weights on first run — "
           "internet must be ON in notebook settings."),
    ("code", BOOTSTRAP),
    ("code", "!pip install -q insightface onnxruntime-gpu onnx2tf onnx"),
    ("code", """\
# Locate the identity-folder root inside the attached dataset
# (layout may nest one level — adjust after inspecting)
!find /kaggle/input/vggface2-112x112 -maxdepth 2 -type d | head -10
VGG_DATA = '/kaggle/input/vggface2-112x112/train'   # one folder per identity
!ls {VGG_DATA} | wc -l"""),
    ("code", """\
# MobileFaceNet + ArcFace loss + embedding-KD from InsightFace teacher.
# --max-identities 2000 keeps one run inside Kaggle's 12h session limit.
# Use --no-teacher to skip distillation if insightface weights fail to load.
!python scripts/train_face_embedding.py --data-dir {VGG_DATA} \\
    --epochs 50 --batch-size 128 --max-identities 2000"""),
    ("code", """\
# Convert to TFLite INT8 (calibrate on a slice of training identities)
!python scripts/convert_to_tflite.py \\
    --checkpoint /kaggle/working/checkpoints/face_embedding_best.pth \\
    --arch mobilefacenet --input-size 112 \\
    --out /kaggle/working/exports/face_embedding_v1.tflite \\
    --calib-dir {VGG_DATA} --benchmark"""),
    ("code", """\
# Publish to HuggingFace: unixio/nova-face-embedding
!python scripts/push_to_huggingface.py --module MOD-05-embed \\
    --pytorch /kaggle/working/checkpoints/face_embedding_best.pth \\
    --tflite /kaggle/working/exports/face_embedding_v1.tflite \\
    --version 1.0.0"""),
])

for name, notebook in [
    ("00_setup_check.ipynb", setup),
    ("01_obstacle_detection.ipynb", obstacle),
    ("02_currency_detection.ipynb", currency),
    ("03_face_embedding.ipynb", face),
]:
    (HERE / name).write_text(json.dumps(notebook, indent=1))
    print(f"wrote {name}")
