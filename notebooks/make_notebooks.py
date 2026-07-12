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
# Detectra-only (v1.0.0 lesson: VisDrone's aerial drone perspective is
# wrong for a chest-mounted camera — its tiny objects at 320px dragged
# mAP50 down to 10%. Pedestrian-height Detectra alone is the right data).
# GENERATES the training YAML with correct nc/names; aborts if 0 images.
!python scripts/prepare_obstacle_dataset.py \\
    --detectra {DETECTRA} \\
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
    --epochs 60 --imgsz 320 --batch 64 --workers 4 --patience 15"""),
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
    --eval-json /kaggle/working/evaluation/obstacle_results.json --version 1.1.0"""),
])

# ── 02: currency ──────────────────────────────────────────────────────
currency = nb([
    ("md", "# NOVA 02 — MOD-04 Currency Detection (MobileNetV3-Small)\n"
           "**10 classes**: notes 500/1000/2000/5000/10000 + coins 25/50/100/200/500 "
           "(incl. the new BEAC Type-2024 coin series).\n\n"
           "**Prerequisite:** upload `cfa_currency_kaggle.zip` (built by "
           "`scrape_cfa_images.py` + `augment_cfa_dataset.py` — official BEAC "
           "scans, augmented; 1830 train / 120 val / 120 test) as a **private "
           "Kaggle dataset**, attach it here, and set CFA_DATA below.\n\n"
           "The bootstrap dataset uses clean official scans + augmentation. "
           "Coin classes have leaky val/test (few sources) — treat coin test "
           "metrics as optimistic until team-collected photos are added."),
    ("code", BOOTSTRAP),
    ("code", "!pip install -q timm onnx2tf onnx"),
    ("code", """\
# Resolve the attached dataset mount (search 3 levels — Kaggle may nest
# under /kaggle/input/datasets/<owner>/<slug>)
import glob, os
inputs = (glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/*/*')
          + glob.glob('/kaggle/input/*/*/*'))
CFA_DATA = next(p for p in inputs
                if 'cfa' in p.split('/')[-1].lower() and os.path.isdir(p))
# If the zip extracted with a wrapper folder, descend into it
if not os.path.isdir(f'{CFA_DATA}/train'):
    CFA_DATA = next(p for p in glob.glob(f'{CFA_DATA}/*')
                    if os.path.isdir(f'{p}/train'))
print('CFA_DATA =', CFA_DATA)
!ls {CFA_DATA}/train"""),
    ("code", """\
# Reuse the already-trained checkpoint from HF if one exists (saves ~26 min
# when only the conversion/publish steps changed). Delete the pytorch/ file
# on the HF repo to force retraining.
import os, shutil
from huggingface_hub import hf_hub_download
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
try:
    p = hf_hub_download('unixio/nova-currency-detection',
                        'pytorch/currency_student_best.pth',
                        token=os.environ['HF_TOKEN'])
    shutil.copy(p, '/kaggle/working/checkpoints/currency_student_best.pth')
    SKIP_TRAINING = True
    print('Reusing trained checkpoint from HF — skipping training.')
except Exception as e:
    SKIP_TRAINING = False
    print('No checkpoint on HF — will train.', e)"""),
    ("code", """\
# Two-phase: fine-tune EfficientNet-B4 teacher, distill into MobileNetV3-Small.
# Class count is auto-detected from the train/ folders (10).
if not SKIP_TRAINING:
    !python scripts/train_currency_distillation.py --data-dir {CFA_DATA} \\
        --teacher-epochs 20 --student-epochs 60 --batch-size 64
else:
    print('Skipped — using HF checkpoint.')"""),
    ("code", """\
# Held-out test evaluation at the 0.85 confidence gate (FR-04-03)
!python scripts/evaluate_models.py currency \\
    --checkpoint /kaggle/working/checkpoints/currency_student_best.pth \\
    --data-dir {CFA_DATA}/test"""),
    ("code", """\
# Convert to TFLite INT8 (calibrate on val images) + benchmark
!python scripts/convert_to_tflite.py \\
    --checkpoint /kaggle/working/checkpoints/currency_student_best.pth \\
    --arch mobilenetv3_small_100 --num-classes 10 --input-size 224 \\
    --out /kaggle/working/exports/currency_detection_v1.tflite \\
    --calib-dir {CFA_DATA}/val --benchmark"""),
    ("code", """\
# Publish to HuggingFace: unixio/nova-currency-detection
!python scripts/push_to_huggingface.py --module MOD-04 \\
    --pytorch /kaggle/working/checkpoints/currency_student_best.pth \\
    --tflite /kaggle/working/exports/currency_detection_v1.tflite \\
    --labels labels/cfa_labels.txt \\
    --eval-json /kaggle/working/evaluation/currency_test_results.json --version 1.0.0"""),
    ("md", "### Confusion matrix (for the report)\n"
           "Evaluates the just-published checkpoint on the held-out test split "
           "and saves a real per-class confusion matrix — no retraining, ~1 min."),
    ("code", """\
!pip install -q scikit-learn matplotlib
!python scripts/evaluate_currency_confusion.py --data-dir {CFA_DATA}
from IPython.display import Image, display
display(Image('/kaggle/working/evaluation/currency_confusion_matrix.png'))"""),
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
# Auto-resolve mounts (Kaggle nests datasets 3 levels deep) and locate the
# identity-folder root: the directory whose children are identity folders.
import glob, os
inputs = (glob.glob('/kaggle/input/*') + glob.glob('/kaggle/input/*/*')
          + glob.glob('/kaggle/input/*/*/*'))
VGG_ROOT = next(p for p in inputs if 'vggface' in p.split('/')[-1].lower())
print('VGG_ROOT =', VGG_ROOT)
!find {VGG_ROOT} -maxdepth 2 -type d | head -10
# Pick the deepest dir that contains many subfolders (identities)
VGG_DATA = None
for cand in [VGG_ROOT] + glob.glob(f'{VGG_ROOT}/*') + glob.glob(f'{VGG_ROOT}/*/*'):
    if os.path.isdir(cand):
        subs = [s for s in os.listdir(cand)[:50]
                if os.path.isdir(os.path.join(cand, s))]
        if len(subs) >= 40:  # many identity folders
            VGG_DATA = cand
            break
assert VGG_DATA, 'Could not locate identity-folder root — inspect the layout above'
print('VGG_DATA =', VGG_DATA, '| identities (sample count):',
      len(os.listdir(VGG_DATA)))"""),
    ("code", """\
# FORCE RETRAIN: the checkpoint currently on HF (v1.0.0) is a known-bad
# collapsed model — LFW accuracy 0.500 (random). Do NOT reuse it. Once a
# genuinely good checkpoint is published, flip this back to True to save
# ~90 min on conversion-only re-runs.
SKIP_TRAINING = False
print('Forcing full retrain — HF checkpoint is the known-collapsed v1.0.0 model.')"""),
    ("code", """\
# MobileFaceNet + ArcFace loss. --max-identities 2000 keeps one run inside
# Kaggle's 12h session limit. --no-teacher: InsightFace teacher only runs
# on CPU here (onnxruntime shadows the GPU build), ~10x slower, AND two
# earlier runs proved the teacher wasn't even the cause of the training
# collapse we hit (see ArcFaceLoss docstring — it was a missing easy-margin
# safeguard, now fixed) — so there is no upside to paying for it.
if not SKIP_TRAINING:
    !python scripts/train_face_embedding.py --data-dir {VGG_DATA} \\
        --epochs 30 --batch-size 256 --max-identities 2000 \\
        --max-per-identity 40 --no-teacher
else:
    print('Skipped — using HF checkpoint.')"""),
    ("code", """\
# Build a SMALL calibration dir (200 images). Never point the converter at
# the full dataset — rglob over 3M files takes forever.
import os, random, shutil
CALIB = '/kaggle/working/calib'
shutil.rmtree(CALIB, ignore_errors=True)
os.makedirs(CALIB)
rng = random.Random(42)
idents = rng.sample(sorted(os.listdir(VGG_DATA)), 200)
for i, ident in enumerate(idents):
    d = os.path.join(VGG_DATA, ident)
    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png'))]
    if imgs:
        shutil.copy(os.path.join(d, imgs[0]), f'{CALIB}/{i:03d}_{imgs[0]}')
print(len(os.listdir(CALIB)), 'calibration images')"""),
    ("code", """\
# Convert: tries INT8 first, falls back to float32 (never ship onnx2tf's
# fp16 — true fp16 tensors fail to load on standard TFLite runtimes).
!python scripts/convert_to_tflite.py \\
    --checkpoint /kaggle/working/checkpoints/face_embedding_best.pth \\
    --arch mobilefacenet --input-size 112 \\
    --out /kaggle/working/exports/face_embedding_v1.tflite \\
    --calib-dir {CALIB} --benchmark"""),
    ("code", """\
# LFW verification (FR-05-05 acceptance metric: accuracy >= 99% target).
# Non-fatal: if the LFW mirror's CSV layout differs, publish proceeds and
# the eval can be added in a later 5-min checkpoint-reuse run.
LFW_ROOT = next((p for p in inputs if 'lfw' in p.split('/')[-1].lower()), None)
print('LFW_ROOT =', LFW_ROOT)
if LFW_ROOT:
    !python scripts/evaluate_lfw.py \\
        --checkpoint /kaggle/working/checkpoints/face_embedding_best.pth \\
        --lfw-root {LFW_ROOT} \\
        --out /kaggle/working/evaluation/lfw_results.json || echo 'LFW eval failed — continuing'
else:
    print('LFW dataset not attached — skipping eval')"""),
    ("code", """\
# Publish to HuggingFace: unixio/nova-face-embedding
!python scripts/push_to_huggingface.py --module MOD-05-embed \\
    --pytorch /kaggle/working/checkpoints/face_embedding_best.pth \\
    --tflite /kaggle/working/exports/face_embedding_v1.tflite \\
    --eval-json /kaggle/working/evaluation/lfw_results.json --version 1.0.0"""),
])

# ── 01b: obstacle COCO baseline (no training — quantize & publish) ────
obstacle_coco = nb([
    ("md", "# NOVA 01b — MOD-01 Obstacle Detection v1.2.0 (COCO baseline)\n"
           "**No training.** Takes stock COCO-pretrained YOLOv8n (80 real class "
           "names), quantizes to INT8 @ 320px, evaluates on COCO128, benchmarks, "
           "and publishes to `unixio/nova-obstacle-detection` as v1.2.0.\n\n"
           "Rationale: Detectra/VisDrone fine-tunes (v1.0.0/v1.1.0) reached only "
           "~10% mAP50 — kept on the Hub as ablations. The stock model satisfies "
           "SRS FR-01-03 ('80-class COCO dataset') with real TTS-speakable names.\n\n"
           "**~10 minutes total. No datasets to attach.** GPU + Internet + HF_TOKEN."),
    ("code", BOOTSTRAP),
    ("code", """\
# Pre-install the LiteRT converter stack so Ultralytics' AutoUpdate doesn't
# swap numpy's deps mid-kernel (that corrupts the running Python process).
!pip install -q ultralytics 'litert-torch>=0.9.0' 'ai-edge-litert>=2.1.4' ai-edge-quantizer"""),
    ("code", """\
# Evaluate stock YOLOv8n on COCO128 at 320px (deployment resolution).
# COCO128 is small — cite Ultralytics' official full-COCO numbers alongside
# (YOLOv8n: 37.3 mAP50-95 @ 640) in the report.
import json, os
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
metrics = model.val(data='coco128.yaml', imgsz=320)
results = {
    'mAP50_coco128@320': float(metrics.box.map50),
    'mAP50-95_coco128@320': float(metrics.box.map),
    'official_coco_mAP50-95@640': 0.373,
    'note': 'stock COCO-pretrained YOLOv8n, no fine-tune',
}
os.makedirs('/kaggle/working/evaluation', exist_ok=True)
json.dump(results, open('/kaggle/working/evaluation/obstacle_results.json', 'w'), indent=2)
print(json.dumps(results, indent=2))"""),
    ("code", """\
# Write the 80 COCO class names — the app's TTS reads these
os.makedirs('/kaggle/working/labels', exist_ok=True)
names = [model.names[i] for i in range(len(model.names))]
open('/kaggle/working/labels/coco_labels.txt', 'w').write('\\n'.join(names) + '\\n')
print(len(names), 'classes:', names[:8], '...')"""),
    ("code", """\
# INT8 quantization at 320 (COCO128 calibration), run in a FRESH subprocess
# via the yolo CLI — immune to any in-kernel package-state issues.
!yolo export model=yolov8n.pt format=litert quantize=int8 imgsz=320 data=coco128.yaml
import glob
candidates = glob.glob('**/yolov8n*int8*.tflite', recursive=True) + \\
             glob.glob('**/yolov8n*_int8.tflite', recursive=True)
print('candidates:', candidates)
assert candidates, 'INT8 TFLite not found after export'
exported = candidates[0]
size_mb = os.path.getsize(exported) / 1e6
print(f'{exported}: {size_mb:.2f} MB (budget: <15 MB)')
assert size_mb < 15"""),
    ("code", """\
# Publish v1.2.0
!python scripts/push_to_huggingface.py --module MOD-01 \\
    --tflite {exported} \\
    --labels /kaggle/working/labels/coco_labels.txt \\
    --eval-json /kaggle/working/evaluation/obstacle_results.json --version 1.2.0"""),
])

for name, notebook in [
    ("00_setup_check.ipynb", setup),
    ("01_obstacle_detection.ipynb", obstacle),
    ("01b_obstacle_coco_baseline.ipynb", obstacle_coco),
    ("02_currency_detection.ipynb", currency),
    ("03_face_embedding.ipynb", face),
]:
    (HERE / name).write_text(json.dumps(notebook, indent=1))
    print(f"wrote {name}")
