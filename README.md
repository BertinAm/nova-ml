# NOVA ML Pipeline

Model Training, Distillation, Quantization & HuggingFace Deployment

| Field | Value |
| --- | --- |
| **Training Framework** | PyTorch 2.3+ with HuggingFace Transformers / timm / Ultralytics |
| **Deployment Format** | TensorFlow Lite INT8 |
| **Compression** | Knowledge Distillation (Hinton et al., 2015) + Post-Training Quantization |
| **Model Registry** | HuggingFace Hub [huggingface.co/nova-assistive](https://www.google.com/search?q=https://huggingface.co/nova-assistive) |
| **Experiment Tracking** | Weights & Biases (wandb) |
| **License** | MIT |

---

## What This Repository Contains

This repository is the offline ML research and training workspace for NOVA. It does not run in production. Its job is to produce the quantized TFLite model files that are bundled into the *nova-mobile* app and served via the *nova-backend* model registry.

All trained models are published to the HuggingFace Hub organisation *nova-assistive*. After publishing, the SHA-256 checksum of each TFLite file is registered in the *nova-backend* model registry so the mobile app can discover and download updates over the air.

## Models

| Module | Task | Teacher | Student | HuggingFace Repo |
| --- | --- | --- | --- | --- |
| MOD-01 | Obstacle detection | YOLOv8m | YOLOv8n (320x320) | *nova-assistive/obstacle-detection* |
| MOD-04 | Currency classification | EfficientNet-B4 | MobileNetV3-Small (224x224) | *nova-assistive/currency-detection* |
| MOD-05 (s1) | Face detection | RetinaFace-ResNet50 | BlazeFace (128x128) | *nova-assistive/face-detection* |
| MOD-05 (s2) | Face embedding | ArcFace R100 | MobileFaceNet (112x112) | *nova-assistive/face-embedding* |

## Repository Structure

```text
nova-ml/
├── scripts/
│   ├── train_obstacle_distillation.py   # MOD-01 KD training
│   ├── kd_trainer.py                    # Custom YOLOv8 KD trainer
│   ├── train_currency_distillation.py   # MOD-04 KD training (2-phase)
│   ├── train_face_detection.py          # MOD-05 BlazeFace fine-tune
│   ├── train_face_embedding.py          # MOD-05 MobileFaceNet + ArcFace KD
│   ├── evaluate_models.py               # Evaluation metrics for all models
│   ├── convert_to_tflite.py             # PyTorch → ONNX → TFLite INT8
│   ├── benchmark_tflite.py              # Latency benchmark on TFLite models
│   ├── push_to_huggingface.py           # Publish all models to HF Hub
│   └── register_model_in_backend.py     # Register new version in nova-backend
├── configs/
│   ├── obstacle_data.yaml               # Dataset config
│   └── training_defaults.yaml           # Shared hyperparameter defaults
├── datasets/                            # Local dataset folders (gitignored)
├── checkpoints/                         # Saved model checkpoints (gitignored)
├── exports/                             # ONNX and TFLite outputs (gitignored)
└── requirements.txt

```

## Getting Started

### Prerequisites

* Python 3.11+
* CUDA-capable GPU strongly recommended for training
* A HuggingFace account with write access to the *nova-assistive* organisation
* A Weights & Biases account for experiment tracking

### 1. Install dependencies

```bash
git clone https://github.com/your-org/nova-ml.git
cd nova-ml
python -m venv nova_ml_env
source nova_ml_env/bin/activate
pip install -r requirements.txt

```

### 2. Configure credentials

```bash
huggingface-cli login
wandb login

```

### 3. Prepare datasets

* **COCO 2017 (obstacle detection):** Download and place in `datasets/obstacle_combined/`.
* **CFA currency:** Place in `datasets/cfa_currency/` with ImageFolder structure.
* **WIDER FACE:** Use `pip install datasets && python -c "from datasets import load_dataset; load_dataset('wider_face')"`.

### 4. Run the training pipeline

1. `python scripts/train_obstacle_distillation.py`
2. `python scripts/train_currency_distillation.py`
3. `python scripts/train_face_embedding.py`
4. `python scripts/evaluate_models.py`
5. `python scripts/convert_to_tflite.py`
6. `python scripts/benchmark_tflite.py`
7. `python scripts/push_to_huggingface.py`
8. `python scripts/register_model_in_backend.py`

## Pipeline Overview

| Step | Script | Input | Output |
| --- | --- | --- | --- |
| 1. Collect | Manual / LabelImg | Raw images | Annotated dataset |
| 2. Train | train_*.py | Dataset + teacher | Student checkpoint |
| 3. Eval | evaluate_models.py | Checkpoint + test data | Metrics (mAP, acc) |
| 4. Convert | convert_to_tflite.py | Student checkpoint | Quantized .tflite |
| 5. Benchmark | benchmark_tflite.py | .tflite file | Latency / FPS |
| 6. Publish | push_to_huggingface.py | Artifacts | HF Repo + SHA-256 |
| 7. Register | register_model_in_backend.py | SHA-256 | DB entry |
| 8. OTA | nova-mobile app | Registry entry | Model download |

## Branching Strategy

* **main**: Stable, merged after model passes evaluation thresholds.
* **dev**: Integration branch.
* **feature/***: Specific module training experiments (e.g., *feature/obstacle-distillation*).

## Important Notes

* `datasets/`, `checkpoints/`, and `exports/` are gitignored to prevent committing large binary files.
* Training from scratch on a single NVIDIA T4: obstacle (~8h), currency (~3h), face embedding (~12h).
* `register_model_in_backend.py` requires an admin JWT—never hardcode it.

## Related Repositories

* [nova-backend](https://www.google.com/search?q=https://github.com/your-org/nova-backend) FastAPI server and model registry
* [nova-mobile](https://www.google.com/search?q=https://github.com/your-org/nova-mobile) Flutter Android application

---

**Licence**
MIT see LICENSE file.

University of Buea, Faculty of Engineering and Technology, Department of Computer Engineering. Internet of Things and Video Processing Academic Year 2025/2026.
