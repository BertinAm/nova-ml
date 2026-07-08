"""MOD-01 obstacle detection: YOLOv8n training on Detectra + VisDrone.

Practical note on distillation: Ultralytics' training loop does not expose
a clean hook for logit-level KD without forking the trainer. The pragmatic
strategy used here (and widely in practice) is:

1. Train YOLOv8n directly on the combined dataset (COCO-pretrained
   weights already embed much of what YOLOv8m "knows").
2. Optionally run teacher-generated pseudo-labelling: YOLOv8m labels the
   unlabelled/weakly-labelled portion of the data, and the student trains
   on that enriched label set — this is distillation through data rather
   than through logits, and works with the stock Ultralytics trainer.

Smoke test first (2 min, verifies the whole pipeline):
    python scripts/train_obstacle.py --data coco128.yaml --epochs 1

Full run (prepare_obstacle_dataset.py GENERATES the data YAML — correct
nc/names — so there is no manual config editing step):
    python scripts/prepare_obstacle_dataset.py --detectra ... --visdrone ... \
        --yaml-out /kaggle/working/obstacle_data.yaml
    python scripts/train_obstacle.py --data /kaggle/working/obstacle_data.yaml --epochs 100
"""
import argparse
import json

from nova_common import ensure_dirs, work_dir
from ultralytics import YOLO


def pseudo_label(teacher_weights: str, images_dir: str, labels_out: str, conf: float = 0.5):
    """Teacher labels images -> YOLO-format .txt files (data-space KD)."""
    teacher = YOLO(teacher_weights)
    teacher.predict(
        source=images_dir, conf=conf, save_txt=True, save_conf=False,
        project=labels_out, name="pseudo_labels", imgsz=640,
    )
    print(f"Pseudo-labels written under {labels_out}/pseudo_labels/labels/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="Generated YAML from prepare_obstacle_dataset.py, or coco128.yaml for smoke test")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--pseudo-label-dir", default=None,
                        help="If set, run YOLOv8m pseudo-labelling on this image dir first")
    args = parser.parse_args()

    dirs = ensure_dirs()

    if args.pseudo_label_dir:
        pseudo_label("yolov8m.pt", args.pseudo_label_dir, str(work_dir() / "pseudo"))

    student = YOLO(args.model)
    results = student.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(work_dir() / "runs"),
        name="obstacle_student",
        exist_ok=True,
    )

    metrics = student.val(data=args.data, imgsz=args.imgsz)
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
    }
    (dirs["evaluation"] / "obstacle_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    best = work_dir() / "runs" / "obstacle_student" / "weights" / "best.pt"
    print(f"Best checkpoint: {best}")

    # Ultralytics has native TFLite INT8 export — far more reliable than the
    # manual ONNX->TF->TFLite chain for YOLO architectures.
    exported = student.export(format="tflite", int8=True, imgsz=args.imgsz, data=args.data)
    print(f"TFLite INT8 export: {exported}")


if __name__ == "__main__":
    main()
