"""PyTorch -> ONNX -> TFLite INT8 conversion + latency benchmark.

Used for the timm/custom models (MOD-04 currency, MOD-05 embedding).
MOD-01 (YOLOv8) uses Ultralytics' built-in `model.export(format='tflite',
int8=True)` instead — see train_obstacle.py.

    python scripts/convert_to_tflite.py \
        --checkpoint checkpoints/currency_student_best.pth \
        --arch mobilenetv3_small_100 --num-classes 5 \
        --input-size 224 --out exports/currency_detection_v1.tflite \
        --calib-dir datasets/cfa_currency/val
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def export_onnx(model, input_size: int, onnx_path: str, opset: int = 17):
    import onnx

    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=opset,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"ONNX exported: {onnx_path}")


def onnx_to_tflite_int8(onnx_path: str, tflite_path: str, representative_gen):
    """Convert via onnx2tf (maintained successor to onnx-tf) then quantize."""
    import subprocess

    import tensorflow as tf

    saved_dir = onnx_path.replace(".onnx", "_saved_model")
    subprocess.run(["onnx2tf", "-i", onnx_path, "-o", saved_dir, "-osd"], check=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tflite_path).write_bytes(tflite_model)
    print(f"TFLite INT8: {tflite_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")


def make_representative_dataset(calib_dir: str, input_size: int, n_samples: int = 200):
    """Yield calibration samples from a folder of images (NHWC float32)."""
    from PIL import Image

    paths = [p for p in Path(calib_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    paths = paths[:n_samples]
    if not paths:
        raise ValueError(f"No calibration images found in {calib_dir}")

    def generator():
        for p in paths:
            img = Image.open(p).convert("RGB").resize((input_size, input_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            yield [arr[np.newaxis, ...].astype(np.float32)]

    return generator


def benchmark_tflite(tflite_path: str, n_runs: int = 100):
    import time

    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    dummy = np.random.randint(-128, 127, inp["shape"], dtype=np.int8)
    for _ in range(5):
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
    times = []
    for _ in range(n_runs):
        interp.set_tensor(inp["index"], dummy)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    p50, p95 = np.percentile(times, 50), np.percentile(times, 95)
    print(f"{tflite_path}: median={p50:.1f}ms  P95={p95:.1f}ms  ~{1000 / p50:.1f} FPS (this machine)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True,
                        help="timm architecture name, or 'mobilefacenet' for the custom net")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib-dir", required=True, help="Folder of images for INT8 calibration")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.arch == "mobilefacenet":
        from train_face_embedding import MobileFaceNet

        model = MobileFaceNet()
    else:
        import timm

        model = timm.create_model(args.arch, pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    onnx_path = args.out.replace(".tflite", ".onnx")
    export_onnx(model, args.input_size, onnx_path)
    rep_gen = make_representative_dataset(args.calib_dir, args.input_size)
    onnx_to_tflite_int8(onnx_path, args.out, rep_gen)

    if args.benchmark:
        benchmark_tflite(args.out)


if __name__ == "__main__":
    main()
