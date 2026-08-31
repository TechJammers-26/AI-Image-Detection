"""
Usage:
    python predict.py \
        --input_dir /path/to/images \
        --output predictions.json \
        --temperature 1.42     # from CALIBRATION["continuous"]["T"] in evaluations.py CELL 39

    # or to override the checkpoint explicitly:
    python predict.py \
        --input_dir /path/to/images \
        --checkpoint efficientnet_b0_spec_best.pth \
        --output predictions.json \
        --temperature 1.83     # from CALIBRATION["spec"]["T"]

Output:
[
  {"image_path": "...", "pred": 0.0-1.0, "label": 0 or 1},
  ...
]
label is 1 (AI-generated) if pred >= --threshold, else 0 (real).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

DEFAULT_CHECKPOINT = "efficientnet_b0_continuous_best.pth"


def find_images(input_dir):
    paths = []
    for current_dir, _, filenames in os.walk(input_dir):
        for filename in sorted(filenames):
            if filename.lower().endswith(VALID_EXT):
                paths.append(os.path.join(current_dir, filename))
    return sorted(paths)


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def load_checkpoint_model(checkpoint_path, device):
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(filepath, canonicalize_fn):
    image = Image.open(filepath).convert("RGB")
    image = canonicalize_fn(image, max_side=512, jpeg_quality=95)
    image = model_transform(image)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", default="predictions.json")
    #temperature -> based on calculated temperature in evaluations.py
    parser.add_argument(
        "--temperature", type=float, default=1.0,
    )
    parser.add_argument("--batch_size", type=int, default=64)
    #threshold -> default 0.5 but can change to inc/dec FP/FN
    parser.add_argument(
        "--threshold", type=float, default=0.5,
    )
    #to find augmentation.py
    parser.add_argument(
        "--project_root", default=None,
    )
    args = parser.parse_args()

    if args.project_root:
        sys.path.append(args.project_root)

    try:
        from augmentations import canonicalize
    except ImportError:
        print(
            "Could not import canonicalize from augmentations.py. "
            "Run this script from the project folder, or pass --project_root.",
            file=sys.stderr,
        )
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] using {device}", file=sys.stderr)

    t0 = time.time()
    image_paths = find_images(args.input_dir)
    print(f"[find_images] {len(image_paths)} images found in {time.time() - t0:.2f}s", file=sys.stderr)
    if not image_paths:
        print(f"No images found under {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    model = load_checkpoint_model(args.checkpoint, device)
    print(f"[load_checkpoint_model] {time.time() - t0:.2f}s", file=sys.stderr)

    results = []
    batch_imgs, batch_paths = [], []

    # running totals for the bottleneck breakdown
    total_preprocess_s = 0.0
    total_inference_s = 0.0
    n_done = 0
    n_skipped = 0
    run_start = time.time()

    def flush():
        nonlocal total_inference_s
        if not batch_imgs:
            return
        t = time.time()
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            logits = model(batch).squeeze(1)
            probs = torch.sigmoid(logits / args.temperature)
        for p, path in zip(probs.cpu().tolist(), batch_paths):
            results.append({
                "image_path": path,
                "pred": float(p),
                "label": int(p >= args.threshold),
            })
        batch_n = len(batch_imgs)
        batch_imgs.clear()
        batch_paths.clear()
        dt = time.time() - t
        total_inference_s += dt
        elapsed = time.time() - run_start
        print(
            f"[batch] {batch_n} imgs inference in {dt:.2f}s "
            f"| done={n_done} skipped={n_skipped} "
            f"| elapsed={elapsed:.1f}s "
            f"| preprocess_total={total_preprocess_s:.1f}s inference_total={total_inference_s:.1f}s",
            file=sys.stderr,
        )

    for i, path in enumerate(image_paths):
        t = time.time()
        try:
            tensor = preprocess(path, canonicalize)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            n_skipped += 1
            continue
        total_preprocess_s += time.time() - t
        batch_imgs.append(tensor)
        batch_paths.append(path)
        n_done += 1
        if len(batch_imgs) == args.batch_size:
            flush()
        elif (i + 1) % 50 == 0:
            elapsed = time.time() - run_start
            rate = n_done / elapsed if elapsed > 0 else 0
            print(
                f"[progress] {i + 1}/{len(image_paths)} scanned "
                f"| elapsed={elapsed:.1f}s | ~{rate:.1f} img/s",
                file=sys.stderr,
            )
    flush()

    total_elapsed = time.time() - run_start
    print(
        f"[summary] total={total_elapsed:.1f}s "
        f"preprocess={total_preprocess_s:.1f}s ({100*total_preprocess_s/total_elapsed:.0f}%) "
        f"inference={total_inference_s:.1f}s ({100*total_inference_s/total_elapsed:.0f}%) "
        f"other={total_elapsed - total_preprocess_s - total_inference_s:.1f}s",
        file=sys.stderr,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()