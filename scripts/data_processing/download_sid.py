"""
download_SID.py
"""

import argparse
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset


LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}


def download_and_sample(out_dir: Path, sample_size: int, split: str):
    print(f"Loading SID_Set split='{split}' in streaming mode (dataset is ~140GB total)...")
    ds = load_dataset("saberzl/SID_Set", split=split, streaming=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    label_counts = Counter()
    mask_counts = Counter()

    print(f"Saving a sample of {sample_size} rows to {out_dir} ...")
    for i, row in enumerate(ds):
        if i >= sample_size:
            break

        label = row["label"]
        label_name = LABEL_NAMES.get(label, f"unknown_{label}")
        label_counts[label_name] += 1

        class_dir = out_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        img = row["image"]
        img_id = row.get("img_id", f"row_{i}")
        img_path = class_dir / f"{img_id}.jpg"
        img.save(img_path)

        mask = row.get("mask")
        if mask is not None:
            mask_dir = out_dir / f"{label_name}_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask.save(mask_dir / f"{img_id}_mask.jpg")
            mask_counts[label_name] += 1

    print(f"\nSaved {sum(label_counts.values())} sample images.")
    print("Label distribution in this sample:", dict(label_counts))
    print("Rows with a mask, by label:", dict(mask_counts))
    return label_counts, mask_counts


def inspect_structure(root: Path):
    """Walk the saved sample and print a structure report."""
    print(f"\n{'='*60}")
    print(f"STRUCTURE REPORT: {root}")
    print(f"{'='*60}\n")

    if not root.exists():
        print(f"Nothing found at {root} — did the download step run?")
        return

    print("Top-level folders found:")
    top_level = sorted([p.name for p in root.iterdir() if p.is_dir()])
    for name in top_level:
        n_files = len(list((root / name).glob("*")))
        print(f"  {name}/  ({n_files} files)")

    print("\nFile extensions found:")
    ext_counter = Counter()
    for f in root.rglob("*"):
        if f.is_file():
            ext_counter[f.suffix.lower()] += 1
    for ext, count in sorted(ext_counter.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no extension)'}: {count}")

    print("\nSample file paths (first 10):")
    for f in list(root.rglob("*.jpg"))[:10]:
        print(f"  {f.relative_to(root)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation"],
                     help="Which SID_Set split to sample from")
    ap.add_argument("--sample-size", type=int, default=200,
                     help="Number of rows to download and save for inspection")
    ap.add_argument("--out", default="./data/sid_set_sample",
                     help="Where to save the sample images")
    args = ap.parse_args()

    out_dir = Path(args.out)
    download_and_sample(out_dir, args.sample_size, args.split)
    inspect_structure(out_dir)


if __name__ == "__main__":
    main()
