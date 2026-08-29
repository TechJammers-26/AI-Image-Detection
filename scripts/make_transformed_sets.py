#!/usr/bin/env python3
"""
make_transformed_sets.py wraps the transformations implemented by augmentation.py
and ouputs the transformed image sets

Materialise transformed copies of an image tree, one folder per condition.

    python scripts/make_transformed_sets.py -- input data/processed/test 
        --out data/transformed --conditions jpeg_q30,blur_s2.0,chain_resize50_jpeg70

Produces, preserving the class subfolders so it stays ImageFolder-compatible:

    data/transformed/jpeg_q30/real/...   jpeg_q30/fake/...
    data/transformed/blur_s2.0/...
    data/transformed/MANIFEST.json       exact params, seeds, backend, checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from aigcdet.augmentations import (ASSUMPTIONS, EVAL_BY_NAME, blur_backend,  # noqa: E402
                                   canonicalize, eval_names, get_eval_transform)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _digest(p: Path) -> str:
    return hashlib.blake2b(p.read_bytes(), digest_size=8).hexdigest()


def process_one(src: Path, in_root: Path, out_root: Path, condition: str,
                fmt: str, do_canon: bool, quality: int) -> dict:
    rel = src.relative_to(in_root)
    key = str(rel)                      # the SAME key evaluate.py seeds with
    spec = get_eval_transform(condition)
    with Image.open(src) as im:
        img = im.convert("RGB")
    if do_canon:
        img = canonicalize(img)
    out_img = spec.apply(img, key)
    ext = ".png" if fmt == "png" else ".jpg"
    dst = out_root / condition / rel.with_suffix(ext)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        out_img.save(dst, format="PNG", optimize=False)
    else:
        out_img.save(dst, format="JPEG", quality=quality, subsampling="4:2:0")
    return {"src": key, "dst": str(dst.relative_to(out_root)), "size": list(out_img.size),
            "sha": _digest(dst)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Materialise transformed image sets")
    ap.add_argument("--input", required=True, help="root of the clean image tree")
    ap.add_argument("--out", default="data/transformed")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated; default = every condition (WARNING: ~17x disk)")
    ap.add_argument("--format", choices=["png", "jpeg"], default="png")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="only used with --format jpeg; the container encode, not the transform")
    ap.add_argument("--canonicalize", action="store_true",
                    help="apply input canonicalisation before the transform")
    ap.add_argument("--limit", type=int, default=None, help="first N images (visual QA)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="print the condition list and exit")
    a = ap.parse_args(argv)

    if a.list:
        for n in eval_names():
            s = EVAL_BY_NAME[n]
            print(f"{n:26s} family={s.family:8s} {s.label:14s} {s.real_world_analog}")
        return 0

    in_root = Path(a.input)
    out_root = Path(a.out)
    if not in_root.is_dir():
        raise SystemExit(f"{in_root} is not a directory")
    files = sorted(p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit(f"no images under {in_root}")
    conditions = a.conditions.split(",") if a.conditions else eval_names()
    for c in conditions:
        if c not in EVAL_BY_NAME:
            raise SystemExit(f"unknown condition {c!r}; --list to see them all")

    print(f"[transform] {len(files)} images x {len(conditions)} conditions "
          f"= {len(files) * len(conditions)} outputs -> {out_root}")
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_root": str(in_root), "format": a.format,
        "canonicalize": bool(a.canonicalize), "blur_backend": blur_backend(),
        "assumptions": ASSUMPTIONS, "conditions": {},
    }
    for cond in conditions:
        spec = EVAL_BY_NAME[cond]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(
                lambda p: process_one(p, in_root, out_root, cond, a.format,
                                      a.canonicalize, a.jpeg_quality), files))
        manifest["conditions"][cond] = {
            "family": spec.family, "label": spec.label, "params": spec.params,
            "stochastic": spec.stochastic,
            "seed_recipe": "blake2b('aigcdet.v1', condition, relative_path)",
            "n": len(rows), "files": rows if len(rows) <= 200 else rows[:200],
            "files_truncated": len(rows) > 200,
        }
        print(f"[transform] {cond:26s} {len(rows)} files ({time.time() - t0:.1f}s)")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[transform] manifest -> {out_root / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
