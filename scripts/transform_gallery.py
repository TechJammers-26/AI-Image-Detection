#!/usr/bin/env python3
"""
transform_gallery.py  --  OWNER: Person 2.  Visual QA + a demo-video asset.
    python scripts/transform_gallery.py --out outputs/gallery.png
    python scripts/transform_gallery.py --image path/to/real.jpg --out outputs/g.png

Produces a PNG we look at ourselves.
Unit tests prove the numbers are right. This proves the numbers LOOK right.
For e.g. A sigma=2.0 blur that renders as an obviously unblurred image means the parameter
is being interpreted differently from how it is documented, and every row of the
robustness table inherits that mistake.


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from aigcdet.augmentations import (EVAL_BY_NAME, blur_backend, canonicalize,  # noqa: E402
                                   eval_names)


def synthetic_sample(size: int = 210) -> Image.Image:
    """
    Four-quadrant test card, built AT the display size so the sheet renders 1:1
    and no resampling can hide a transform's effect.

      top-left     2px checkerboard   -> dies first under blur / resize
      top-right    smooth gradient    -> shows JPEG blocking and noise
      bottom-left  concentric rings   -> shows the low-pass cutoff directly
      bottom-right colour patches      -> shows jitter
    """
    n = size
    h = w = n
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float64)
    half = n // 2

    check = (((xx // 2) + (yy // 2)) % 2) * 255.0
    img[:half, :half] = check[:half, :half, None]

    img[:half, half:, 0] = 255 * (xx[:half, half:] - half) / max(w - half, 1)
    img[:half, half:, 1] = 255 * yy[:half, half:] / max(half, 1)
    img[:half, half:, 2] = 110

    ring = np.sin(np.hypot(xx - half * 0.5, yy - h * 0.78) / 2.0) * 115 + 128
    img[half:, :half] = ring[half:, :half, None]

    patches = [(230, 30, 30), (30, 200, 30), (30, 60, 230), (240, 220, 40)]
    qh, qw = (h - half) // 2, (w - half) // 2
    for i, c in enumerate(patches):
        y0 = half + (i // 2) * qh
        x0 = half + (i % 2) * qw
        img[y0 : y0 + qh, x0 : x0 + qw] = c
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def _fit(img: Image.Image, tile: int) -> Image.Image:
    """
    Place `img` in a tile x tile canvas at NATIVE resolution: centre-crop if it is
    bigger, centre-pad if smaller. Never resample.

    This matters. An earlier version resized every tile to fit, and the resize
    itself low-pass filtered the image -- sigma=2.0 blur and clean rendered
    almost identically, which would have passed visual QA while hiding whether
    the blur was applied at all.
    """
    canvas = Image.new("RGB", (tile, tile), (32, 32, 36))
    w, h = img.size
    if w > tile or h > tile:
        left, top = max((w - tile) // 2, 0), max((h - tile) // 2, 0)
        img = img.crop((left, top, left + min(w, tile), top + min(h, tile)))
        w, h = img.size
    canvas.paste(img, ((tile - w) // 2, (tile - h) // 2))
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Visual QA sheet for every transform")
    ap.add_argument("--image", default=None, help="sample image; omit for a synthetic test card")
    ap.add_argument("--out", default="outputs/gallery.png")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=210)
    ap.add_argument("--canonicalize", action="store_true")
    ap.add_argument("--diff", action="store_true",
                    help="render the amplified residual vs clean instead of the image")
    a = ap.parse_args(argv)

    if a.image:
        with Image.open(a.image) as im:
            base = im.convert("RGB")
    else:
        base = synthetic_sample(a.tile)
    if a.canonicalize:
        base = canonicalize(base)
    key = "gallery/sample.png"
    clean_ref = np.asarray(_fit(base, a.tile), dtype=np.float64)

    names = eval_names()
    cols = a.cols
    rows = (len(names) + cols - 1) // cols
    pad, cap = 8, 34
    tile = a.tile
    sheet = Image.new("RGB", (cols * (tile + pad) + pad, rows * (tile + cap + pad) + pad + 26),
                      (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), f"Transform suite QA  --  blur backend: {blur_backend()}"
                        f"{'  |  residual x8 vs clean' if a.diff else ''}",
              fill=(235, 235, 240))

    for i, name in enumerate(names):
        spec = EVAL_BY_NAME[name]
        out = _fit(spec.apply(base, key), tile)
        if a.diff and name != "clean":
            d = (np.asarray(out, np.float64) - clean_ref) * 8.0 + 128.0
            out = Image.fromarray(np.clip(d, 0, 255).astype(np.uint8), "RGB")
        cx, cy = i % cols, i // cols
        x = pad + cx * (tile + pad)
        y = pad + 26 + cy * (tile + cap + pad)
        sheet.paste(out, (x, y))
        draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(70, 70, 78))
        draw.text((x + 2, y + tile + 4), name, fill=(240, 240, 245))
        draw.text((x + 2, y + tile + 18), f"{spec.family} / {spec.label}", fill=(150, 150, 160))

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(outp)
    print(f"[gallery] {len(names)} conditions -> {outp}  ({sheet.size[0]}x{sheet.size[1]})")
    print("[gallery] tiles are 1:1 native pixels (centre-cropped / padded, never resized).")
    print("[gallery] check by eye: q30 blocks the gradient, sigma=2.0 greys out the "
          "checkerboard and softens the rings, 0.25x destroys both, crop_80 is a smaller "
          "tile, jitter shifts the colour patches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
