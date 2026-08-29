"""
Unit tests for augmentations.py 

Checks for the following:
  - every spec transform is registered and runnable
  - parameters actually match the spec table
  - eval transforms are deterministic across processes
  - stochastic eval transforms differ BETWEEN images (else noise/jitter rows lie)
  - severity is monotone (q30 degrades more than q90) -- catches inverted params
  - training policy 'heldout' never emits an eval grid value

Run:  python -m pytest tests -q or python tests/test_augmentations.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import warnings

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aigcdet import augmentations as A  # noqa: E402


# -- fixtures --
def sample_image(seed: int = 0, size=(256, 192)) -> Image.Image:
    """
    Structured + textured test image. A flat colour block would pass blur and
    JPEG tests trivially, so we need to use real high-frequency content.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        127
        + 90 * np.sin(xx / 6.0)
        + 60 * np.cos(yy / 9.0)
        + rng.normal(0, 18, size=(h, w))
    )
    arr = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=0)], axis=-1)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def mse(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    assert x.shape == y.shape, f"shape mismatch {x.shape} vs {y.shape}"
    return float(np.mean((x - y) ** 2))


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# -- suite wiring --
def test_suite_is_complete_and_wired():
    """Every spec row exists, every entry has an implementation."""
    expected = {
        "clean",
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
        "blur_s0.5", "blur_s1.0", "blur_s2.0",
        "resize_0.5", "resize_0.25",
        "noise_s0.02", "noise_s0.05", "noise_s0.1",
        "jitter_20",
        "crop_80",
        "chain_resize50_jpeg70", "chain_crop80_jpeg50",
    }
    got = set(A.EVAL_BY_NAME)
    _check(got == expected, f"suite mismatch\nmissing={expected - got}\nextra={got - expected}")

    img = sample_image()
    for name in A.eval_names():
        out = A.apply_eval_transform(img, name, image_key="a/b.png")
        _check(isinstance(out, Image.Image), f"{name} did not return a PIL image")
        _check(out.mode == "RGB", f"{name} returned mode {out.mode}, expected RGB")
        _check(np.asarray(out).dtype == np.uint8, f"{name} returned non-uint8")

    # 14 spec severities (4 jpeg + 3 blur + 2 resize + 3 noise + 1 jitter + 1 crop)
    # + clean = 15 non-chain rows; plus 2 composite chains.
    _check(len(A.SPEC_ONLY) == 15, f"expected 15 spec rows incl. clean, got {len(A.SPEC_ONLY)}")


def test_shapes_preserved_except_crop():
    """Only the crop family may change resolution. Anything else will be flagged"""
    img = sample_image()
    for name in A.eval_names():
        out = A.apply_eval_transform(img, name, "k")
        fam = A.EVAL_BY_NAME[name].family
        if fam in ("crop", "chain") and "crop" in name:
            _check(out.size < img.size, f"{name} should shrink, got {out.size}")
        else:
            _check(out.size == img.size, f"{name} changed size {img.size} -> {out.size}")


def test_jpeg_is_a_real_roundtrip_and_monotone():
    """catching the event where lower quality produces less distortion"""
    img = sample_image()
    errs = {}
    for q in (90, 70, 50, 30):
        out = A.apply_eval_transform(img, f"jpeg_q{q}", "k")
        errs[q] = mse(img, out)
        # Encoding at the same quality again must reproduce the same bytes.
        b1, b2 = io.BytesIO(), io.BytesIO()
        A.jpeg_compress(img, q).save(b1, format="PNG")
        A.jpeg_compress(img, q).save(b2, format="PNG")
        _check(b1.getvalue() == b2.getvalue(), f"jpeg q={q} is not deterministic")
    _check(errs[30] > errs[50] > errs[70] > errs[90] > 0,
           f"JPEG distortion not monotone in quality: {errs}")


def test_blur_sigma_monotone_and_lowpass():
    img = sample_image()
    prev_err, prev_hf = 0.0, None
    for s in (0.5, 1.0, 2.0):
        out = A.apply_eval_transform(img, f"blur_s{s}", "k")
        err = mse(img, out)
        # High-frequency energy must fall as sigma rises.
        hf = float(np.std(np.diff(np.asarray(out.convert("L"), dtype=np.float64), axis=1)))
        _check(err > prev_err, f"blur sigma={s} distorted less than the previous level")
        if prev_hf is not None:
            _check(hf < prev_hf, f"blur sigma={s} did not reduce high-freq energy")
        prev_err, prev_hf = err, hf
    _check(A.blur_backend() in ("cv2", "pil"), "unknown blur backend")


def test_resize_returns_to_original_size_and_loses_detail():
    img = sample_image()
    e50 = mse(img, A.apply_eval_transform(img, "resize_0.5", "k"))
    e25 = mse(img, A.apply_eval_transform(img, "resize_0.25", "k"))
    _check(e25 > e50 > 0, f"0.25x should lose more detail than 0.5x ({e25} vs {e50})")


def test_noise_sigma_matches_spec_scale():
    """
    sigma is on a [0,1] scale, so the realised per-pixel std must be approx sigma*255.
    Clipping at the extremes pulls it slightly low; 15% tolerance is generous
    enough for that and tight enough to catch a wrong scale (e.g. sigma=0.05
    interpreted as 0.05 grey levels, or as 5%-of-value multiplicative noise).
    """
    img = sample_image()
    for s in (0.02, 0.05, 0.10):
        name = f"noise_s{s}"
        out = A.apply_eval_transform(img, name, "k")
        realised = float(np.std(np.asarray(out, np.float64) - np.asarray(img, np.float64)))
        target = s * 255.0
        _check(abs(realised - target) / target < 0.15,
               f"{name}: realised std {realised:.2f} vs expected {target:.2f}")


def test_color_jitter_within_20_percent():
    """
    Every factor lives in [0.8, 1.2], so mean luminance cannot flucutate too much.
    Checked across many images because a single draw could sit near 1.0.
    """
    ratios = []
    for i in range(60):
        img = sample_image(seed=i)
        out = A.apply_eval_transform(img, "jitter_20", f"img{i}.png")
        m_in = float(np.mean(np.asarray(img.convert("L"), np.float64)))
        m_out = float(np.mean(np.asarray(out.convert("L"), np.float64)))
        ratios.append(m_out / max(m_in, 1e-6))
    _check(max(ratios) < 1.55 and min(ratios) > 0.55,
           f"jitter luminance ratio out of plausible range: [{min(ratios):.3f}, {max(ratios):.3f}]")
    _check(np.std(ratios) > 1e-3, "jitter appears to be a no-op across images")


def test_center_crop_80_percent_of_side():
    img = sample_image(size=(200, 100))
    out = A.apply_eval_transform(img, "crop_80", "k")
    _check(out.size == (160, 80), f"expected (160, 80) for 80% of each side, got {out.size}")
    by_area = A.center_crop(img, 0.80, by_area=True)
    _check(by_area.size == (179, 89), f"by-area crop unexpected: {by_area.size}")
    # Crop must be centered: content should match the central region exactly.
    ref = np.asarray(img)[10:90, 20:180]
    _check(np.array_equal(np.asarray(out), ref), "crop is not centered")


# -- stochastics --
def test_stochastic_eval_is_deterministic_per_image():
    img = sample_image()
    for name in ("noise_s0.05", "jitter_20"):
        a = A.apply_eval_transform(img, name, "val/fake/0001.png")
        b = A.apply_eval_transform(img, name, "val/fake/0001.png")
        _check(np.array_equal(np.asarray(a), np.asarray(b)),
               f"{name} not reproducible for a fixed image key")


def test_stochastic_eval_differs_across_images():
    """
    If the seed ignored the image key, every image would get the same noise
    field and the same jitter factors -- the noise/jitter rows would then be
    measuring one fixed perturbation, not a distribution.
    """
    img = sample_image()
    a = A.apply_eval_transform(img, "noise_s0.05", "val/real/0001.png")
    b = A.apply_eval_transform(img, "noise_s0.05", "val/real/0002.png")
    _check(not np.array_equal(np.asarray(a), np.asarray(b)),
           "noise realisation does not depend on the image key")


def test_seed_is_stable_across_processes():
    """
    Guards against anyone 'simplifying' stable_seed() into hash(), which is
    salted per process and would make teammates' eval sets disagree.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from aigcdet.augmentations import stable_seed;"
        "print(stable_seed('aigcdet.v1', 'noise_s0.05', 'val/real/1.png'))"
        % os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    r1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    env["PYTHONHASHSEED"] = "999"
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    _check(r1.returncode == 0 and r2.returncode == 0, f"subprocess failed: {r1.stderr}{r2.stderr}")
    _check(r1.stdout.strip() == r2.stdout.strip(),
           f"seed changed with PYTHONHASHSEED: {r1.stdout!r} vs {r2.stdout!r}")


# -- train policies --
def test_heldout_policy_avoids_eval_grid():
    rng_names = ("jpeg", "blur", "resize", "noise", "jitter", "crop")
    rng = np.random.default_rng(7)
    for fam in rng_names:
        pts, half = A._HELDOUT_POINTS[fam]
        for _ in range(400):
            v = A._sample_param(fam, rng, "heldout")
            for p in pts:
                _check(abs(v - p) > half,
                       f"heldout sampled {v} within {half} of eval point {p} ({fam})")


def test_spec_policy_warns_and_hits_grid():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aug = A.TrainAugment(policy="spec", seed=0)
        _check(any("train-on-test" in str(x.message) for x in w),
               "policy='spec' must warn about leakage")
    rng = np.random.default_rng(0)
    vals = {A._sample_param("jpeg", rng, "spec") for _ in range(50)}
    _check(vals <= {90.0, 70.0, 50.0, 30.0}, f"spec policy emitted off-grid values: {vals}")


def test_train_augment_runs_and_is_reproducible():
    img = sample_image()
    for policy in ("none", "continuous", "heldout"):
        aug = A.TrainAugment(policy=policy, seed=0)
        out = aug(img, image_key="train/fake/7.png")
        _check(isinstance(out, Image.Image) and out.mode == "RGB", f"{policy} bad output")
        again = A.TrainAugment(policy=policy, seed=0)(img, image_key="train/fake/7.png")
        _check(np.array_equal(np.asarray(out), np.asarray(again)),
               f"{policy} not reproducible for a fixed image key")
    none_out = A.TrainAugment(policy="none")(img)
    _check(mse(img, none_out) == 0.0, "policy='none' must be a no-op")
    cfg = A.TrainAugment(policy="heldout").config()
    for key in ("policy", "ranges", "heldout", "blur_backend"):
        _check(key in cfg, f"config() missing {key}")


def test_canonicalize_removes_container_bias():
    """A PNG 'real' and a PNG 'fake' at different sizes must come out matched."""
    big = sample_image(size=(1024, 1024))
    small = sample_image(size=(640, 480), seed=3)
    cb, cs = A.canonicalize(big), A.canonicalize(small)
    _check(max(cb.size) == 512, f"canonicalize did not cap the long side: {cb.size}")
    _check(max(cs.size) == 512, f"canonicalize did not cap the long side: {cs.size}")
    _check(abs(cb.size[0] / cb.size[1] - 1.0) < 0.01, "aspect ratio not preserved")
    _check(abs(cs.size[0] / cs.size[1] - 640 / 480) < 0.02, "aspect ratio not preserved")


def test_assumptions_are_documented():
    for key in ("crop_fraction", "resize", "blur_sigma", "noise_sigma", "color_jitter", "jpeg"):
        _check(key in A.ASSUMPTIONS and len(A.ASSUMPTIONS[key]) > 40,
               f"ASSUMPTIONS[{key!r}] missing or too thin to put in the README")


# -- runner --
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed  (blur backend: {A.blur_backend()})")
    sys.exit(1 if failed else 0)
