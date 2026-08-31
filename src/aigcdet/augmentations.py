"""
create augmentations package for augmentations later.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ASSUMPTIONS: Dict[str, str] = {
    "crop_fraction": (
        "'Center Crop 80%' is read as 80% of each SIDE LENGTH (=64% of area), "
        "then the cropped region is fed to the model at its own input size. "
        "The crop is NOT resized back to the original resolution, because the "
        "real-world analog (profile-picture cropping) discards those pixels."
    ),
    "resize": (
        "'scale 0.5x / 0.25x then upscale' is a bicubic downscale followed by a "
        "bicubic upscale back to the ORIGINAL resolution, so the output is "
        "resolution-matched to the clean image and only detail is lost."
    ),
    "blur_sigma": (
        "'kernel sigma' is the Gaussian standard deviation in pixels. Kernel "
        "size is 2*ceil(3*sigma)+1. cv2 is used when available (true separable "
        "Gaussian); Pillow's 3-pass box approximation is the fallback and the "
        "backend actually used is recorded in every output manifest."
    ),
    "noise_sigma": (
        "Gaussian noise sigma is expressed on a [0,1] intensity scale, i.e. "
        "sigma=0.05 means 0.05*255 ~= 12.8 grey levels, added i.i.d. per pixel "
        "per channel, then clipped to [0,255]."
    ),
    "color_jitter": (
        "'brightness/contrast/sat. +/-20%' samples an independent multiplicative "
        "factor in [0.8, 1.2] for each of the three properties, applied in the "
        "fixed order brightness -> contrast -> saturation. At eval time the "
        "factors are derived deterministically from the image identity, so the "
        "jitter is random across images but reproducible across runs."
    ),
    "jpeg": (
        "JPEG is a real libjpeg encode/decode round trip at the stated quality "
        "with default 4:2:0 chroma subsampling, not an approximation."
    ),
}

# Blur backend
try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore

    _BLUR_BACKEND = "cv2"
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _BLUR_BACKEND = "pil"


def blur_backend() -> str:
    return _BLUR_BACKEND

# Determinism helper
def stable_seed(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "big") >> 1


def _as_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

# The six augmentations
def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """JPEG compression in quality 90/70/50/30"""
    img = _as_rgb(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # force decode before the buffer goes out of scope
    return _as_rgb(out)


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with standard deviation 'sigma' px."""
    img = _as_rgb(img)
    if sigma <= 0:
        return img.copy()
    if _BLUR_BACKEND == "cv2":
        k = 2 * int(np.ceil(3.0 * sigma)) + 1
        arr = np.asarray(img)
        out = cv2.GaussianBlur(arr, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
        return Image.fromarray(out, mode="RGB")
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_downup(img: Image.Image, scale: float) -> Image.Image:
    """Bicubic downscale by `scale`, then bicubic upscale back to original size."""
    img = _as_rgb(img)
    w, h = img.size
    dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = img.resize((dw, dh), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Adding Gaussian noise, sigma on a [0,1] intensity scale."""
    img = _as_rgb(img)
    if sigma <= 0:
        return img.copy()
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr + rng.normal(0.0, float(sigma), size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(arr.round().astype(np.uint8), mode="RGB")


def color_jitter(
    img: Image.Image,
    strength: float,
    rng: np.random.Generator,
) -> Image.Image:
    """
    'strength' = 0.20 means each factor is approximately U[0.8, 1.2].
    """
    img = _as_rgb(img)
    lo, hi = 1.0 - strength, 1.0 + strength
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(float(rng.uniform(lo, hi)))
    return img


def center_crop(img: Image.Image, fraction: float, by_area: bool = False) -> Image.Image:
    """
    Center crop to crop 80%.
    """
    img = _as_rgb(img)
    side = float(np.sqrt(fraction)) if by_area else float(fraction)
    w, h = img.size
    cw, ch = max(1, int(round(w * side))), max(1, int(round(h * side)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def identity(img: Image.Image) -> Image.Image:
    """The clean baseline. Present so 'clean' is a row in the same table."""
    return _as_rgb(img).copy()

@dataclass(frozen=True)
class TransformSpec:
    """One column of the Robustness Evaluation Summary."""

    name: str                      # used as a folder name and table key
    family: str                    # groups the severity levels for the table
    label: str                     # label for the table cell
    real_world_analog: str
    params: Dict[str, object] = field(default_factory=dict)
    stochastic: bool = False       # needs a seeded RNG -> derived from image id

    def apply(self, img: Image.Image, image_key: str = "") -> Image.Image:
        fn = _EVAL_IMPL[self.name]
        if self.stochastic:
            rng = np.random.default_rng(stable_seed("aigcdet.v1", self.name, image_key))
            return fn(img, rng)
        return fn(img)


_EVAL_IMPL: Dict[str, Callable] = {
    "clean": identity,
    **{f"jpeg_q{q}": (lambda q: lambda im: jpeg_compress(im, q))(q) for q in (90, 70, 50, 30)},
    **{
        f"blur_s{s}": (lambda s: lambda im: gaussian_blur(im, s))(s)
        for s in (0.5, 1.0, 2.0)
    },
    **{
        f"resize_{sc}": (lambda sc: lambda im: resize_downup(im, sc))(sc)
        for sc in (0.5, 0.25)
    },
    **{
        f"noise_s{s}": (lambda s: lambda im, rng: gaussian_noise(im, s, rng))(s)
        for s in (0.02, 0.05, 0.10)
    },
    "jitter_20": lambda im, rng: color_jitter(im, 0.20, rng),
    "crop_80": lambda im: center_crop(im, 0.80),
    "chain_resize50_jpeg70": lambda im: jpeg_compress(resize_downup(im, 0.5), 70),
    "chain_crop80_jpeg50": lambda im: jpeg_compress(center_crop(im, 0.80), 50),
}


def _mk(name, family, label, analog, params, stochastic=False) -> TransformSpec:
    return TransformSpec(name, family, label, analog, params, stochastic)


EVAL_SUITE: Tuple[TransformSpec, ...] = (
    _mk("clean", "clean", "clean", "original upload", {}),
    _mk("jpeg_q90", "jpeg", "q=90", "Social-media re-encode, messaging", {"quality": 90}),
    _mk("jpeg_q70", "jpeg", "q=70", "Social-media re-encode, messaging", {"quality": 70}),
    _mk("jpeg_q50", "jpeg", "q=50", "Social-media re-encode, messaging", {"quality": 50}),
    _mk("jpeg_q30", "jpeg", "q=30", "Social-media re-encode, messaging", {"quality": 30}),
    _mk("blur_s0.5", "blur", "sigma=0.5", "Out-of-focus", {"sigma": 0.5}),
    _mk("blur_s1.0", "blur", "sigma=1.0", "Out-of-focus", {"sigma": 1.0}),
    _mk("blur_s2.0", "blur", "sigma=2.0", "Out-of-focus", {"sigma": 2.0}),
    _mk("resize_0.5", "resize", "0.5x up", "Thumbnail generation", {"scale": 0.5}),
    _mk("resize_0.25", "resize", "0.25x up", "Thumbnail generation", {"scale": 0.25}),
    _mk("noise_s0.02", "noise", "sigma=0.02", "Low-light sensor noise", {"sigma": 0.02}, True),
    _mk("noise_s0.05", "noise", "sigma=0.05", "Low-light sensor noise", {"sigma": 0.05}, True),
    _mk("noise_s0.1", "noise", "sigma=0.10", "Low-light sensor noise", {"sigma": 0.10}, True),
    _mk("jitter_20", "jitter", "+/-20%", "Filter apps, auto-enhance", {"strength": 0.20}, True),
    _mk("crop_80", "crop", "80%", "Profile-picture cropping", {"fraction": 0.80}),
    _mk("chain_resize50_jpeg70", "chain", "0.5x + q70", "Repost pipeline", {}),
    _mk("chain_crop80_jpeg50", "chain", "crop80 + q50", "Crop then repost", {}),
)

EVAL_BY_NAME: Dict[str, TransformSpec] = {t.name: t for t in EVAL_SUITE}
SPEC_ONLY: Tuple[str, ...] = tuple(t.name for t in EVAL_SUITE if t.family != "chain")


def get_eval_transform(name: str) -> TransformSpec:
    if name not in EVAL_BY_NAME:
        raise KeyError(f"unknown transform {name!r}; known: {sorted(EVAL_BY_NAME)}")
    return EVAL_BY_NAME[name]


def eval_names(include_chains: bool = True, include_clean: bool = True) -> List[str]:
    out = [t.name for t in EVAL_SUITE if include_chains or t.family != "chain"]
    if not include_clean:
        out = [n for n in out if n != "clean"]
    return out


def apply_eval_transform(img: Image.Image, name: str, image_key: str = "") -> Image.Image:
    """
    Deterministic transform for evaluation.

    `image_key` should be the image's RELATIVE path inside the dataset root, so
    the same image gets the same noise realisation on every machine. Passing ""
    still works but makes stochastic transforms identical for all images, which
    biases the noise and jitter rows -- don't.
    """
    return get_eval_transform(name).apply(img, image_key)


# Training-time augmentation
# Continuous ranges spanning the eval grid.
_TRAIN_RANGES = {
    "jpeg": (25, 95),        # quality
    "blur": (0.0, 2.5),      # sigma px
    "resize": (0.2, 1.0),    # scale
    "noise": (0.0, 0.12),    # sigma [0,1]
    "jitter": (0.0, 0.25),   # strength
    "crop": (0.7, 1.0),      # side fraction
}

_HELDOUT_POINTS = {
    "jpeg": ([90, 70, 50, 30], 5.0),
    "blur": ([0.5, 1.0, 2.0], 0.15),
    "resize": ([0.5, 0.25], 0.05),
    "noise": ([0.02, 0.05, 0.10], 0.008),
    "jitter": ([0.20], 0.03),
    "crop": ([0.80], 0.03),
}


def _sample_param(family: str, rng: np.random.Generator, policy: str) -> float:
    if policy == "spec":
        return float(rng.choice(_HELDOUT_POINTS[family][0]))
    lo, hi = _TRAIN_RANGES[family]
    for _ in range(64):
        v = float(rng.uniform(lo, hi))
        if policy != "heldout":
            return v
        pts, half = _HELDOUT_POINTS[family]
        if all(abs(v - p) > half for p in pts):
            return v
    return float(rng.uniform(lo, hi))  # pathological range


class TrainAugment:
    FAMILIES: Tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

    def __init__(
        self,
        policy: str = "heldout",
        n_ops: Tuple[int, int] = (0, 2),
        p: float = 0.9,
        families: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if policy not in {"none", "continuous", "heldout", "spec"}:
            raise ValueError(f"bad policy {policy!r}")
        if policy == "spec":
            warnings.warn(
                "TrainAugment(policy='spec') trains on the exact evaluation "
                "parameter grid. Any robustness number produced this way is "
                "train-on-test. Use it only to measure the leakage gap.",
                stacklevel=2,
            )
        self.policy = policy
        self.n_ops = n_ops
        self.p = p
        self.families = tuple(families) if families else self.FAMILIES
        self._rng = np.random.default_rng(seed)

    # Log records of what was used 
    def config(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "n_ops": list(self.n_ops),
            "p": self.p,
            "families": list(self.families),
            "ranges": {k: list(v) for k, v in _TRAIN_RANGES.items()},
            "heldout": {k: [list(v[0]), v[1]] for k, v in _HELDOUT_POINTS.items()},
            "blur_backend": blur_backend(),
        }

    def _apply_one(self, img: Image.Image, family: str, rng) -> Image.Image:
        v = _sample_param(family, rng, self.policy)
        if family == "jpeg":
            return jpeg_compress(img, int(round(v)))
        if family == "blur":
            return gaussian_blur(img, v)
        if family == "resize":
            return resize_downup(img, v)
        if family == "noise":
            return gaussian_noise(img, v, rng)
        if family == "jitter":
            return color_jitter(img, v, rng)
        if family == "crop":
            return center_crop(img, v)
        raise ValueError(family)

    def __call__(self, img: Image.Image, image_key: str | None = None) -> Image.Image:
        if self.policy == "none":
            return _as_rgb(img).copy()
        rng = (
            np.random.default_rng(stable_seed("train", self.policy, image_key))
            if image_key is not None
            else self._rng
        )
        img = _as_rgb(img)
        if rng.random() > self.p:
            return img.copy()
        k = int(rng.integers(self.n_ops[0], self.n_ops[1] + 1))
        if k == 0:
            return img.copy()
        order = list(rng.permutation(len(self.families)))[:k]
        for i in order:
            img = self._apply_one(img, self.families[i], rng)
        return img


def build_train_augment(policy: str = "heldout", **kw) -> TrainAugment:
    """Factory used by configs/*.yaml so the policy is logged, not hardcoded."""
    return TrainAugment(policy=policy, **kw)

# Shortcut / container-bias neutralisation
def canonicalize(
    img: Image.Image,
    max_side: int = 512,
    jpeg_quality: int | None = 95,
) -> Image.Image:
    """
    Strip container bias BEFORE anything else touches the image.

    Report clean accuracy with AND without this
    step: the gap is the size of the shortcut your model was leaning on, and it
    is the most defensible number in the whole submission.

    Apply to train, val and test identically, and apply it before the eval
    transforms so severities are measured on comparable inputs.
    """
    img = _as_rgb(img)
    w, h = img.size
    if max_side and max(w, h) > max_side:
        s = max_side / float(max(w, h))
        img = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BICUBIC)
    if jpeg_quality is not None:
        img = jpeg_compress(img, jpeg_quality)
    return img


__all__ = [
    "ASSUMPTIONS",
    "EVAL_SUITE",
    "EVAL_BY_NAME",
    "SPEC_ONLY",
    "TransformSpec",
    "TrainAugment",
    "apply_eval_transform",
    "blur_backend",
    "build_train_augment",
    "canonicalize",
    "center_crop",
    "color_jitter",
    "eval_names",
    "gaussian_blur",
    "gaussian_noise",
    "get_eval_transform",
    "identity",
    "jpeg_compress",
    "resize_downup",
    "stable_seed",
]
