# Robust Detection of AI-Generated Images Under Real-World Transformations

Hackathon prototype. Detects AI-generated images and, more importantly, keeps
detecting them after the image has been compressed, blurred, resized, noised,
colour-shifted, or cropped on its way across a platform.

**Label convention, project-wide: `1 = AI-generated`, `0 = real.** A false
positive is a real photo flagged as AI. Do not flip this anywhere.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src            # or: pip install -e .

python scripts/smoke_test.py     # 30s, no GPU, no data needed. Run this first.
```

`smoke_test.py` builds a synthetic dataset and exercises the whole chain:
transforms, data gate, evaluation, robustness table, error analysis, and the
graded inference script. If it passes on your machine, the harness works and you
are only waiting on data and a checkpoint.

### The graded script

```bash
python -m aigcdet.predict --input_dir /path/to/images --output predictions.json \
    --model outputs/run_heldout/best_model.pt --canonicalize
```

Output is a JSON list of `{"image_path": ..., "pred": <P(AI-generated) in [0,1]>}`.
It recurses into subdirectories, never crashes on a corrupt file (unreadable
images are listed in `predictions.failures.json`), and works with `--model dummy`
before any checkpoint exists.

---

## Ownership map

| File | Owner | Deliverable |
|---|---|---|
| `scripts/check_data.py`, `src/aigcdet/datasets.py` | Hui Shi | clean dataset folders + sanity report |
| `src/aigcdet/augmentations.py`, `scripts/make_transformed_sets.py`, `tests/` | Yu Yang | transform module |
| `src/aigcdet/models.py`, `src/aigcdet/train.py` | Erika | `best_model.pt` + training logs |
| `src/aigcdet/evaluate.py`, `src/aigcdet/metrics.py` | Celine | Robustness Evaluation Summary (#4) |
| `src/aigcdet/predict.py`, `src/aigcdet/error_analysis.py` | Yi Jun | inference script, Error Analysis Note (#5), README, Devpost, video |


---

## Data contract (Person 1 produces exactly this)

```
data/
  processed/
    train/real/*   train/fake/*
    val/real/*     val/fake/*
    test/real/*    test/fake/*
  benchmark/               # reference-only. NEVER trained on.
    real/*                 # COCO val2017 subset (4,998)
    fake/*                 # DALL-E Advanced subset (8,843)
```

```bash
python scripts/check_data.py --data data/processed --benchmark data/benchmark --leaks
```

Prints `PASSED` or `FAILED`. Nobody starts training on a `FAILED` dataset.
`load_benchmark()` refuses any path containing `train`, so "we did not train on
the validation set" is enforced by code rather than by memory at 4am.

**`--leaks` matters more than it looks.** It hashes decoded, downscaled pixels,
not filenames or bytes. That catches the same image sitting in train as a PNG and
in test as a JPEG — invisible to filename dedup, and it inflates every number in
the report.

---

## The two ideas this project is actually about

### 1. Container bias is the shortcut, and it looks exactly like success

In the reference benchmark the real class is COCO val2017 (already JPEG, ≤640px)
and the fake class is DALL-E output (PNG, square, high-res). A model can separate
those two on **file format and resolution alone**, hit near-perfect validation
AUC, and collapse the moment a real image arrives as a PNG or a fake arrives
re-encoded. This is documented in
[Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/html/2403.17608v1),
which finds that removing compression and resolution bias shifts cross-generator
performance by more than 11 points — the bias was carrying the score. The same
fragility is traced to over-fitted local artefacts in
[GlobalForge](https://arxiv.org/html/2607.14684v1).

`augmentations.canonicalize()` caps every image, both classes, to one long-side
limit and one encoding before anything else touches it. `check_data.py` flags the
bias automatically when one class is >95% one container format and the other
isn't.

**Report clean performance with and without canonicalisation.** The gap is the
size of the shortcut the model was leaning on. It is the most defensible number
in the submission and almost nobody else will have it.

### 2. Train-on-test hiding inside "we trained with augmentation"

If training samples blur at σ ∈ {0.5, 1.0, 2.0} and the robustness table reports
σ ∈ {0.5, 1.0, 2.0}, the table measures memorisation of the exact severities. The
number looks excellent and means nothing, and one judge question ends it.

`TrainAugment` therefore has explicit policies:

| Policy | Behaviour | Use |
|---|---|---|
| `none` | no augmentation | honest lower bound, always run it |
| `continuous` | continuous ranges spanning the eval grid | standard practice |
| `heldout` | continuous ranges **excluding a neighbourhood of every eval point** | headline number |
| `spec` | exactly the eval grid | only to quantify the leakage gap; emits a warning |

Run all four. The ablation table across policies *is* the contribution — closer
to the intent of
[Degradation-Consistent Paired Training](https://arxiv.org/html/2604.10102v1),
which reports +9.1pp on degraded conditions from a training-side intervention.

---

## Transform suite

Exactly the problem statement's grid, plus two composite conditions. Real
reposted images are never degraded one way at a time, and detectors that survive
single transforms often die on chains — which makes those the most informative
rows in the table.

| Condition | Params | Analog |
|---|---|---|
| `jpeg_q{90,70,50,30}` | quality | social re-encode, messaging |
| `blur_s{0.5,1.0,2.0}` | Gaussian σ px | out-of-focus |
| `resize_{0.5,0.25}` | down then back up | thumbnail generation |
| `noise_s{0.02,0.05,0.10}` | σ on a [0,1] scale | low-light sensor noise |
| `jitter_20` | brightness/contrast/sat ±20% | filter apps, auto-enhance |
| `crop_80` | 80% of each side | profile-picture cropping |
| `chain_resize50_jpeg70` | 0.5× then q=70 | full repost pipeline |
| `chain_crop80_jpeg50` | crop then q=50 | crop then repost |

```bash
python scripts/make_transformed_sets.py --list          # see them all
python tests/test_augmentations.py                      # 16 spec-compliance tests
python scripts/transform_gallery.py --out outputs/gallery.png   # eyeball them
```

Properties the tests enforce, each of which silently invalidates the whole
robustness table if broken:

- JPEG is a real libjpeg round trip, and distortion is monotone in quality.
- Blur is a true Gaussian (cv2 when available, Pillow's box approximation as
  fallback, and the backend used is recorded in every output manifest).
- Noise σ is verified against a realised pixel standard deviation of σ·255, which
  catches a mis-scaled σ — the most common silent bug in these suites.
- Colour jitter factors stay inside ±20%.
- `crop_80` on a 200×100 image is exactly 160×80 and exactly centred.
- Stochastic conditions are **reproducible per image** but **differ across
  images**. The seed comes from `blake2b`, not Python's `hash()`, which is salted
  per process and would give two teammates different "deterministic" eval sets.
- `heldout` sampling never emits a value within the excluded neighbourhood of an
  eval grid point (400 draws per family).

### Stated assumptions

The brief allows assumptions if stated. These live in
`augmentations.ASSUMPTIONS`, are copied into every output manifest, and are the
readings we committed to:

- **Centre crop 80%** = 80% of each *side length* (64% of area). The crop is not
  resized back up, because profile-picture cropping genuinely discards pixels.
- **Resize** = bicubic down, bicubic back to the original resolution, so only
  detail is lost and the output stays resolution-matched to clean.
- **Blur σ** = Gaussian standard deviation in pixels; kernel is `2·ceil(3σ)+1`.
- **Noise σ** is on a [0,1] intensity scale, so σ=0.05 ≈ 12.8 grey levels.
- **Colour jitter** samples an independent factor in [0.8, 1.2] per property,
  applied brightness → contrast → saturation (these do not commute).

---

## Training

```bash
for P in none continuous heldout spec; do
  python -m aigcdet.train --data data/processed --arch resnet50 \
      --aug-policy $P --canonicalize --epochs 6 --out outputs/run_$P
done
```

Checkpoint selection is on **clean-validation AUC**, not accuracy: the splits can
be imbalanced and accuracy lets a degenerate all-one-class model win. Validation
is never augmented — it selects the checkpoint and sets the operating threshold,
and augmenting it would corrupt both.

Preprocessing (input size, normalisation, canonicalisation flag, the full
augmentation config) is stored **inside** `best_model.pt`. The single most common
way a hackathon detector mysteriously scores ~0.5 AUC at demo time is that
inference resized or normalised differently from training. Storing it removes the
possibility.

Parameter budget: the brief caps models at <2B parameters. ResNet-50 is 25.6M and
`train.py` asserts the limit, so the cap is not your constraint — GPU hours are.
Three seeds of a ResNet-50 beat one run of something exotic, because the story is
"we measured the effect of the intervention", not "we have a model".

---

## Evaluation (Deliverable #4)

```bash
python -m aigcdet.evaluate --data data/processed --split test \
    --model outputs/run_heldout/best_model.pt --canonicalize \
    --out outputs/robustness

python -m aigcdet.evaluate --benchmark data/benchmark \
    --model outputs/run_heldout/best_model.pt --canonicalize \
    --out outputs/benchmark_ref          # reference-only, not a scored result
```

Writes `robustness.md` (paste-ready table), `robustness.csv`, `results.json`
(metrics + git rev + platform + threshold provenance), and `scores.jsonl`
(per-image, per-condition — feeds error analysis).

Three defensible choices:

1. **One threshold**, selected on the clean validation split at 1% FPR, reused
   unchanged for every condition. Re-tuning per transform would let the detector
   cheat by knowing which degradation it is looking at — something no deployed
   system can do.
2. **Transforms applied on the fly** from the clean originals. No 17× disk blowup
   and no risk of evaluating a stale transformed dump after Person 1 rebuilds.
3. **AUC leads, accuracy follows.** On the 4,998 real / 8,843 fake reference set,
   always predicting "fake" already scores 63.9% accuracy. `TPR@1%FPR` is
   reported alongside because the brief calls out false positives explicitly.

---

## Error analysis (Deliverable #5)

```bash
python -m aigcdet.error_analysis --scores outputs/robustness/scores.jsonl \
    --out outputs/error_analysis
```

Produces the note, `failures.csv`, and FP/FN contact sheets. It reads
`scores.jsonl` only — no model, no GPU — so Person 5 can iterate on the write-up
while Person 3 is still training.

The centrepiece is **flip analysis**: not "which images are wrong" but "which
images were *right* when clean and became *wrong* after this transform". That
separates fragility from intrinsic difficulty and is the direct evidence for the
robustness claim.

---

## Limitations

- Robustness is measured against the six specified transforms plus two chains.
  An adversary who knows the detector can strip its cue deliberately; nothing
  here simulates that.
- Held-out **images**, not held-out **generators**. Generalisation to an unseen
  generator is the harder claim and is only partially addressed.
- Scores are separable but not calibrated. `pred` should not be read as a literal
  probability.
- Trained at hackathon scale on subsampled public datasets, so absolute numbers
  are not comparable to published benchmarks.

## Datasets

- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  — 120k images at 32×32 ([dataset card](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset)).
  Useful as a same-day pipeline smoke test; note that 80% centre crop of 32px is
  25px and JPEG q30 on a thumbnail is close to noise, so robustness conclusions
  drawn at that resolution do not transfer to native-resolution evaluation.
- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — 210k train /
  30k val, roughly 140GB download
  ([card](https://huggingface.co/datasets/saberzl/SID_Set/blob/main/README.md)).
  Stream it or sample; do not pull it whole.
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary).
- Reference-only benchmark: COCO val2017 (4,998) + DALL·E Advanced (8,843).
  Never used for training.

## Tooling

Python 3.10+, PyTorch + torchvision, Pillow, NumPy, OpenCV (optional, for the
true Gaussian). `metrics.py` is pure NumPy — no scikit-learn — so the harness runs
in a fresh venv on a laptop with no GPU stack, which is what you want at 3am on
day 3.
