# AI-Generated Image Detection — Robustness to Real-World Degradation

**TikTok TechJam 2026 — Track 5**

> Detecting AI-generated images is only useful if the detector still works after the image has been through the kind of processing real content goes through on a platform — re-compression, resizing, cropping, filters. This project builds and evaluates an AI image detector's robustness to exactly that: six real-world-motivated image transformations, applied at graded severities, tested against a model trained under four different augmentation policies.

- **Demo video:** _TBD_
- **Devpost submission:** _TBD_
- **Repository:** https://github.com/TechJammers-26/AI-Image-Detection

---

## 1. Project Overview

We fine-tune an **EfficientNet-B0** binary classifier (real vs. AI-generated) on a merged corpus of three public datasets (**CIFAKE**, **SID_Set**, and **WildFake**) and evaluate it not just on clean images, but on a locked grid of **17 degradation conditions** spanning six transform families, plus two realistic chained transforms (e.g., resize-then-recompress, crop-then-recompress) that mimic a "repost" pipeline. EfficientNet-B0 has 5.3M parameters — well inside the competition's 2B cap — and uses only publicly available ImageNet-pretrained weights.

To test whether robustness comes from data augmentation or from a training shortcut, we train four separate checkpoints under different augmentation policies and compare them on the *same* held-out test set and the *same* eval grid:

| Policy | What it does | Why it exists |
|---|---|---|
| `none` | No augmentation | Acts as the baseline model |
| `continuous` | Random severity sampled from a continuous range per transform family | Standard robust-training approach |
| `heldout` | Same as `continuous`, but severities that fall too close to the exact eval-grid points are deliberately rejected and resampled | Prevents overfitting; tests whether the model generalizes to degradation severities it never saw in training, not just ones it memorized |
| `spec` | Trains directly on the exact eval-grid severities | Deliberate leakage check. It measures the train-on-test gap, not a real robustness number |

**Label convention:** `1 = AI-generated`, `0 = real`. All precision and recall figures treat AI-generated as the positive class, so recall answers "of the AI images, how many did we catch?" and a false positive is a real photo flagged as AI.

### Robustness evaluation grid

| Family | Severities tested | Real-world analog |
|---|---|---|
| Clean | — | Original upload |
| JPEG compression | q = 90, 70, 50, 30 | Social-media re-encode, messaging apps |
| Gaussian blur | σ = 0.5, 1.0, 2.0 px | Out-of-focus capture |
| Resize down→up | 0.5×, 0.25× | Thumbnail generation |
| Gaussian noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color jitter | ±20% (brightness/contrast/saturation) | Filter apps, auto-enhance |
| Center crop | 80% of each side | Profile-picture cropping |
| Chained | 0.5× resize + q70 JPEG | Repost pipeline |
| Chained | 80% crop + q50 JPEG | Crop then repost |

For each checkpoint × condition, we report **accuracy, precision, recall, and AUC**, plus the accuracy drop relative to that same checkpoint's clean-image performance — the "Robustness Evaluation Summary" (Deliverable #4).

We also run a **reference-only demo benchmark** on a COCO val2017 (real) + DALL·E Advanced (AI-generated) subset. This set is kept fully separate from training and validation data and does not affect the reported score — it exists purely to sanity-check the model on generators and content it has never seen, for the demo video.

### The container confound

Our source datasets are confounded before we look at any pixels. The real class arrives as JPEG at moderate resolution; the generated class arrives as PNG, square, and larger. A model can separate those two on file format and resolution alone, report a near-perfect AUC, and never learn anything about synthesis. We would have no way to detect that from the score, because a container classifier and a real detector produce the same number on this data. Published work on this bias found that eliminating compression and resolution confounds shifts cross-generator performance by more than 11 points, which is the scale of the effect we are controlling for.

Our control is `augmentations.canonicalize()`, which caps every image in both classes to one long-side limit and one encoding before anything else touches it. Because it is applied identically to both classes, it cannot introduce a class-dependent bias — it can only remove one. `dataset_sanity_check.py` also flags the confound automatically when one class is more than 95% a single container format and the other is not.

This matters most for our false-positive numbers. Without canonicalization, a genuine high-resolution PNG is pushed toward "AI" by the container rule alone, so our false-positive rate would be measured against a real class that happens to be uniformly JPEG — a rate that would not survive real inputs. Canonicalization does cost us signal, since capping resolution and re-encoding both attenuate the high-frequency traces detectors rely on. We accept that cost because a smaller number we can interpret is worth more than a larger number we cannot.

### Stated assumptions

The brief permits documented interpretations. Ours:

- **Center crop 80%** means 80% of each side length (64% of area), and the crop is **not** resized back to the original resolution.
- **Gaussian noise σ** is defined on a [0,1] pixel scale, added per channel, then clipped back to [0,1].
- **Gaussian blur kernel size** is `2·ceil(3σ)+1`, giving full ±3σ support.
- **Color jitter** applies brightness → contrast → saturation in that fixed order; ±20% means a factor sampled from [0.8, 1.2].
- **Resize down→up** uses bicubic in both directions and returns the image to its original resolution, so the degradation is resampling loss rather than a size change.
- **Chained conditions** apply their components in the order named (`chain_resize50_jpeg70` = resize first, then JPEG).
- **Stochastic conditions** (`noise_*`, `jitter_20`) are seeded per image from a BLAKE2b hash of the image's relative path, so the eval set is byte-identical on any machine.

---

## 2. Repository Structure and Script Descriptions

```
AI-Image-Detection/
├── src/aigcdet/
│   └── augmentations.py                # All 6 transforms + EVAL_SUITE + TrainAugment
├── scripts/
│   ├── download_sid.py                 # Downloads SID_Set (Hugging Face, parquet format)
│   ├── download_cifake.py              # Downloads CIFAKE (via kagglehub)
│   ├── dataset_sanity_check.py         # Class balance, corrupt-file, duplicate/near-dupe checks
│   ├── restructure_dataset.py          # Builds the unified ImageFolder train/val/test layout
│   ├── deduplicate_dataset.py          # Removes duplicates flagged by the sanity check
│   ├── make_transformed_sets.py        # Applies transforms to a whole folder
│   └── transform_gallery.py            # Visual sanity-check gallery of all 17 conditions
├── tests/
│   └── test_augmentations.py           # Unit tests for every transform + policy (16 tests)
├── augmentation_scripts/
│   ├── data_pipeline_and_curation.py
│   ├── augmentation_transforms.py      # Module smoke test / gallery / unit tests
│   ├── model_training.py               # policy = none (baseline)
│   ├── augmentation_continuous.py      # policy = continuous
│   ├── augmentations_heldout.py        # policy = heldout
│   ├── augmentation_spec.py            # policy = spec (leakage check)
│   ├── evaluations.py                  # Clean + robustness eval, demo benchmark
│   └── error_analysis.py
├── inference.py                        # Image directory in -> JSON predictions out
├── checkpoints/                        # Trained weights (not committed — see Setup)
├── LICENSE
├── Makefile
├── requirements.txt
└── README.md
```

---

## 3. Setup & Installation

**Requirements:** Python 3.13, a CUDA-capable GPU (all notebooks were developed and run on Colab with a **T4 GPU**).

```bash
git clone https://github.com/TechJammers-26/AI-Image-Detection.git
cd AI-Image-Detection

python3.13 -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Key dependencies (pinned in `requirements.txt`): `torch==2.13.0`, `torchvision==0.28.0`, `Pillow==12.3.0`, `imagehash==4.3.2` (near-duplicate detection), `numpy==2.5.2`, `kagglehub==1.0.2`.

**Trained checkpoints are not distributed in this repository.** Reviewers who want to reproduce our numbers should retrain via Step 3 below; each policy takes roughly one Colab T4 session at 6 epochs.

### Dataset access

1. **CIFAKE** — downloaded automatically via `kagglehub` (no credentials needed, public dataset).
2. **SID_Set** — downloaded with `download_sid.py`, via the Hugging Face `datasets` library. Ships as parquet (`image`, `mask`, `label` columns, where `0 = real`, `1 = full_synthetic`, `2 = tampered`).
3. **WildFake** — downloaded from `/Images/Other_based.zip` within the WildFake repository. This subset is entirely AI-generated; it contains no real-image examples.
4. **Reference demo benchmark** — COCO val2017 (real) + a curated DALL·E Advanced subset (AI-generated), provided as `cocoDemo.zip` and `DallEDemo.zip`. Not used for training; used only for benchmarking during evaluation.

The downloaded data for CIFAKE, SID_Set, and WildFake is as follows:

| Source | Train real | Train fake | Val real | Val fake | Test real | Test fake | Source total |
|---|---|---|---|---|---|---|---|
| CIFAKE | 42,500 | 41,397 | 7,500 | 7,305 | 10,000 | 10,000 | 118,702 |
| SID_Set | 44 | 96 | 9 | 21 | 10 | 20 | 200 |
| WildFake | 0 | 350 | 0 | 75 | 0 | 75 | 500 |
| **Split total** | **42,544** | **41,843** | **7,509** | **7,401** | **10,010** | **10,095** | **119,402** |

For Colab-based notebooks: the cleaned dataset zips (`sid_clean.zip`, `cifake_clean.zip`, `wildfake_clean.zip`, `cocoDemo.zip`, `DallEDemo.zip`) live in `/content/drive/MyDrive/TikTok TechJam/dataset`. Each notebook mounts Drive and pulls from that path via `GOOGLE_DRIVE_DATASET_PATH`.

---

## 4. Steps to Reproduce

### Step 1 — Data pipeline (`data_pipeline_and_curation.py`)

All zip files are downloaded and run through `dataset_sanity_check.py` (class balance, corrupted-file check, and exact/near-duplicate detection — important for catching cross-split leakage), then `restructure_dataset.py` and `deduplicate_dataset.py` (the latter only if `dataset_sanity_check.py` reports exact duplicates) to produce a unified ImageFolder layout:

```
dataset/
├── train/{real,fake}/
├── val/{real,fake}/
└── test/{real,fake}/
```

This step also prepares the COCO/DALL·E reference demo set, kept in a separate folder from the training data. Output: `sid_clean.zip`, `cifake_clean.zip`, `wildfake_clean.zip`, `cocoDemo.zip`, `DallEDemo.zip`.

### Step 2 — Verify the augmentation module (`augmentation_transforms.py`)

Run this before training. It renders the visual gallery (all 17 conditions against a test image) and runs the unit test suite (`pytest tests/test_augmentations.py`, 16 tests) to confirm `augmentations.py` matches the spec table exactly before anyone builds on top of it.

The 17 conditions are:

| # | Condition name | Family | Params | Stochastic | Real-world analog |
|:---:|---|---|---|:---:|---|
| 1 | `clean` | clean | — | No | Original upload |
| 2 | `jpeg_q90` | jpeg | quality = 90 | No | Social re-encode, messaging |
| 3 | `jpeg_q70` | jpeg | quality = 70 | No | Social re-encode, messaging |
| 4 | `jpeg_q50` | jpeg | quality = 50 | No | Social re-encode, messaging |
| 5 | `jpeg_q30` | jpeg | quality = 30 | No | Social re-encode, messaging |
| 6 | `blur_s0.5` | blur | sigma = 0.5 | No | Out-of-focus capture |
| 7 | `blur_s1.0` | blur | sigma = 1.0 | No | Out-of-focus capture |
| 8 | `blur_s2.0` | blur | sigma = 2.0 | No | Out-of-focus capture |
| 9 | `resize_0.5` | resize | scale = 0.5 | No | Thumbnail generation |
| 10 | `resize_0.25` | resize | scale = 0.25 | No | Thumbnail generation |
| 11 | `noise_s0.02` | noise | sigma = 0.02 | Yes | Low-light sensor noise |
| 12 | `noise_s0.05` | noise | sigma = 0.05 | Yes | Low-light sensor noise |
| 13 | `noise_s0.1` | noise | sigma = 0.1 | Yes | Low-light sensor noise |
| 14 | `jitter_20` | jitter | strength = 0.2 | Yes | Filter apps, auto-enhance |
| 15 | `crop_80` | crop | fraction = 0.8 | No | Profile-picture cropping |
| 16 | `chain_resize50_jpeg70` | chain | 0.5× + q70 | No | Full repost pipeline |
| 17 | `chain_crop80_jpeg50` | chain | crop 80% + q50 | No | Crop then repost |

Fifteen of the 17 are fully deterministic. `clean` is the undegraded reference used to compute accuracy drop, so 16 are actual degradations.

### Step 3 — Train four checkpoints (`none`, `spec`, `continuous`, `heldout`)

Each notebook loads the consolidated dataset, builds `EfficientNet-B0` with an ImageNet-pretrained backbone and a replaced single-logit classifier head, and trains for 6 epochs (`AdamW`, cosine LR schedule), saving the best-AUC checkpoint:

| Notebook | Policy | Output checkpoint |
|---|---|---|
| `model_training.py` | `none` | `efficientnet_b0_none_best.pth` |
| `augmentation_continuous.py` | `continuous` | `efficientnet_b0_continuous_best.pth` |
| `augmentations_heldout.py` | `heldout` | `efficientnet_b0_heldout_best.pth` |
| `augmentation_spec.py` | `spec` | `efficientnet_b0_spec_best.pth` |

Run each with **Runtime → Restart runtime → Run all** for a reproducible result.

### Step 4 — Evaluate (`evaluations.py`)

Loads all four checkpoints and:

- Computes **accuracy, precision, recall, and AUC** on the untouched test split.
- Runs every checkpoint through all 17 eval-grid conditions, deterministically — each image's stochastic transforms are seeded from the image's own file path, so results are identical on any machine.
- Aggregates into the **Robustness Evaluation Summary** table (per-condition accuracy/AUC/loss for all four policies, plus accuracy-drop-from-clean).
- Runs the reference-only COCO/DALL·E demo benchmark.

This outputs the CSVs and tables used for the demo video and the robustness evaluation summary.

### Step 5 — Error analysis & inference

- **Error analysis** pulls false positives and false negatives from the evaluation output, looks for patterns (which conditions, which source dataset, which severity), and writes the Error Analysis Note.
- **Inference:**

  ```bash
  python3 inference.py --input_dir <folder_of_images> --output preds.json
  ```

  This produces a JSON list of `{"image_path": "...", "pred": 0.9312}`, where `pred` is P(AI-generated) as a float in [0,1]. Images that fail to decode are recorded in a `preds.failures.json` sidecar rather than dropped silently. Test this end-to-end against a small folder before recording the demo.

---

## 5. Limitations & What We'd Improve With More Time

- **Dataset diversity across sources (not class balance).** Real-vs-fake balance is close to even in every split (train 42,544 real / 41,843 fake; val 7,509 / 7,401; test 10,010 / 10,095). However, because the SID_Set and WildFake files are large, we mainly opted for CIFAKE images. CIFAKE accounts for the large majority of images by volume, while WildFake and SID_Set — the two sources meant to add variety beyond CIFAKE's generator — contribute far fewer (WildFake: 350/75/75 across train/val/test, and fake-only; SID_Set: 200 images total). Near-ceiling clean-validation numbers across all four policies likely reflect CIFAKE's scale more than true cross-generator robustness. Given more time, we'd oversample WildFake and SID_Set relative to CIFAKE (or cap CIFAKE's contribution) so no single generator dominates what the model learns, and report metrics per source dataset separately so this doesn't hide inside an aggregate number.
- **Single fixed decision threshold.** All reported accuracy, precision, and recall use a 0.5 cutoff on the sigmoid output. We haven't calibrated or swept the threshold per degradation condition, even though confidence naturally shifts under heavy degradation. A proper threshold analysis or a calibration step would make the precision and recall numbers more representative of a real deployment setting.
- **One backbone only.** We evaluated EfficientNet-B0 exclusively; we didn't get to compare against a ResNet or a small ViT backbone to see whether the robustness gap between policies is architecture-dependent.
- **A curated, not adversarial, degradation set.** The 17-condition grid covers common real-world processing (compression, blur, resize, noise, jitter, crop, and two chained pipelines) but not adversarially optimized perturbations or multi-generation repost chains — re-uploading a repost of a repost. Extending the eval grid to compounding degradations would be a natural next step.
- **Reference benchmark size.** The COCO/DALL·E reference-only set is intentionally small (demo-video scale) rather than a large independent test set. It is useful as a sanity check, not as a statistically powered generalization claim.

---

## 6. Team Member Contributions

| Name | Role | Responsibilities | Deliverable owned |
|---|---|---|---|
| Hui Shi | **Data Pipeline & Dataset Curation** | Downloaded and curated CIFAKE, SID_Set, and WildFake; built the ImageFolder train/val/test restructuring; ran class-balance, corruption, and duplicate sanity checks; prepared the reference-only COCO + DALL·E demo benchmark set, kept separate from training data. | Clean, ready-to-use dataset folders |
| Yu Yang | **Augmentation / Transform Module** | Built all 6 transform functions to spec (JPEG compression, Gaussian blur, resize down→up, Gaussian noise, color jitter, center crop); built a folder-level batch wrapper; unit-tested every transform against the spec table; verified augmentations by visual inspection. | `augmentations.py`, `make_transformed_sets.py`, `transform_gallery.py` |
| Erika & Celine | **Model & Training Pipeline** | Selected and set up the EfficientNet-B0 backbone; created and adapted the training script; integrated the augmentation module for the robust-training runs; trained and tuned all four policy checkpoints. | Trained checkpoints + training logs and loss curves |
| Erika & Celine | **Evaluation & Robustness Analysis** | Built the clean eval script (accuracy, precision, recall, AUC); built the robustness eval script running the test set through every transform × severity; aggregated results into the Robustness Evaluation Summary; ran the COCO/DALL·E reference benchmark. | Robustness Evaluation Summary (Deliverable #4) |
| Yi Jiun | **Error Analysis, Inference Script & Packaging** | Built the error analysis script and wrote the Error Analysis Note; built and tested the inference script end-to-end; owns this README, the Devpost writeup, and the demo video. | README, Devpost description, demo video, inference script |

---

## 7. Acknowledgments

Built for **TikTok TechJam 2026, Track 5**. Uses the CIFAKE, SID_Set, and WildFake datasets, and a reference-only subset of COCO val2017 and DALL·E Advanced generations for demo purposes.
