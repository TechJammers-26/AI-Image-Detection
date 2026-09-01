# AI-Generated Image Detection — Robustness to Real-World Degradation

**TikTok TechJam 2026 — Track 5**

> Detecting AI-generated images is only useful if the detector still works after the image has been through the kind of processing real content goes through on a platform — re-compression, resizing, cropping, filters. This project builds and evaluates an AI image detector's robustness to exactly that: six real-world-motivated image transformations, applied at graded severities, tested against a model trained under four different augmentation policies.

- **Demo video:** https://youtu.be/GYSgM-fP8v4
- **Devpost Submission:** https://devpost.com/software/techjammers
- **Repository:** https://github.com/TechJammers-26/AI-Image-Detection
- **Cleaned Dataset:** https://drive.google.com/drive/folders/19DBJ-A4DaWMVt5lbpMty1tngD19SZZ9G?usp=sharing
---

## 1. Project Overview

We fine-tune an **EfficientNet-B0** binary classifier (real vs. AI-generated) on a merged folder of three public datasets (**CIFAKE**, **SID_Set**, and **WildFake**) and evaluate it not just on clean images, but on a locked grid of **17 degradation conditions** spanning six transform families, plus two realistic chained transforms (e.g., resize-then-recompress, crop-then-recompress) that mimic a "repost" pipeline. EfficientNet-B0 has 5.3M parameters and uses only publicly available ImageNet-pretrained weights.

To test whether robustness comes from data augmentation or from a training shortcut, we train four separate checkpoints under different augmentation policies and compare them on the *same* held-out test set and the *same* eval grid:

| Policy | What it does | Why it exists |
|---|---|---|
| `none` | No augmentation | Acts as the baseline model |
| `continuous` | Random severity sampled from a continuous range per transform family | Standard robust-training approach |
| `heldout` | Same as `continuous`, but severities that fall too close to the exact eval-grid points are deliberately rejected and resampled | Prevents overfitting; tests whether the model generalizes to degradation severities it never saw in training, not just ones it memorized |
| `spec` | Trains directly on the exact eval-grid severities | Deliberate leakage check. It measures the train-on-test gap, not a real robustness number |

**Label convention:** `1 = AI-generated`, `0 = real`.

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

For each checkpoint × condition, we report **accuracy, precision, recall, and AUC**, plus the accuracy drop relative to that same checkpoint's clean-image performance in the "Robustness Evaluation Summary" (Deliverable #4).

We also run a **reference-only demo benchmark** on a COCO val2017 (real) + DALL·E Advanced (AI-generated) subset. This set is kept fully separate from training and validation data and does not affect the reported score. It is only used to evaluate the model's performance on content it has never seen, for the demo video.

### The container confound

Our source datasets are confounded before we look at any pixels. The real class is formattted as JPEG at moderate resolution while the generated class is formatted as PNG, square, and larger. Thus a problem arises, the model might be recognising and learning to differentiate real vs AI based on formatting alone. By eliminating compression and resolution confounds, we can effectively shift the cross-generator performance.

Our solution is `augmentations.canonicalize()`, which caps every image in both classes to one long-side limit and encoding. Because it is applied identically to both classes, it helps to remove the class-dependent biases. `dataset_sanity_check.py` also flags the confound automatically when one class is more than 95% a single container format and the other is not.

This matters most for our false-positive numbers. Without canonicalization, a genuine high-resolution PNG is pushed toward "AI" by the container rule alone, so our false-positive rate would be measured against a real class that happens to be uniformly JPEG. Canonicalization does cost us signal, since capping resolution and re-encoding both attenuate the high-frequency traces detectors rely on. It serves as a devil's advocate, and it improves our models' detection ability but at the cost of accuracy, and this trade-off is acceptable given that ultimately what we want to prioritise developing a robust model with more complexed technicalities compared to a "production-grade service".

### Stated assumptions

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
│   ├── __init__.py
│   └── augmentations.py                # All 6 transforms + EVAL_SUITE + TrainAugment
├── scripts/
│   ├── data_processing/
│   │   ├── data_pipeline_and_curation.py
│   │   ├── dataset_sanity_check.py     # Class balance, corrupt-file, duplicate/near-dupe checks
│   │   ├── deduplicate_dataset.py      # Removes duplicates flagged by the sanity check
│   │   ├── download_cifake.py          # Downloads CIFAKE (via kagglehub)
│   │   ├── download_sid.py             # Downloads SID_Set (Hugging Face, parquet format)
│   │   ├── make_transformed_sets.py    # Applies transforms to a whole folder
│   │   └── transform_gallery.py        # Visual sanity-check gallery of all 17 conditions
│   └── prediction/inference/
│       └── predict.py                  # Image directory in -> JSON predictions out
├── tests/
│   ├── augmentation_test_procedure.py
│   └── test_augmentations.py           # Unit tests for every transform + policy (16 tests)
├── augmentation_scripts/
│   ├── augmentation_continuous.py      # policy = continuous
│   ├── augmentation_spec.py            # policy = spec (leakage check)
│   ├── augmentations_heldout.py        # policy = heldout
│   ├── evaluations.py                  # Clean + robustness eval, demo benchmark
│   └── model_training.py               # policy = none (baseline)
├── best_checkpoints/                   # Trained weights (see Setup)
│   ├── efficientnet_b0_continuous_best.pth
│   ├── efficientnet_b0_heldout_best.pth
│   ├── efficientnet_b0_none_best.pth
│   └── efficientnet_b0_spec_best.pth
├── demo_results/
│   ├── predictions_demo_cont.json
│   ├── predictions_demo_heldout.json
│   └── predictions_demo_spec.json
├── outputs/
│   ├── gallery.png
│   └── gallery_residual.png
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
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

**Trained checkpoints are provided in this repository, but reviewers are highly encouraged to retrain via "#4. Steps to Reproduce" below.** Each policy takes roughly one Colab T4 session at 6 epochs, lasting ~1 hour.

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

For Colab-based notebooks: The cleaned dataset zips (`sid_clean.zip`, `cifake_clean.zip`, `wildfake_clean.zip`, `cocoDemo.zip`, `DallEDemo.zip`) live in `/content/drive/MyDrive/TikTok TechJam/dataset`. Each notebook mounts Drive and pulls from that path via `GOOGLE_DRIVE_DATASET_PATH`.

---

## 4. Steps to Reproduce

### Step 1 — Data pipeline (`data_pipeline_and_curation.py`)

All zip files are downloaded and ran through `dataset_sanity_check.py` (class balance, corrupted-file check, and exact/near-duplicate detection — important for catching cross-split leakage), then `restructure_dataset.py` and `deduplicate_dataset.py` (the latter only if `dataset_sanity_check.py` reports exact duplicates) to produce a unified ImageFolder layout:

```
dataset/
├── train/{real,fake}/
├── val/{real,fake}/
└── test/{real,fake}/
```

This step also prepares the COCO/DALL·E reference demo set, kept in a separate folder from the training data. The DALL-E reference set was downloaded from WildFake, and then filtered to only contain “advanced” images, according to the benchmark. Output: `sid_clean.zip`, `cifake_clean.zip`, `wildfake_clean.zip`, `cocoDemo.zip`, `DallEDemo.zip`.

### Step 2 — Verify the augmentation module (`augmentation_transforms.py`)

Run this before training. It renders the visual gallery (all 17 conditions against a test image) and runs the unit test suite (`pytest tests/test_augmentations.py`, 15 tests) to confirm `augmentations.py` matches the spec table exactly before anyone builds on top of it.

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

UNIT TESTS:

| Test Name | Test Result |
|---|:---:|
| `test_blur_sigma_monotone_and_lowpass` | 🟢 **Pass** |
| `test_canonicalize_removes_container_bias` | 🟢 **Pass** |
| `test_center_crop_80_percent_of_side` | 🟢 **Pass** |
| `test_color_jitter_within_20_percent` | 🟢 **Pass** |
| `test_heldout_policy_avoids_eval_grid` | 🟢 **Pass** |
| `test_jpeg_is_a_real_roundtrip_and_monotone` | 🟢 **Pass** |
| `test_noise_sigma_matches_spec_scale` | 🟢 **Pass** |
| `test_resize_returns_to_original_size_and_loses_detail` | 🟢 **Pass** |
| `test_seed_is_stable_across_processes` | 🟢 **Pass** |
| `test_shapes_preserved_except_crop` | 🟢 **Pass** |
| `test_spec_policy_warns_and_hits_grid` | 🟢 **Pass** |
| `test_stochastic_eval_differs_across_images` | 🟢 **Pass** |
| `test_stochastic_eval_is_deterministic_per_image` | 🟢 **Pass** |
| `test_suite_is_complete_and_wired` | 🟢 **Pass** |
| `test_train_augment_runs_and_is_reproducible` | 🟢 **Pass** |

### Step 3 — Train four checkpoints (`none`, `spec`, `continuous`, `heldout`)

4 models were trained concurrently, namely:
`none` -> Baseline with no training-time augmentation. Images are canonicalized only (compression/resolution fingerprint stripped), with no distortions applied. This isolates how well the model performs using only "raw" content-level cues, without any exposure to the transformation types it will later be tested against.

`spec` -> Trained using the exact deterministic parameter values specified by TechJam's fixed evaluation grid, `EVAL_GIRD` (JPEG quality 90/70/50/30; Gaussian blur σ 0.5/1.0/2.0; resize 0.5×/0.25× then upscale; Gaussian noise σ 0.02/0.05/0.10; ±20% color jitter; 80% center crop). This represents the best-case scenario if the model only needs to handle exactly the conditions it will be graded on.

`continuous` -> Trained using randomized, continuous-range augmentation parameters that deliberately do not match the eval grid's exact values, to prevent the model from simply memorizing the specific test conditions rather than learning generalizable robustness.

`heldout` -> Trained using continuous-range augmentation, but with parameter neighborhoods immediately surrounding the eval grid's exact values excluded from training. This tests whether the model generalizes to distortion severities near what its evaluated on without ever training on those exact points.

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
- Fits per-checkpoint temperature scaling and selects a Youden's-J threshold on the validation set.
- Runs the reference-only COCO/DALL·E demo benchmark.

This outputs the CSVs and tables used for the demo video and the robustness evaluation summary.

#### Clean Test Performance

All four policies converge to near-ceiling performance on the untouched, unseen test split:

| Policy | Accuracy | AUC | Precision | Recall | F1 | Loss |
|---|---|---|---|---|---|---|
| none | 0.9851 | 0.9990 | 0.9905 | 0.9797 | 0.9851 | 0.0467 |
| continuous | 0.9852 | 0.9989 | 0.9854 | 0.9850 | 0.9852 | 0.0409 |
| heldout | 0.9838 | 0.9988 | 0.9792 | 0.9887 | 0.9839 | 0.0444 |
| spec | 0.9815 | 0.9985 | 0.9816 | 0.9817 | 0.9816 | 0.0490 |

Based on clean data alone, the policies perform relatively well compared to the baseline, `none`.

#### Mean Distortion Robustness

Averaged across all 16 non-clean conditions, the picture changes sharply:

 Policy | Mean Accuracy | Mean AUC | Mean Precision | Mean Recall | Mean F1 |
|---|---|---|---|---|---|
| none | 0.8065 | 0.9232 | 0.9140 | 0.7110 | 0.7501 |
| **continuous** | **0.9655** | **0.9944** | 0.9698 | 0.9613 | **0.9655** |
| heldout | 0.9635 | 0.9939 | 0.9618 | **0.9659** | 0.9637 |
| spec | 0.9603 | 0.9931 | 0.9627 | 0.9585 | 0.9604 |

`continuous` achieves the highest mean distorted accuracy, but only marginally better than `heldout` and `spec`.
`none`, the no-augmentation baseline, degrades significantly once any distortion is applied, confirming that augmentation-aware training is paramount for robustness on the locked eval grid.

#### Accuracy Drop From Clean

This table allows us to compare the **relative accuracy drop** among the policies, so as to ensure fairness during comparison of how distortion affected each policy.

| Policy | Clean Accuracy | Mean Distorted Accuracy | Accuracy Drop |
|---|---|---|---|
| none | 0.9851 | 0.8065 | 0.1785 |
| **continuous** | 0.9852 | 0.9655 | **0.0197** |
| heldout | 0.9838 | 0.9635 | 0.0203 |
| spec | 0.9815 | 0.9603 | 0.0213 |

`none` drops nearly 18 percentage points once distortions are introduced, the largest fall of any policy by far, which re-inforces the need for augmentation training.
The three augmentation-trained policies all maintained within ~2 percentage points of their clean ceiling.

#### Worst-Case Condition Per Policy (see Appendix below for full table breakdown)

The hardest condition differs by policy, and reveals `none`'s failure mode most clearly:

| Policy | Worst Condition | Accuracy | AUC | F1 |
|---|---|---|---|---|
| continuous | resize_0.25 | 0.9251 | 0.9785 | 0.9255 |
| heldout | resize_0.25 | 0.9159 | 0.9748 | 0.9170 |
| spec | resize_0.25 | 0.9228 | 0.9778 | 0.9249 |
| **none** | **chain_resize50_jpeg70** | **0.5687** | 0.8857 | **0.2510** |

All three augmentation-trained policies bottom out on `resize_0.25` but are still able to maintain 91% accuracy. `none`'s worst case is a compound distortion (50% resize + JPEG-70 re-encode) where accuracy collapses to 56.9%, with F1 falling to 0.251, which is only slightly better than a random uninformed guess, indicating the model is failing to recall the AI class almost entirely under that condition.

#### Calibration (Temperature & Threshold)

Each checkpoint's raw logits are temperature-scaled to correct overconfidence, then a Youden's-J-optimal threshold is selected on calibrated validation probabilities:

| Policy | Temperature (T) | Threshold |
|---|---|---|
| none | 1.4478 | 0.3705 |
| continuous | 1.0477 | 0.4519 |
| heldout | 1.0255 | 0.6187 |
| spec | 1.2205 | 0.4795 |

`none` requires the heaviest softening (T=1.4478), consistent with it having no augmentation exposure to temper its confidence. `heldout`'s comparatively low T (1.0255) but much higher decision threshold (0.6187) reflects a well-calibrated model whose probability mass for the AI class is higher than average, ie it isn't overconfident but just biased toward higher raw scores, which the threshold selection corrects for.

---

### Final Model Selection

#### Out-of-Distribution Generalization (WildFake Reference Benchmark)

As a reference check (not part of the scored evaluation) the four checkpoints were additionally run against a held-out benchmark built from COCO val2017 (4,998 real images) and DALL·E 3 Advanced (8,843 AI images), which were used as a benchmark in the demo version of the TechJam document and were completely absent from training:

| Policy | OOD AUC |
|---|---|
| continuous | 0.6953 |
| none | 0.6524 |
| heldout | 0.5096 |
| spec | 0.4951 |

On this out-of-sample reference set, `heldout` and `spec`, the two policies trained closest to the exact eval-grid parameters, perform noticeably weaker than `continuous` and `none`. Since this benchmark draws from a completely different real-image source and an unseen generator, some drop-off relative to in-distribution performance is expected. The lower AUC suggests `heldout` and `spec` are currently leaning more heavily on cues tied to the training sources than the other two policies.

#### Why Canonicalization Matters

`canonicalize()` strips dataset-level "container" artifacts from an image before it reaches the model, like the specific JPEG compression signature, resolution, and re-encoding history left behind by whichever pipeline originally produced the file. These artifacts have nothing to do with whether an image is AI-generated, but they correlate strongly with *which source dataset* an image came from (SID vs. CIFAKE vs. WildFake, each with their own typical resolution/compression characteristics). Without canonicalization, a model can achieve high accuracy by learning to recognize *which dataset an image belongs to* rather than learning genuine AI-vs-real content cues. This shortcut inflates in-distribution metrics while quietly failing to transfer to any image processed through a different pipeline, hence when evaluating against an unseen dataset from a completely different source, the model might achieve even lower accuracy than expected.

`heldout` is trained with `canonicalize()` applied to force the model to rely on content-level signal rather than dataset fingerprints, hence making it a more robust model despite its lower accuracy.

#### A Confound in the `continuous` OOD Result

`continuous` was trained **without** `canonicalize()`. This matters specifically for the WildFake reference benchmark, because `continuous`'s own training data includes `wildfake_clean` which is a *different* subset of the same WildFake dataset that the reference benchmark's DALL·E-Advanced and CocoVal images are also drawn from. Without canonicalization stripping that pipeline-level signature out, `continuous` may have learned to recognize WildFake's characteristic compression/processing fingerprint during training, and could then be partially "recognizing" that same fingerprint in the reference benchmark's AI-labeled images.

Thus its higher OOD AUC may not be because it generalizes better to unseen AI generators in general, but because it has effectively seen this *dataset's* signature before, just not these *specific images*.

Therefore, `continuous` higher accuracy may not be enough grounds to reasonably justify and be the determining factor of which is the better policy due to the reason mentioned above. This supports `heldout` as the more defensible choice for a proof-of-concept: canonicalization is introduced in this policy to prevent such misleading result, though at the expense of raw accuracy. 

#### Selection

We selected **`heldout`** as our flagship model. Its augmentation design, training on continuous-range distortions while deliberately excluding the exact neighbourhoods around the locked eval grid, is the most methodologically principled of the four for the scored evaluation itself, and it pairs with `canonicalize()` to strip source-specific compression/resolution fingerprints before training, directly targeting the kind of shortcut learning this reference benchmark is sensitive to.

We prioritised a proof-of-concept demonstrating robustness against the locked evaluation grid, rather than a production-grade service, and `heldout`'s in-distribution performance (98.4% clean accuracy, 96.4% mean accuracy across all 16 distortion conditions, only a 2.0% drop from clean) proves this policy is noteworthy.


---

### Step 5 — Error analysis & prediction

**heldout — clean confusion matrix**

| | Predicted Real | Predicted AI |
|---|---|---|
| **True Real** | 9798 | 212 |
| **True AI** | 114 | 9981 |

At the operating threshold (0.5), `heldout` produces roughly twice as many false positives (212 real images called AI) as false negatives (114 AI images called real) on the clean test split. This is a mild bias towards flagging real content as AI, and aligns with its Youden's-J threshold (0.6187) sitting well above 0.5 to counter this.

- **Prediction:**
```bash
  python3 predict.py \
      --input_dir <folder_of_images> \
      --checkpoint efficientnet_b0_heldout_best.pth \
      --output preds.json \
      --temperature 1.0255 \
      --threshold 0.6187
```
This produces a JSON list of `{"image_path": "...", "pred": 0.9312, "label": 1}`, where `pred` is the calibrated P(AI-generated) as a float in [0,1] and `label` is `1` if `pred >= --threshold` else `0`. The command above uses `heldout`'s calibrated values (our selected flagship model, per **Final Model Selection** above). 
To test other policies, one can swap `--checkpoint`, `--temperature`, and `--threshold`. 
Note: This prediction script requires `augmentations.py` on the path (`--project_root` if not running from the same project folder).

---

## 5. Limitations & What We'd Improve With More Time

- **Dataset diversity across sources (not class balance).** Real-vs-fake balance is close to even in every split (train 42,544 real / 41,843 fake; val 7,509 / 7,401; test 10,010 / 10,095). However, because the SID_Set and WildFake files are large, we mainly opted for CIFAKE images. CIFAKE accounts for the large majority of images by volume, while WildFake and SID_Set have significantly less volume (WildFake: 350/75/75 across train/val/test, and fake-only; SID_Set: 200 images total). Near-ceiling clean-validation numbers across all four policies likely reflect CIFAKE's scale more than true cross-generator robustness. Given more time, we would oversample WildFake and SID_Set relative to CIFAKE (or cap CIFAKE's contribution) so no single generator dominates what the model learns and it wont be trained to report based on a specific dataset signature.
- **Single fixed decision threshold.** All reported accuracy, precision, and recall use a 0.5 cutoff on the sigmoid output. We haven't calibrated or swept the threshold per degradation condition, even though confidence naturally shifts under heavy degradation. A proper threshold analysis or a calibration step would make the precision and recall numbers more representative of a real deployment setting.
- **One backbone only.** We evaluated EfficientNet-B0 exclusively; we didn't get to compare against a ResNet or a small ViT backbone to see whether the robustness gap between policies is architecture-dependent.
- **A curated, not adversarial, degradation set.** The 17-condition grid covers common real-world processing (compression, blur, resize, noise, jitter, crop, and two chained pipelines) but there are no optimized perturbations or multi-generation repost chains (re-uploading a repost of a repost). Extending the eval grid to compounding degradations is something we could have done if we had more time.
- **Reference benchmark size.** The COCO/DALL·E reference-only set is intentionally small (demo-video scale) rather than a large independent test set. It is useful as a sanity check but not as a statistically powered generalization claim.

---

## 6. Team Member Contributions

| Name | Role | Responsibilities | Deliverable |
|---|---|---|---|
| Hui Shi | **Data Pipeline & Dataset Curation** | Downloaded and curated CIFAKE, SID_Set, and WildFake; built the ImageFolder train/val/test restructuring; ran class-balance, corruption, and duplicate sanity checks; prepared the reference-only COCO + DALL·E demo benchmark set, kept separate from training data. | Clean, ready-to-use dataset folders |
| Yu Yang | **Augmentation / Transform Module** | Built all 6 transform functions to spec (JPEG compression, Gaussian blur, resize down→up, Gaussian noise, color jitter, center crop); built a folder-level batch wrapper; unit-tested every transform against the spec table; verified augmentations by visual inspection. | `augmentations.py`, `make_transformed_sets.py`, `transform_gallery.py` |
| Erika & Celine | **Model & Training Pipeline** | Selected and set up the EfficientNet-B0 backbone; created and adapted the training script; integrated the augmentation module for the robust-training runs; trained and tuned all four policy checkpoints. | Trained checkpoints + training logs and loss curves |
| Erika & Celine | **Evaluation & Robustness Analysis** | Built the clean eval script (accuracy, precision, recall, AUC); built the robustness eval script running the test set through every transform × severity; aggregated results into the Robustness Evaluation Summary; Developed the demo video| Robustness Evaluation Summary (Deliverable #4), Demo Video |
| Yi Jiun | **Error Analysis, Prediction Script & Packaging** | Built the error analysis script and wrote the Error Analysis Note; Built the prediction script end-to-end, and ran the COCO/DALL·E reference benchmark; Developed README and the Devpost writeup; Oversaw the execution of other deliverables and provided troubleshooting help | README, Devpost description, prediction script |

---

## 7. Acknowledgments

Built for **TikTok TechJam 2026, Track 5**. Uses the CIFAKE, SID_Set, and WildFake datasets, and a reference-only subset of COCO val2017 and DALL·E Advanced generations for demo purposes.

---

## Appendix — Full Per-Condition Breakdown

The summary tables in Step 4 (Mean Distortion Robustness, Accuracy Drop From Clean, Worst-Case Condition) condense the full 17-condition eval grid down to means and single worst-case rows. The full breakdown is provided here for reference.

### Accuracy by Condition (%)

| Condition | none | continuous | heldout | spec |
|---|---|---|---|---|
| clean | 98.5 | 98.5 | 98.4 | 98.2 |
| jpeg_q90 | 98.4 | 98.4 | 98.4 | 98.1 |
| jpeg_q70 | 98.5 | 98.5 | 98.1 | 98.2 |
| jpeg_q50 | 96.2 | 97.5 | 97.3 | 96.3 |
| jpeg_q30 | 92.7 | 96.6 | 96.6 | 96.1 |
| blur_s0.5 | 85.4 | 97.6 | 97.7 | 96.8 |
| blur_s1.0 | 60.9 | 96.5 | 96.1 | 95.6 |
| blur_s2.0 | 65.9 | 93.4 | 93.0 | 92.9 |
| resize_0.5 | 58.1 | 96.5 | 96.2 | 95.7 |
| resize_0.25 | 66.1 | 92.5 | 91.6 | 92.3 |
| noise_s0.02 | 97.2 | 98.1 | 97.9 | 98.0 |
| noise_s0.05 | 91.7 | 97.5 | 97.0 | 97.9 |
| noise_s0.1 | 63.3 | 96.1 | 95.9 | 95.3 |
| jitter_20 | 97.8 | 97.9 | 97.7 | 97.6 |
| crop_80 | 86.9 | 97.5 | 97.7 | 97.1 |
| chain_resize50_jpeg70 | 56.9 | 95.0 | 94.9 | 94.8 |
| chain_crop80_jpeg50 | 74.5 | 95.4 | 95.5 | 93.9 |
| **MEAN (distorted)** | **80.7** | **96.6** | **96.4** | **96.0** |

### ROC-AUC by Condition (%)

| Condition | none | continuous | heldout | spec |
|---|---|---|---|---|
| clean | 99.9 | 99.9 | 99.9 | 99.9 |
| jpeg_q90 | 99.9 | 99.9 | 99.9 | 99.8 |
| jpeg_q70 | 99.9 | 99.9 | 99.9 | 99.8 |
| jpeg_q50 | 99.6 | 99.7 | 99.7 | 99.6 |
| jpeg_q30 | 99.1 | 99.5 | 99.5 | 99.5 |
| blur_s0.5 | 99.6 | 99.8 | 99.8 | 99.7 |
| blur_s1.0 | 82.6 | 99.5 | 99.4 | 99.3 |
| blur_s2.0 | 73.5 | 98.4 | 98.2 | 98.1 |
| resize_0.5 | 87.2 | 99.5 | 99.4 | 99.3 |
| resize_0.25 | 73.2 | 97.9 | 97.5 | 97.8 |
| noise_s0.02 | 99.8 | 99.8 | 99.8 | 99.8 |
| noise_s0.05 | 97.8 | 99.7 | 99.7 | 99.7 |
| noise_s0.1 | 80.7 | 99.3 | 99.4 | 99.1 |
| jitter_20 | 99.8 | 99.8 | 99.8 | 99.8 |
| crop_80 | 99.4 | 99.7 | 99.7 | 99.7 |
| chain_resize50_jpeg70 | 88.6 | 99.3 | 99.2 | 99.0 |
| chain_crop80_jpeg50 | 96.6 | 99.2 | 99.2 | 98.9 |
| **MEAN (distorted)** | **92.3** | **99.4** | **99.4** | **99.3** |

### Accuracy Drop From Clean (%)

| Condition | none | continuous | heldout | spec |
|---|---|---|---|---|
| clean | 0.0 | 0.0 | 0.0 | 0.0 |
| jpeg_q90 | 0.1 | 0.1 | 0.0 | 0.1 |
| jpeg_q70 | 0.0 | 0.1 | 0.3 | -0.0 |
| jpeg_q50 | 2.3 | 1.1 | 1.1 | 1.2 |
| jpeg_q30 | 5.8 | 1.9 | 1.8 | 2.0 |
| blur_s0.5 | 13.1 | 1.0 | 0.7 | 1.4 |
| blur_s1.0 | 37.7 | 2.0 | 2.3 | 2.6 |
| blur_s2.0 | 32.6 | 5.1 | 5.4 | 5.3 |
| resize_0.5 | 40.4 | 2.0 | 2.1 | 2.5 |
| resize_0.25 | 32.4 | 6.0 | 6.8 | 5.9 |
| noise_s0.02 | 1.3 | 0.4 | 0.5 | 0.2 |
| noise_s0.05 | 6.8 | 1.0 | 1.4 | 0.8 |
| noise_s0.1 | 35.3 | 2.4 | 2.5 | 2.9 |
| jitter_20 | 0.7 | 0.6 | 0.7 | 0.5 |
| crop_80 | 11.6 | 1.0 | 0.7 | 1.1 |
| chain_resize50_jpeg70 | 41.6 | 3.5 | 3.5 | 3.4 |
| chain_crop80_jpeg50 | 24.0 | 3.1 | 2.8 | 4.2 |
| **MEAN** | **16.8** | **1.9** | **1.9** | **2.0** |
