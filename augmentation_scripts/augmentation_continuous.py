import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, roc_auc_score

# Define the local directory where the dataset will be stored and unzipped
DATASET_DIR = '/content/dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

# List of zip file names to process from GOOGLE_DRIVE_DATASET_PATH
ZIP_FILE_NAMES = [
    'sid_clean.zip',
    'cifake_clean.zip',
    'wildfake_clean.zip'
]

# Define TRAIN_ROOT and VAL_ROOT based on the local dataset directory
TRAIN_ROOT = os.path.join(DATASET_DIR, "train")
VAL_ROOT = os.path.join(DATASET_DIR, "val")
TEST_ROOT = os.path.join(DATASET_DIR, "test")

print(f"TRAIN_ROOT set to: {TRAIN_ROOT}")
print(f"VAL_ROOT set to: {VAL_ROOT}")
print(f"TEST_ROOT set to: {TEST_ROOT}")

import zipfile

print(f"Starting dataset extraction to: {DATASET_DIR}")

for zip_file_name in ZIP_FILE_NAMES:
    # Construct the full path to the zip file in Google Drive
    google_drive_zip_path = os.path.join(GOOGLE_DRIVE_DATASET_PATH, zip_file_name)

    # Define a temporary path for the zip file in the Colab environment
    temp_zip_path = os.path.join('/content', zip_file_name)

    print(f"Copying {zip_file_name} from Google Drive...")
    # Copy the zip file from Google Drive to the Colab environment
    !cp "{google_drive_zip_path}" "{temp_zip_path}"
    print(f"Copied {zip_file_name} to {temp_zip_path}")

    print(f"Unzipping {zip_file_name}...")
    # Unzip the file into the DATASET_DIR
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print(f"Unzipped {zip_file_name} to {DATASET_DIR}")

    # Remove the temporary zip file to save space
    os.remove(temp_zip_path)
    print(f"Removed temporary zip file: {temp_zip_path}")

print("All datasets extracted successfully!")

import shutil

# Define the target consolidated directories
CONSOLIDATED_TRAIN_ROOT = os.path.join(DATASET_DIR, "train")
CONSOLIDATED_VAL_ROOT = os.path.join(DATASET_DIR, "val")
CONSOLIDATED_TEST_ROOT = os.path.join(DATASET_DIR, "test")

split_roots = {
    'train': CONSOLIDATED_TRAIN_ROOT,
    'val': CONSOLIDATED_VAL_ROOT,
    'test': CONSOLIDATED_TEST_ROOT,
}

# Create the consolidated root directories
for root in split_roots.values():
    os.makedirs(os.path.join(root, "real"), exist_ok=True)
    os.makedirs(os.path.join(root, "ai"), exist_ok=True)

# List of unzipped dataset base directories
UNZIPPED_DATASET_DIRS = [
    os.path.join(DATASET_DIR, 'sid_clean'),
    os.path.join(DATASET_DIR, 'cifake_dataset_clean'),
    os.path.join(DATASET_DIR, 'wildfake_clean')
]

print("Consolidating datasets...")

total_moved = 0
total_errors = 0

for dataset_base_dir in UNZIPPED_DATASET_DIRS:
    dataset_name = os.path.basename(dataset_base_dir)
    print(f"\nProcessing dataset: {dataset_base_dir}")

    if not os.path.exists(dataset_base_dir):
        print(f"  ERROR: dataset directory does not exist, skipping entirely: {dataset_base_dir}")
        total_errors += 1
        continue

    for split in ['train', 'val', 'test']:
        source_split_dir = os.path.join(dataset_base_dir, split)

        if not os.path.exists(source_split_dir):
            print(f"  WARNING: no '{split}' split found for {dataset_name}, skipping this split.")
            continue

        target_real_dir = os.path.join(split_roots[split], 'real')
        target_ai_dir = os.path.join(split_roots[split], 'ai')

        #real
        source_real_dir = os.path.join(source_split_dir, 'real')
        if os.path.exists(source_real_dir):
            files = os.listdir(source_real_dir)
            if len(files) == 0:
                print(f"  WARNING: {source_real_dir} exists but is empty.")
            moved = 0
            for filename in files:
                src_path = os.path.join(source_real_dir, filename)
                if not os.path.isfile(src_path):
                    continue
                new_name = f"{dataset_name}_{filename}"
                dst_path = os.path.join(target_real_dir, new_name)
                if os.path.exists(dst_path):
                    print(f"    ERROR: collision, skipping: {new_name}")
                    total_errors += 1
                    continue
                try:
                    shutil.move(src_path, dst_path)
                    moved += 1
                except Exception as e:
                    print(f"    ERROR moving {src_path}: {e}")
                    total_errors += 1
            print(f"  Moved {moved} real images: {source_real_dir} -> {target_real_dir}")
            total_moved += moved
        else:
            print(f"  WARNING: no 'real' folder in {dataset_name}/{split}.")

        #fake
        source_fake_dir = os.path.join(source_split_dir, 'fake')
        if os.path.exists(source_fake_dir):
            files = os.listdir(source_fake_dir)
            if len(files) == 0:
                print(f"  WARNING: {source_fake_dir} exists but is empty.")
            moved = 0
            for filename in files:
                src_path = os.path.join(source_fake_dir, filename)
                if not os.path.isfile(src_path):
                    continue
                new_name = f"{dataset_name}_{filename}"
                dst_path = os.path.join(target_ai_dir, new_name)
                if os.path.exists(dst_path):
                    print(f"    ERROR: collision, skipping: {new_name}")
                    total_errors += 1
                    continue
                try:
                    shutil.move(src_path, dst_path)
                    moved += 1
                except Exception as e:
                    print(f"    ERROR moving {src_path}: {e}")
                    total_errors += 1
            print(f"  Moved {moved} fake images: {source_fake_dir} -> {target_ai_dir} (as ai)")
            total_moved += moved
        else:
            print(f"  WARNING: no 'fake' folder in {dataset_name}/{split}.")

print(f"\nTotal images moved: {total_moved}")
print(f"Total errors: {total_errors}")

if total_errors > 0:
    print("\n!! Errors occurred during consolidation. Review the log above before deleting source folders. !!")

print("Dataset consolidation complete!")

# Re-run the ROOT definitions after consolidation
TRAIN_ROOT = CONSOLIDATED_TRAIN_ROOT
VAL_ROOT = CONSOLIDATED_VAL_ROOT
TEST_ROOT = CONSOLIDATED_TEST_ROOT
print(f"TRAIN_ROOT re-set to: {TRAIN_ROOT}")
print(f"VAL_ROOT re-set to: {VAL_ROOT}")
print(f"TEST_ROOT re-set to: {TEST_ROOT}")

# For verification purposes 
for split in ['train', 'val', 'test']:
    for cls in ['real', 'ai']:
        path = os.path.join(DATASET_DIR, split, cls)
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"{split}/{cls}: {count} files")
        else:
            print(f"{split}/{cls}: MISSING")

# Checking device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Model configuration
BACKBONE = "efficientnet_b0"
POLICY = "continuous"

IMAGE_SIZE = 224
BATCH_SIZE = 32

EPOCHS = 6

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

# Continuous augmentation ranges
CONTINUOUS_RANGES = {
    "jpeg":   [30, 95],
    "blur":   [0.3, 2.2],
    "resize": [0.20, 1.00],
    "noise":  [0.01, 0.12],
    "jitter": [0.00, 0.25],
    "crop":   [0.70, 1.00],
}

# Apply augmentation to about 90% of training images
AUGMENTATION_PROB = 0.90

# Choose 0, 1, or 2 augmentation families
N_OPS = (0, 2)

# Checkpoint
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    f"{BACKBONE}_{POLICY}_best.pth"
)

print("Policy:", POLICY)
print("Checkpoint:", CHECKPOINT_PATH)

# Loading and creating training augmentation policies
# Connect to Repo for Augmentation functions
REPO_PATH = "/content/AI-Image-Detection/src/aigcdet"

if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from augmentations import (
    build_train_augment,
    apply_eval_transform,
    eval_names,
    get_eval_transform,
    canonicalize
)

print("augmentations.py imported successfully!")

# Build training augmenter
train_augmenter = build_train_augment(
    policy="continuous",
    n_ops=N_OPS,
    p=AUGMENTATION_PROB,
    seed=SEED
)

train_augmenter.ranges = CONTINUOUS_RANGES

augmentation_config = train_augmenter.config()

print("Continuous augmentation configuration:")
print(augmentation_config)

# Dataset
class AIImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        augmentation=None
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation

        self.samples = []

        class_mapping = {
            "real": 0,
            "ai": 1
        }

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp"
        )

        for class_name, label in class_mapping.items():

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.exists(class_dir):
                raise FileNotFoundError(
                    f"Cannot find folder: {class_dir}"
                )

            for filename in os.listdir(class_dir):

                if filename.lower().endswith(valid_extensions):

                    filepath = os.path.join(
                        class_dir,
                        filename
                    )

                    self.samples.append(
                        (filepath, label)
                    )

        print(
            f"Loaded {len(self.samples)} images "
            f"from {root_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filepath, label = self.samples[idx]

        image = Image.open(
            filepath
        ).convert("RGB")

        if self.augmentation is not None:
            image = self.augmentation(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# Data loading 
train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = AIImageDataset(
    root_dir=TRAIN_ROOT,
    transform=train_transform,
    augmentation=train_augmenter
)

val_dataset = AIImageDataset(
    root_dir=VAL_ROOT,
    transform=val_transform,
    augmentation=None
)

print()
print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Example labels:", labels[:10])

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

print(model.classifier)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    1
)

model = model.to(device)

print(model.classifier)

# Loss + Optimisation

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    probabilities = []
    ground_truth = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        # BCEWithLogitsLoss needs FLOAT labels
        labels = labels.float().to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        # Forward
        logits = model(images).squeeze(1)

        # Loss
        loss = criterion(
            logits,
            labels
        )

        # Backprop
        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probs = torch.sigmoid(logits)

        probabilities.extend(
            probs.detach().cpu().numpy()
        )

        ground_truth.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        total_loss / len(loader.dataset)
    )

    probabilities = np.array(probabilities)
    ground_truth = np.array(ground_truth)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        ground_truth,
        predictions
    )

    auc = roc_auc_score(
        ground_truth,
        probabilities
    )

    return (
        epoch_loss,
        accuracy,
        auc
    )

# Validating clean 
def evaluate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    probabilities = []
    ground_truth = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.float().to(
                device,
                non_blocking=True
            )

            logits = model(
                images
            ).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            ground_truth.extend(
                labels.cpu().numpy()
            )

    loss = (
        total_loss
        /
        len(loader.dataset)
    )

    probabilities = np.array(
        probabilities
    )

    ground_truth = np.array(
        ground_truth
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        ground_truth,
        predictions
    )

    auc = roc_auc_score(
        ground_truth,
        probabilities
    )

    return (
        loss,
        accuracy,
        auc
    )

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "train_auc": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_auc": []
}

best_val_accuracy = -1.0
best_epoch = -1

for epoch in range(EPOCHS):

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # TRAIN
    (
        train_loss,
        train_accuracy,
        train_auc
    ) = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # CLEAN VALIDATION
    (
        val_loss,
        val_accuracy,
        val_auc
    ) = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    # SAVE HISTORY

    history["epoch"].append(epoch + 1)

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["train_auc"].append(
        train_auc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["val_auc"].append(
        val_auc
    )

    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  {train_accuracy:.4f}"
    )

    print(
        f"Train AUC:       {train_auc:.4f}"
    )

    print()

    print(
        f"Clean Val Loss:     {val_loss:.4f}"
    )

    print(
        f"Clean Val Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Clean Val AUC:      {val_auc:.4f}"
    )

    # save best chechkpoint
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        checkpoint = {

            "epoch":
                best_epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_accuracy":
                val_accuracy,

            "val_auc":
                val_auc,

            "policy":
                POLICY,

            "augmentation_config":
                train_augmenter.config(),

            "backbone":
                BACKBONE,

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "seed":
                SEED
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

        print()
        print("✓ New best checkpoint saved")
        print(
            f"Best validation AUC: {val_auc:.4f}"
        )

    scheduler.step()

# Testing continuous augmentation
import matplotlib.pyplot as plt

# Get one raw image
sample_path, sample_label = train_dataset.samples[0]

raw_image = Image.open(
    sample_path
).convert("RGB")

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

for i, ax in enumerate(axes.flat):

    augmented_image = train_augmenter(
        raw_image.copy()
    )

    ax.imshow(
        augmented_image
    )

    ax.set_title(
        f"Augmented sample {i + 1}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

print()
print("=" * 50)

print("TRAINING COMPLETE")

print(
    "Policy:",
    POLICY
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best clean accuracy:",
    best_val_accuracy
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

# Plotting training history
history_df = pd.DataFrame(history)

history_df

# Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Train Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    f"Loss — EfficientNet-B0 ({POLICY})"
)

plt.legend()
plt.show()


## Training Accuracy

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    label="Clean Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    f"Accuracy — EfficientNet-B0 ({POLICY})"
)

plt.legend()
plt.show()


## AUC
plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_auc"],
    label="Train AUC"
)

plt.plot(
    history_df["epoch"],
    history_df["val_auc"],
    label="Clean Validation AUC"
)

plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")

plt.title(
    f"AUC — EfficientNet-B0 ({POLICY})"
)

plt.legend()
plt.show()

# Reloading the best training epoch
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

print(
    "Loaded checkpoint:"
)

print(
    "Policy:",
    checkpoint["policy"]
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Clean val accuracy:",
    checkpoint["val_accuracy"]
)

print(
    "Clean val AUC:",
    checkpoint["val_auc"]
)

# Verify
val_loss, val_accuracy, val_auc = evaluate_model(
    model,
    val_loader,
    criterion,
    device
)

print(
    f"Reloaded checkpoint accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Reloaded checkpoint AUC: "
    f"{val_auc:.4f}"
)
