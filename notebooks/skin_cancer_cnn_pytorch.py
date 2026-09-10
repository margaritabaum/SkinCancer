# %% [markdown]
# # Skin Lesion Classification with PyTorch CNNs
# ### ISIC 2019 + HAM10000, deduplicated | 8-class dermatoscopic classification
#
# This notebook builds three models in sequence, so you can see what each addition buys you:
#
# | # | Model | What it adds |
# |---|-------|--------------|
# | A | Baseline CNN (from scratch) | Reference point — plain conv stack, no regularization |
# | B | CNN + BatchNorm + Dropout + Augmentation | Regularization and data augmentation |
# | C | Fine-tuned EfficientNet (ImageNet pretrained) | Transfer learning — the high-accuracy model |
#
# Everything is PyTorch. Everything runs on a free Colab T4.
#
# ---
#
# ## Scope note — please read before quoting any accuracy number
#
# **1. ISIC 2019 already contains HAM10000.** The ISIC 2019 training set (25,331 images) is
# HAM10000 + BCN20000 + MSK combined. Concatenating the two datasets naively would duplicate
# ~10,000 images and scatter copies of the same image across train and test, inflating your
# accuracy by several points for no real reason. Section 2 deduplicates by ISIC image ID.
#
# **2. This merge does not give you skin tone diversity.** Both archives come from European and
# Australian dermatology clinics and are overwhelmingly Fitzpatrick I–III. Dermatoscopy also
# optically flattens the surrounding skin, so tone is weakly represented even where it exists.
# Neither dataset ships Fitzpatrick labels at all, so this notebook *cannot* report per-skin-tone
# performance. If you later want a defensible fairness claim, the datasets that carry darker skin
# are **Fitzpatrick17k** (~16k images, FST I–VI labels), **DDI** (656 biopsy-confirmed, FST V–VI
# enriched), and **PAD-UFES-20** (~2.3k smartphone images). Section 2.6 marks exactly where they
# would plug in. Note they are *clinical* photographs, not dermatoscopic — a real domain shift.
#
# **3. This is a coursework/research model, not a medical device.** Do not use it for diagnosis.

# %% [markdown]
# ---
# ## 0. Environment and configuration

# %%
# Colab has torch/torchvision preinstalled. We add timm (pretrained backbones) and kaggle.
!pip install -q timm==1.0.11 kaggle 2>&1 | tail -1
print("installs done")

# %%
import os
import re
import glob
import json
import time
import random
import shutil
import zipfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("whitegrid")

print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)


# %%
def set_seed(seed: int = 42):
    """Make runs reproducible. cudnn.benchmark=True is faster but slightly nondeterministic;
    we keep it on because the speedup matters more than bit-exact reproducibility here."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
GPU_MEM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
    if torch.cuda.is_available() else 0.0
)
print(f"device: {DEVICE} | {GPU_NAME} | {GPU_MEM_GB:.1f} GB")

if DEVICE.type == "cpu":
    print("\n!! No GPU detected. In Colab: Runtime > Change runtime type > T4 GPU.")


# %% [markdown]
# ### 0.1 Config
#
# You said you weren't sure which Colab tier you'd be on, so this auto-detects the GPU and picks
# a resolution / batch size / backbone that fits. Override anything by editing the dataclass
# defaults below — every knob in the notebook reads from this one object.

# %%
@dataclass
class Config:
    # --- data ---
    seed: int = 42
    num_classes: int = 8
    test_frac: float = 0.15          # group-aware, stratified
    val_frac: float = 0.15           # of the remainder

    # --- image / batch (auto-tuned in __post_init__) ---
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2

    # --- scratch CNN training ---
    scratch_epochs: int = 30
    scratch_lr: float = 1e-3

    # --- pretrained fine-tune ---
    backbone: str = "efficientnet_b0"
    ft_epochs: int = 20
    ft_head_lr: float = 1e-3
    ft_backbone_lr: float = 1e-4
    ft_weight_decay: float = 1e-4
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2

    # --- shared ---
    label_smoothing: float = 0.05
    use_class_weights: bool = True   # weighted loss for imbalance
    use_balanced_sampler: bool = False  # alternative to weighted loss; don't use both
    early_stop_patience: int = 7
    amp: bool = True

    # --- paths ---
    data_root: str = "/content/data"
    work_dir: str = "/content/work"
    drive_backup: str = ""           # e.g. "/content/drive/MyDrive/skin_cancer" to persist ckpts

    def __post_init__(self):
        """Scale to the GPU we actually got. T4 (16GB) is the free-tier default."""
        if GPU_MEM_GB >= 38:         # A100 40/80GB
            self.img_size, self.batch_size, self.backbone = 384, 48, "efficientnet_b3"
        elif GPU_MEM_GB >= 20:       # L4 24GB / V100 32GB
            self.img_size, self.batch_size, self.backbone = 300, 48, "efficientnet_b3"
        elif GPU_MEM_GB >= 14:       # T4 16GB  <-- free tier
            self.img_size, self.batch_size, self.backbone = 224, 64, "efficientnet_b0"
        else:                        # CPU or tiny GPU: keep it runnable, not good
            self.img_size, self.batch_size = 160, 16
            self.scratch_epochs, self.ft_epochs = 3, 3


CFG = Config()
Path(CFG.data_root).mkdir(parents=True, exist_ok=True)
Path(CFG.work_dir).mkdir(parents=True, exist_ok=True)

print(json.dumps(asdict(CFG), indent=2))


# %% [markdown]
# ### 0.2 (Optional) persist checkpoints to Google Drive
#
# Colab wipes local disk when the runtime dies. If you want trained weights to survive that,
# uncomment the two lines below and set `CFG.drive_backup`.

# %%
# from google.colab import drive
# drive.mount("/content/drive")
# CFG.drive_backup = "/content/drive/MyDrive/skin_cancer"

if CFG.drive_backup:
    Path(CFG.drive_backup).mkdir(parents=True, exist_ok=True)
    print("checkpoints will be mirrored to:", CFG.drive_backup)
else:
    print("checkpoints stay on local Colab disk only (lost on runtime restart)")


# %% [markdown]
# ---
# ## 1. Data acquisition (Kaggle API)
#
# **One-time setup:** kaggle.com → your profile → Settings → API → *Create New Token*.
# That downloads `kaggle.json`. Run the cell below and upload it.

# %%
from google.colab import files as colab_files  # noqa: E402

kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if not kaggle_json.exists():
    print("Upload your kaggle.json:")
    colab_files.upload()
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.move("kaggle.json", kaggle_json)
    kaggle_json.chmod(0o600)
print("kaggle credentials:", "found" if kaggle_json.exists() else "MISSING")


# %% [markdown]
# ### 1.1 Download
#
# Kaggle mirrors get renamed occasionally. If a slug 404s, search Kaggle for the dataset and
# paste the new `owner/dataset-name` into `DATASET_SLUGS`. Section 1.2 verifies what landed
# on disk regardless of how it got there, so a manual upload works too.

# %%
DATASET_SLUGS = {
    "ham10000": "kmader/skin-cancer-mnist-ham10000",
    "isic2019": "andrewmvd/isic-2019",
}


def download_kaggle(slug: str, dest: str) -> bool:
    dest_p = Path(dest)
    if dest_p.exists() and any(dest_p.iterdir()):
        print(f"[skip] {slug} already present at {dest}")
        return True
    dest_p.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {slug} -> {dest}")
    code = os.system(f"kaggle datasets download -d {slug} -p {dest} --unzip -q")
    ok = code == 0 and any(dest_p.iterdir())
    print(f"[{'ok  ' if ok else 'FAIL'}] {slug}")
    return ok


for key, slug in DATASET_SLUGS.items():
    download_kaggle(slug, f"{CFG.data_root}/{key}")


# %% [markdown]
# ### 1.2 Verify what actually landed on disk
#
# Rather than hardcoding directory layouts (which differ between mirrors), we index every
# `.jpg` recursively and locate the metadata CSVs by pattern.

# %%
def index_images(root: str) -> dict:
    """Map ISIC image id -> filepath, recursively. Ids look like ISIC_0024306."""
    idx = {}
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.png"):
        for p in Path(root).rglob(ext):
            idx.setdefault(p.stem, str(p))
    return idx


IMG_INDEX = index_images(CFG.data_root)
print(f"indexed {len(IMG_INDEX):,} image files under {CFG.data_root}")

csvs = sorted(Path(CFG.data_root).rglob("*.csv"))
print("\nCSVs found:")
for c in csvs:
    print("  ", c.relative_to(CFG.data_root), f"({c.stat().st_size/1e6:.1f} MB)")


# %% [markdown]
# ---
# ## 2. Building one clean dataset out of two overlapping ones
#
# ### 2.1 A shared label space
#
# HAM10000 uses 7 labels; ISIC 2019 uses 8. They line up except that ISIC splits out **SCC**
# (squamous cell carcinoma), which HAM folds into `akiec` ("actinic keratoses / intraepithelial
# carcinoma"). We adopt the ISIC 8-class space and map HAM's `akiec` to `AK`.
#
# This is a real, if minor, label-noise source: some HAM `akiec` images are biologically SCC.
# It is the standard compromise in the literature and worth one sentence in your write-up.

# %%
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CLASS_FULL_NAME = {
    "MEL":  "Melanoma",
    "NV":   "Melanocytic nevus",
    "BCC":  "Basal cell carcinoma",
    "AK":   "Actinic keratosis",
    "BKL":  "Benign keratosis",
    "DF":   "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC":  "Squamous cell carcinoma",
}

# Which classes are malignant — used later for the clinically meaningful metrics.
MALIGNANT = {"MEL", "BCC", "SCC"}

HAM_TO_UNIFIED = {
    "mel": "MEL", "nv": "NV", "bcc": "BCC", "akiec": "AK",
    "bkl": "BKL", "df": "DF", "vasc": "VASC",
}


# %% [markdown]
# ### 2.2 Load HAM10000 metadata
#
# HAM10000 gives us something ISIC 2019 does not: `lesion_id`. Several images can be the *same
# physical lesion* photographed more than once. If those copies straddle a train/test boundary
# the model can memorize rather than generalize — so we carry `lesion_id` through and split on it.

# %%
def find_csv(patterns) -> Path:
    """Locate a metadata CSV by filename pattern, tolerant of mirror layout differences."""
    for pat in patterns:
        hits = sorted(Path(CFG.data_root).rglob(pat))
        if hits:
            return max(hits, key=lambda p: p.stat().st_size)
    raise FileNotFoundError(f"none of {patterns} found under {CFG.data_root}")


ham_csv = find_csv(["HAM10000_metadata*", "*HAM10000*metadata*.csv"])
ham = pd.read_csv(ham_csv)
print("HAM10000 metadata:", ham_csv.name, ham.shape)
ham.head()

# %%
ham_df = pd.DataFrame({
    "image_id":  ham["image_id"].astype(str),
    "label":     ham["dx"].map(HAM_TO_UNIFIED),
    "lesion_id": ham["lesion_id"].astype(str),
    "age":       ham["age"],
    "sex":       ham["sex"],
    "site":      ham["localization"],
    "source":    "HAM10000",
})
assert ham_df["label"].notna().all(), "unmapped HAM label"
print(f"HAM10000: {len(ham_df):,} images, {ham_df.lesion_id.nunique():,} unique lesions")
print(f"  -> {len(ham_df) - ham_df.lesion_id.nunique():,} images are repeat shots of a lesion "
      "already in the set")


# %% [markdown]
# ### 2.3 Load ISIC 2019 ground truth
#
# The ground-truth CSV is one-hot: one column per class plus `UNK`. We take the argmax and
# drop `UNK` rows (unlabeled / out-of-distribution).

# %%
isic_csv = find_csv(["*Training_GroundTruth*.csv", "*ISIC_2019*GroundTruth*.csv"])
isic_raw = pd.read_csv(isic_csv)
print("ISIC 2019 ground truth:", isic_csv.name, isic_raw.shape)
print("columns:", list(isic_raw.columns))
isic_raw.head()

# %%
id_col = isic_raw.columns[0]                       # usually "image"
label_cols = [c for c in isic_raw.columns if c.upper() in CLASS_TO_IDX]
unk_col = [c for c in isic_raw.columns if c.upper() == "UNK"]

isic = isic_raw.copy()
if unk_col:
    n_unk = int(isic[unk_col[0]].sum())
    isic = isic[isic[unk_col[0]] != 1]
    print(f"dropped {n_unk:,} UNK rows")

isic_df = pd.DataFrame({
    "image_id": isic[id_col].astype(str).str.replace(r"\.jpe?g$", "", regex=True),
    "label":    isic[label_cols].to_numpy().argmax(axis=1),
    "source":   "ISIC2019",
})
isic_df["label"] = isic_df["label"].map(lambda i: label_cols[i].upper())
print(f"\nISIC 2019: {len(isic_df):,} labeled images")
isic_df["label"].value_counts()


# %% [markdown]
# ### 2.4 Deduplicate — the step that makes this merge honest
#
# HAM10000 image ids *are* ISIC archive ids (`ISIC_0024306`), so the overlap is found by exact
# id match. We keep one row per image, preferring the ISIC 2019 label (it has the finer
# AK/SCC distinction) while inheriting HAM's `lesion_id` and patient metadata where available.

# %%
overlap = set(ham_df.image_id) & set(isic_df.image_id)
print(f"HAM10000 images:            {len(ham_df):>7,}")
print(f"ISIC 2019 images:           {len(isic_df):>7,}")
print(f"Present in BOTH:            {len(overlap):>7,}  <-- would have been duplicated")
print(f"HAM images NOT in ISIC:     {len(set(ham_df.image_id) - overlap):>7,}")
print(f"Naive concat would give:    {len(ham_df) + len(isic_df):>7,} rows")
print(f"Correct union is:           "
      f"{len(set(ham_df.image_id) | set(isic_df.image_id)):>7,} rows")

# %%
ham_meta = ham_df.drop(columns=["label", "source"])

# Start from ISIC (finer labels), attach HAM metadata where the image is shared.
merged = isic_df.merge(ham_meta, on="image_id", how="left")

# Add any HAM-only images ISIC 2019 does not carry.
ham_only = ham_df[~ham_df.image_id.isin(set(isic_df.image_id))].copy()
merged = pd.concat([merged, ham_only], ignore_index=True)

# Images with no lesion_id (ISIC-only rows) are their own group of one.
merged["lesion_id"] = merged["lesion_id"].fillna(merged["image_id"])

# Label disagreements between the two sources, for the record.
cmp = ham_df.merge(isic_df, on="image_id", suffixes=("_ham", "_isic"))
disagree = cmp[cmp.label_ham != cmp.label_isic]
print(f"label disagreements on shared images: {len(disagree):,} "
      f"({100*len(disagree)/max(len(cmp),1):.2f}%)  [ISIC label kept]")
if len(disagree):
    print(disagree.groupby(["label_ham", "label_isic"]).size()
                  .sort_values(ascending=False).head(8))

assert merged.image_id.is_unique, "deduplication failed - duplicate image_id remains"
print(f"\nmerged dataset: {len(merged):,} unique images, "
      f"{merged.lesion_id.nunique():,} unique lesions")


# %% [markdown]
# ### 2.5 Attach file paths and drop rows whose image is missing

# %%
merged["path"] = merged["image_id"].map(IMG_INDEX)
missing = merged["path"].isna().sum()
if missing:
    print(f"WARNING: {missing:,} rows have no image file on disk - dropping them.")
    print("If this number is large, a download probably failed. Re-run section 1.")
merged = merged[merged["path"].notna()].reset_index(drop=True)
merged["target"] = merged["label"].map(CLASS_TO_IDX)

print(f"final dataset: {len(merged):,} images across {merged.label.nunique()} classes")
merged.head()


# %% [markdown]
# ### 2.6 Hook: adding a skin-tone-diverse dataset later
#
# If you decide to pursue the fairness angle, this is the one place you'd change. Load
# Fitzpatrick17k / DDI / PAD-UFES-20 into a frame with the same columns
# (`image_id, label, lesion_id, path, source, target`), add an `fst` column with the
# Fitzpatrick score, and concat here. Then stratify the split on `(label, fst)` and report
# per-FST metrics in section 9.
#
# Do not skip the domain-shift question: those are clinical photographs, not dermatoscopy.
# The standard handling is either a separate held-out evaluation set (cleanest) or a
# domain-adaptation term in the loss. Naively mixing them and reporting one aggregate accuracy
# would hide the very disparity you set out to measure.

# %%
# --- placeholder: no additional datasets loaded in this run ---
merged["fst"] = np.nan   # Fitzpatrick skin type, unavailable for ISIC/HAM
print("skin tone labels available for "
      f"{merged.fst.notna().sum()} / {len(merged)} images "
      "(expected: 0 - neither source provides them)")


# %% [markdown]
# ### 2.7 Exploratory data analysis

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

counts = merged["label"].value_counts()
sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette="Paired")
for c in axes[0].containers:
    axes[0].bar_label(c, fmt="%d")
axes[0].set(title="Class distribution (merged, deduplicated)",
            xlabel="Class", ylabel="Images")

src = merged.groupby(["label", "source"]).size().unstack(fill_value=0)
src.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
axes[1].set(title="Contribution by source", xlabel="Class", ylabel="Images")

sns.histplot(merged["age"].dropna(), bins=30, ax=axes[2], color="indianred")
axes[2].set(title="Age distribution (where known)", xlabel="Age")

plt.tight_layout()
plt.show()

imb = counts.max() / counts.min()
print(f"Imbalance ratio (largest / smallest class): {imb:.1f}x")
print("\nThis is severe. Accuracy alone will be misleading — a model that predicts NV for")
print("every image already scores ~50%. Section 9 reports balanced accuracy and macro-F1,")
print("which are the numbers to actually judge this on.")

# %%
# Look at real examples — always eyeball your data before training on it.
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, cls in zip(axes.ravel(), CLASSES):
    row = merged[merged.label == cls].sample(1, random_state=CFG.seed).iloc[0]
    ax.imshow(Image.open(row["path"]).convert("RGB"))
    ax.set_title(f"{cls} — {CLASS_FULL_NAME[cls]}\n(n={counts[cls]:,})", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 3. Splitting the data without leaking
#
# Two rules, both of which the example notebook you shared gets wrong:
#
# 1. **Group by lesion.** All images of one physical lesion go to the same split. A plain
#    `train_test_split` puts near-identical photos on both sides and inflates test accuracy.
# 2. **Stratify by class.** DF and VASC have only a few hundred images; a random split can
#    leave the test set with too few to measure anything.
#
# `StratifiedGroupKFold` does both at once.

# %%
def grouped_split(df, frac, seed):
    """Carve `frac` off `df`, keeping lesion groups intact and classes stratified."""
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(df, df["target"], groups=df["lesion_id"]))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


trainval_df, test_df = grouped_split(merged, CFG.test_frac, CFG.seed)
train_df, val_df = grouped_split(trainval_df, CFG.val_frac, CFG.seed)

for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:>5}: {len(d):>6,} images | {d.lesion_id.nunique():>6,} lesions "
          f"| {100*len(d)/len(merged):4.1f}%")

# The assertion that matters: no lesion appears in two splits.
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    da, db = {"train": train_df, "val": val_df, "test": test_df}[a], \
             {"train": train_df, "val": val_df, "test": test_df}[b]
    shared = set(da.lesion_id) & set(db.lesion_id)
    assert not shared, f"LEAK: {len(shared)} lesions shared between {a} and {b}"
print("\nno lesion-level leakage between any pair of splits")

# %%
dist = pd.DataFrame({
    "train": train_df.label.value_counts(),
    "val":   val_df.label.value_counts(),
    "test":  test_df.label.value_counts(),
}).reindex(CLASSES).fillna(0).astype(int)
dist["train %"] = (100 * dist["train"] / dist["train"].sum()).round(1)
dist["test %"] = (100 * dist["test"] / dist["test"].sum()).round(1)
dist


# %% [markdown]
# ---
# ## 4. Dataset, transforms, dataloaders

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SkinLesionDataset(Dataset):
    """Reads JPEGs from disk on demand. The full merged set is ~25k images at ~600x450,
    far too large to hold in RAM as an array (the way the reference notebook does it)."""

    def __init__(self, df: pd.DataFrame, tfms):
        self.paths = df["path"].tolist()
        self.targets = df["target"].to_numpy()
        self.tfms = tfms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfms(img), int(self.targets[i])


def build_transforms(img_size: int, train: bool, augment: bool = True):
    """Eval transform is deterministic. Train transform is either the same (model A, so the
    augmentation comparison is clean) or the full augmentation stack (models B and C)."""
    if not train or not augment:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # Lesions have no canonical orientation, so flips and full rotations are all label-preserving.
    # Colour jitter is kept mild: hue is diagnostically meaningful in dermatoscopy and shifting
    # it aggressively teaches the model to ignore a real signal.
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def build_loaders(cfg: Config, augment: bool = True):
    train_ds = SkinLesionDataset(train_df, build_transforms(cfg.img_size, True, augment))
    val_ds = SkinLesionDataset(val_df, build_transforms(cfg.img_size, False))
    test_ds = SkinLesionDataset(test_df, build_transforms(cfg.img_size, False))

    if cfg.use_balanced_sampler:
        # Oversample rare classes. Alternative to weighted loss — using both double-counts.
        cls_count = np.bincount(train_df["target"], minlength=cfg.num_classes)
        w = (1.0 / cls_count)[train_df["target"].to_numpy()]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    common = dict(num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    return (
        DataLoader(train_ds, cfg.batch_size, shuffle=shuffle, sampler=sampler,
                   drop_last=True, **common),
        DataLoader(val_ds, cfg.batch_size * 2, shuffle=False, **common),
        DataLoader(test_ds, cfg.batch_size * 2, shuffle=False, **common),
    )


# %%
# Sanity-check the augmentation pipeline visually before committing GPU hours to it.
def denorm(t):
    x = t.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(x, 0, 1)


sample_path = train_df.iloc[0]["path"]
aug = build_transforms(CFG.img_size, train=True, augment=True)
img = Image.open(sample_path).convert("RGB")

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
axes[0].imshow(img.resize((CFG.img_size, CFG.img_size)))
axes[0].set_title("original")
axes[0].axis("off")
for ax in axes[1:]:
    ax.imshow(denorm(aug(img)))
    ax.set_title("augmented")
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 5. Training engine
#
# One reusable trainer for all three models: mixed precision, cosine LR schedule with warmup,
# class-weighted loss, early stopping on **balanced accuracy** (not raw accuracy — with a 50%
# majority class, raw accuracy would happily select a model that ignores the rare classes).

# %%
def make_grad_scaler(enabled: bool):
    """torch.amp.GradScaler is the current API; older torch builds only have
    torch.cuda.amp.GradScaler. Colab is usually new enough, but this keeps the notebook
    runnable if you pin an older torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_criterion(cfg: Config):
    if cfg.use_class_weights and not cfg.use_balanced_sampler:
        counts = np.bincount(train_df["target"], minlength=cfg.num_classes)
        # Inverse-frequency weights, normalized to mean 1 so the loss scale stays comparable.
        w = counts.sum() / (cfg.num_classes * np.maximum(counts, 1))
        w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)
        print("class weights:", {c: round(float(w[i]), 2) for i, c in enumerate(CLASSES)})
    else:
        w = None
    return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)


def cosine_warmup(optimizer, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def predict(model, loader, tta: bool = False):
    """Returns (probs, targets). TTA averages predictions over the 4 flip variants —
    free accuracy, at 4x inference cost."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=CFG.amp and DEVICE.type == "cuda"):
            p = model(x).softmax(1)
            if tta:
                for dims in ([3], [2], [2, 3]):
                    p = p + model(torch.flip(x, dims)).softmax(1)
                p = p / 4
        probs.append(p.float().cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def fit(model, cfg: Config, epochs, optimizer, loaders, tag: str):
    """Trains, early-stops on val balanced accuracy, restores the best weights, returns history."""
    train_loader, val_loader, _ = loaders
    criterion = make_criterion(cfg)
    scaler = make_grad_scaler(enabled=cfg.amp and DEVICE.type == "cuda")
    total_steps = epochs * len(train_loader)
    scheduler = cosine_warmup(optimizer, warmup_steps=len(train_loader), total_steps=total_steps)

    ckpt = Path(cfg.work_dir) / f"{tag}.pt"
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bacc": [], "lr": []}
    best_bacc, best_epoch, since_improved = -1.0, -1, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.amp and DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        probs, ys = predict(model, val_loader)
        preds = probs.argmax(1)
        val_loss = float(F.cross_entropy(torch.log(torch.tensor(probs) + 1e-9),
                                         torch.tensor(ys)))
        acc = accuracy_score(ys, preds)
        bacc = balanced_accuracy_score(ys, preds)

        hist["train_loss"].append(running / seen)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(acc)
        hist["val_bacc"].append(bacc)
        hist["lr"].append(scheduler.get_last_lr()[0])

        flag = ""
        if bacc > best_bacc:
            best_bacc, best_epoch, since_improved = bacc, epoch, 0
            torch.save(model.state_dict(), ckpt)
            flag = "  <- best"
        else:
            since_improved += 1

        print(f"[{tag}] epoch {epoch+1:>2}/{epochs} | "
              f"train_loss {running/seen:.4f} | val_loss {val_loss:.4f} | "
              f"acc {acc:.4f} | balanced_acc {bacc:.4f}{flag}")

        if since_improved >= cfg.early_stop_patience:
            print(f"[{tag}] early stop - no improvement for {cfg.early_stop_patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f} min | "
          f"best val balanced_acc {best_bacc:.4f} @ epoch {best_epoch+1}")

    if cfg.drive_backup:
        shutil.copy(ckpt, Path(cfg.drive_backup) / ckpt.name)
    return hist


def plot_history(histories: dict):
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{name} train")
        axes[0].plot(h["val_loss"], "--", label=f"{name} val")
        axes[1].plot(h["val_acc"], label=name)
        axes[2].plot(h["val_bacc"], label=name)
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].set(title="Validation accuracy", xlabel="epoch")
    axes[2].set(title="Validation balanced accuracy", xlabel="epoch")
    for a in axes:
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## 6. Model A — baseline CNN from scratch
#
# A plain VGG-style stack: no BatchNorm, no Dropout, no augmentation. This is deliberately
# under-regularized. Its job is to be the number the next two models beat, and to show you
# what overfitting looks like on this data (train loss falls, val loss turns upward).

# %%
class ScratchCNN(nn.Module):
    """Configurable from-scratch CNN. `regularized=False` gives the plain baseline;
    `regularized=True` adds BatchNorm and Dropout for model B."""

    def __init__(self, num_classes=8, widths=(32, 64, 128, 256), regularized=False,
                 dropout=0.4):
        super().__init__()
        blocks, in_ch = [], 3
        for w in widths:
            layer = [nn.Conv2d(in_ch, w, 3, padding=1, bias=not regularized)]
            if regularized:
                layer.append(nn.BatchNorm2d(w))
            layer.append(nn.ReLU(inplace=True))

            layer.append(nn.Conv2d(w, w, 3, padding=1, bias=not regularized))
            if regularized:
                layer.append(nn.BatchNorm2d(w))
            layer.append(nn.ReLU(inplace=True))

            layer.append(nn.MaxPool2d(2))
            if regularized:
                layer.append(nn.Dropout2d(0.1))
            blocks.append(nn.Sequential(*layer))
            in_ch = w

        self.features = nn.Sequential(*blocks)
        # Global average pooling instead of Flatten+Dense: far fewer parameters and much
        # less prone to overfitting than the 1024-unit dense layers in the reference notebook.
        self.pool = nn.AdaptiveAvgPool2d(1)
        head = [nn.Flatten()]
        if regularized:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(in_ch, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# %%
set_seed(CFG.seed)
loaders_noaug = build_loaders(CFG, augment=False)

model_a = ScratchCNN(CFG.num_classes, regularized=False).to(DEVICE)
print(f"Model A parameters: {count_params(model_a):,}")

opt_a = torch.optim.AdamW(model_a.parameters(), lr=CFG.scratch_lr, weight_decay=1e-5)
hist_a = fit(model_a, CFG, CFG.scratch_epochs, opt_a, loaders_noaug, tag="A_baseline_cnn")


# %% [markdown]
# ---
# ## 7. Model B — scratch CNN + BatchNorm + Dropout + augmentation
#
# Same depth and width as A. The only differences are regularization and the augmented
# training transform, so any gap between A and B is attributable to those two changes.

# %%
set_seed(CFG.seed)
loaders_aug = build_loaders(CFG, augment=True)

model_b = ScratchCNN(CFG.num_classes, regularized=True, dropout=0.4).to(DEVICE)
print(f"Model B parameters: {count_params(model_b):,}")

opt_b = torch.optim.AdamW(model_b.parameters(), lr=CFG.scratch_lr, weight_decay=1e-4)
hist_b = fit(model_b, CFG, CFG.scratch_epochs, opt_b, loaders_aug, tag="B_regularized_cnn")


# %% [markdown]
# ---
# ## 8. Model C — fine-tuned EfficientNet (the accuracy model)
#
# Two things make this the strongest model:
#
# - **ImageNet pretraining.** Edge, texture and colour-blob detectors transfer directly to
#   dermatoscopy. 25k images is nowhere near enough to learn them from scratch.
# - **Discriminative learning rates.** The pretrained backbone gets a 10x smaller LR than the
#   randomly-initialized head, so early high-LR steps don't destroy the transferred features.
#
# `drop_path_rate` (stochastic depth) is the regularizer that matters most when fine-tuning
# a large model on a small dataset.

# %%
import timm  # noqa: E402

set_seed(CFG.seed)
model_c = timm.create_model(
    CFG.backbone,
    pretrained=True,
    num_classes=CFG.num_classes,
    drop_rate=CFG.drop_rate,
    drop_path_rate=CFG.drop_path_rate,
).to(DEVICE)

classifier_name = model_c.default_cfg.get("classifier", "classifier")
head_params, backbone_params = [], []
for n, p in model_c.named_parameters():
    (head_params if n.startswith(classifier_name) else backbone_params).append(p)

print(f"Model C: {CFG.backbone} | {count_params(model_c):,} params "
      f"({len(head_params)} head tensors, {len(backbone_params)} backbone tensors)")

opt_c = torch.optim.AdamW([
    {"params": backbone_params, "lr": CFG.ft_backbone_lr},
    {"params": head_params, "lr": CFG.ft_head_lr},
], weight_decay=CFG.ft_weight_decay)

hist_c = fit(model_c, CFG, CFG.ft_epochs, opt_c, loaders_aug, tag="C_efficientnet_finetuned")

# %%
plot_history({"A baseline": hist_a, "B regularized": hist_b, "C pretrained": hist_c})


# %% [markdown]
# ---
# ## 9. Evaluation
#
# ### What to report, and why not accuracy
#
# NV alone is ~50% of this dataset, so "always predict NV" scores ~50% accuracy while being
# clinically worthless. The metrics below are the ones that survive that objection:
#
# - **Balanced accuracy** — mean per-class recall. The constant predictor scores 12.5%.
# - **Macro-F1** — equal weight per class regardless of size.
# - **Melanoma recall** — a missed melanoma is the costly error in this problem. This is the
#   single number a clinician would ask for.
# - **Malignant-vs-benign sensitivity/specificity** — collapses the 8 classes to the triage
#   decision that actually gets made.

# %%
def evaluate(model, loader, name, tta=False, plot=True):
    probs, ys = predict(model, loader, tta=tta)
    preds = probs.argmax(1)

    res = {
        "model": name,
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro"),
        "weighted_f1": f1_score(ys, preds, average="weighted"),
    }
    try:
        res["macro_auc"] = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
    except ValueError:
        res["macro_auc"] = float("nan")

    # Melanoma recall — the error that matters most.
    mel = CLASS_TO_IDX["MEL"]
    res["melanoma_recall"] = float(((preds == mel) & (ys == mel)).sum() / max((ys == mel).sum(), 1))

    # Collapse to the malignant / benign triage decision.
    mal_idx = [CLASS_TO_IDX[c] for c in MALIGNANT]
    y_mal = np.isin(ys, mal_idx)
    p_mal = np.isin(preds, mal_idx)
    tp = int((y_mal & p_mal).sum()); fn = int((y_mal & ~p_mal).sum())
    tn = int((~y_mal & ~p_mal).sum()); fp = int((~y_mal & p_mal).sum())
    res["malignant_sensitivity"] = tp / max(tp + fn, 1)
    res["malignant_specificity"] = tn / max(tn + fp, 1)
    res["malignant_missed"] = fn

    if plot:
        print(f"\n{'='*70}\n{name}{'  (+TTA)' if tta else ''}\n{'='*70}")
        for k, v in res.items():
            if k != "model":
                print(f"  {k:<24}: {v:.4f}" if isinstance(v, float) else f"  {k:<24}: {v}")

        print("\n" + classification_report(ys, preds, target_names=CLASSES, digits=3,
                                           zero_division=0))

        cm = confusion_matrix(ys, preds)
        cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, axes = plt.subplots(1, 2, figsize=(19, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                    yticklabels=CLASSES, ax=axes[0], cbar=False)
        axes[0].set(title=f"{name} — confusion matrix (counts)",
                    xlabel="Predicted", ylabel="True")
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES,
                    yticklabels=CLASSES, ax=axes[1], vmin=0, vmax=1, cbar=False)
        axes[1].set(title=f"{name} — normalized by true class (diagonal = recall)",
                    xlabel="Predicted", ylabel="True")
        plt.tight_layout()
        plt.show()

    return res, probs, ys


# %%
test_loader = loaders_aug[2]

res_a, _, _ = evaluate(model_a, test_loader, "A — baseline CNN")
res_b, _, _ = evaluate(model_b, test_loader, "B — regularized CNN + augmentation")
res_c, probs_c, ys_c = evaluate(model_c, test_loader, f"C — {CFG.backbone} fine-tuned")

# %% [markdown]
# ### 9.1 Test-time augmentation
#
# Averaging predictions over the four flips typically adds a fraction of a point of balanced
# accuracy for no extra training. It costs 4x at inference, which is irrelevant here.

# %%
res_c_tta, probs_tta, _ = evaluate(model_c, test_loader,
                                   f"C — {CFG.backbone} + TTA", tta=True)

# %%
summary = pd.DataFrame([res_a, res_b, res_c, res_c_tta]).set_index("model").round(4)
display_cols = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc",
                "melanoma_recall", "malignant_sensitivity", "malignant_specificity",
                "malignant_missed"]
print("\nFINAL COMPARISON (held-out test set, no lesion overlap with training)\n")
summary[display_cols]

# %%
fig, ax = plt.subplots(figsize=(12, 5))
summary[["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall"]].plot(
    kind="bar", ax=ax, colormap="Set2", rot=12)
ax.set(title="Model comparison on the held-out test set", ylabel="Score", ylim=(0, 1))
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 9.2 Trading specificity for melanoma sensitivity
#
# `argmax` treats every error as equally bad. Clinically it is not: missing a melanoma is far
# worse than over-referring a benign nevus. Lowering the decision threshold on the melanoma
# probability buys recall at the cost of false positives. This curve lets you pick that
# operating point deliberately rather than accepting whatever `argmax` happens to give.

# %%
mel_idx = CLASS_TO_IDX["MEL"]
y_is_mel = (ys_c == mel_idx)
rows = []
for thr in np.arange(0.05, 0.85, 0.05):
    pred_mel = probs_tta[:, mel_idx] >= thr
    tp = int((pred_mel & y_is_mel).sum()); fp = int((pred_mel & ~y_is_mel).sum())
    fn = int((~pred_mel & y_is_mel).sum()); tn = int((~pred_mel & ~y_is_mel).sum())
    rows.append({
        "threshold": round(thr, 2),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "melanomas_missed": fn,
    })
thr_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(thr_df.threshold, thr_df.sensitivity, "o-", label="Melanoma sensitivity (recall)")
ax.plot(thr_df.threshold, thr_df.specificity, "s-", label="Specificity")
ax.plot(thr_df.threshold, thr_df.precision, "^-", label="Precision")
ax.axhline(0.95, ls="--", c="crimson", alpha=0.6, label="95% sensitivity target")
ax.set(xlabel="Melanoma probability threshold", ylabel="Score",
       title="Choosing the melanoma operating point")
ax.legend()
plt.tight_layout()
plt.show()

thr_df


# %% [markdown]
# ### 9.3 Grad-CAM — is it looking at the lesion?
#
# A model can reach high accuracy for the wrong reason: dermatoscopic datasets are famous for
# containing surgical skin markings, rulers and ink that correlate with malignancy because
# suspicious lesions get marked before excision. Grad-CAM shows which pixels drove the
# prediction. Heat centred on the lesion is good; heat on a ruler or a marker line means your
# accuracy number is measuring an artifact.

# %%
def grad_cam(model, img_tensor, target_class=None):
    """Grad-CAM on the last feature map. Implemented with hooks — no extra dependency."""
    model.eval()
    feats, grads = {}, {}

    target_layer = (model.features[-1] if isinstance(model, ScratchCNN)
                    else model.conv_head if hasattr(model, "conv_head")
                    else list(model.children())[-4])

    # These hooks must return None. A forward hook that returns a value replaces the layer
    # output, and a backward hook that returns a value replaces grad_input — which raises
    # "hook has changed the size of value". Hence statements, not lambdas.
    def save_feats(module, inp, out):
        feats["v"] = out

    def save_grads(module, grad_in, grad_out):
        grads["v"] = grad_out[0]

    h1 = target_layer.register_forward_hook(save_feats)
    h2 = target_layer.register_full_backward_hook(save_grads)

    x = img_tensor.unsqueeze(0).to(DEVICE)
    logits = model(x)
    cls = int(logits.argmax(1)) if target_class is None else target_class
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    h1.remove(); h2.remove()

    w = grads["v"].mean(dim=(2, 3), keepdim=True)          # channel importance
    cam = F.relu((w * feats["v"]).sum(1, keepdim=True))     # weighted sum, keep positives
    cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    return cam, cls, logits.softmax(1)[0, cls].item()


eval_tfms = build_transforms(CFG.img_size, train=False)
picks = test_df.groupby("label", group_keys=False).apply(
    lambda g: g.sample(1, random_state=CFG.seed)).head(8)

fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
for j, (_, row) in enumerate(picks.iterrows()):
    img = Image.open(row["path"]).convert("RGB")
    t = eval_tfms(img)
    cam, cls, conf = grad_cam(model_c, t)

    axes[0, j].imshow(denorm(t))
    axes[0, j].set_title(f"true: {row['label']}", fontsize=10)
    axes[0, j].axis("off")

    axes[1, j].imshow(denorm(t))
    axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
    ok = "OK" if CLASSES[cls] == row["label"] else "WRONG"
    axes[1, j].set_title(f"pred: {CLASSES[cls]} ({conf:.2f}) {ok}", fontsize=10,
                         color="green" if ok == "OK" else "crimson")
    axes[1, j].axis("off")
plt.suptitle("Grad-CAM on the fine-tuned model — check the heat sits on the lesion", y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 10. Export
#
# Saves weights, the label mapping, the config and the test metrics together, so the artifact
# is self-describing when you come back to it.

# %%
export_dir = Path(CFG.work_dir) / "export"
export_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    "state_dict": model_c.state_dict(),
    "backbone": CFG.backbone,
    "classes": CLASSES,
    "img_size": CFG.img_size,
    "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "config": asdict(CFG),
    "test_metrics": {k: v for k, v in res_c_tta.items() if k != "model"},
}, export_dir / "skin_lesion_model.pt")

summary.to_csv(export_dir / "test_metrics.csv")
merged[["image_id", "label", "lesion_id", "source"]].to_csv(
    export_dir / "merged_dataset_manifest.csv", index=False)

if CFG.drive_backup:
    shutil.copytree(export_dir, Path(CFG.drive_backup) / "export", dirs_exist_ok=True)

print("exported to", export_dir)
for f in sorted(export_dir.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")


# %% [markdown]
# ---
# ## 11. Where the remaining accuracy is
#
# In rough order of return on effort:
#
# 1. **A bigger backbone at higher resolution.** `efficientnet_b3` at 300px, or `convnext_tiny`
#    / `swin_tiny` at 224px. Change `CFG.backbone` and `CFG.img_size`. Needs more than a T4.
# 2. **5-fold cross-validation with an ensemble.** Reliably worth 1–2 points of balanced
#    accuracy, and it also gives you error bars — which matter more than the headline number
#    when your rare classes have only a few hundred test images. Reuse `StratifiedGroupKFold`
#    from section 3 with all folds instead of just the first.
# 3. **Longer schedules with MixUp / CutMix.** Helps most on the rare classes.
# 4. **Metadata fusion.** Age, sex and body site are genuinely predictive and already in
#    `merged`. Concatenate them to the pooled CNN features before the classifier head.
# 5. **Progressive resizing.** Train at 224, fine-tune the last few epochs at 384.
#
# ### On the claim you can make
#
# The model this notebook produces is trained and tested on dermatoscopic images from
# predominantly light-skinned European and Australian populations. Reporting its accuracy as
# general skin cancer classification performance would overstate it. You can say the merge is
# correctly deduplicated and lesion-grouped, and that the numbers are honest *for this
# distribution* — that is a defensible claim, and it is the one the data supports. Section 2.6
# is where you would extend it if you want to say more.
