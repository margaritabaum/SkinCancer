#!/usr/bin/env python3
"""Run the whole notebook end-to-end on a synthetic dataset, on CPU, in ~1 minute.

Catches broken code before you spend Colab GPU hours on it. Builds fake HAM10000 / ISIC 2019
metadata (with a deliberate overlap between them) and tiny JPEGs, then executes every code
cell in order with the Kaggle/Colab-only cells stubbed out.

    python tools/smoke_test.py
"""
import json
import re
import sys
import types
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "notebooks" / "skin_cancer_cnn_pytorch.ipynb"

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
HAM_DX = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

# Cells that need Kaggle credentials, a Colab runtime, or a network download.
SKIP_MARKERS = [
    "!pip install",
    "from google.colab import files",
    "for key, slug in DATASET_SLUGS.items()",
]


def build_fake_data(root: Path, n_ham=240, n_isic=300, n_overlap=180):
    """HAM ids overlap ISIC ids, mirroring the real relationship between the two datasets."""
    rng = np.random.default_rng(0)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ham_ids = [f"ISIC_{i:07d}" for i in range(n_ham)]
    isic_ids = ham_ids[:n_overlap] + [f"ISIC_{i:07d}" for i in range(9000, 9000 + n_isic - n_overlap)]

    for iid in set(ham_ids) | set(isic_ids):
        arr = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{iid}.jpg", quality=60)

    # HAM: several images can share a lesion_id — the thing the split must respect.
    lesion_ids = [f"HAM_{i // 2:05d}" for i in range(n_ham)]
    pd.DataFrame({
        "lesion_id": lesion_ids,
        "image_id": ham_ids,
        "dx": [HAM_DX[i % len(HAM_DX)] for i in range(n_ham)],
        "dx_type": "histo",
        "age": rng.integers(20, 85, n_ham).astype(float),
        "sex": rng.choice(["male", "female"], n_ham),
        "localization": rng.choice(["back", "face", "trunk", "arm"], n_ham),
    }).to_csv(root / "HAM10000_metadata.csv", index=False)

    # ISIC: one-hot ground truth over 8 classes + UNK.
    onehot = np.zeros((len(isic_ids), len(CLASSES)), dtype=int)
    onehot[np.arange(len(isic_ids)), np.arange(len(isic_ids)) % len(CLASSES)] = 1
    gt = pd.DataFrame(onehot, columns=CLASSES)
    gt.insert(0, "image", isic_ids)
    gt["UNK"] = 0
    # Mark UNK on ISIC-only images so they are genuinely unrecoverable. (An UNK image that
    # also appears in HAM10000 should be kept with the HAM label - that is correct behaviour,
    # so it would not test the drop path.)
    gt.loc[gt.index[-5:], "UNK"] = 1
    gt.to_csv(root / "ISIC_2019_Training_GroundTruth.csv", index=False)

    n_unique = len(set(ham_ids) | set(isic_ids))
    return n_unique, n_unique - 5      # (files on disk, rows expected after merge)


def fake_timm():
    """Stand-in for timm so the fine-tuning path runs without downloading ImageNet weights."""
    import torch.nn as nn

    class TinyBackbone(nn.Module):
        default_cfg = {"classifier": "classifier"}

        def __init__(self, num_classes):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            )
            self.conv_head = nn.Sequential(nn.Conv2d(32, 64, 1), nn.ReLU())
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            return self.classifier(self.pool(self.conv_head(self.stem(x))).flatten(1))

    m = types.ModuleType("timm")
    m.__version__ = "stub"
    m.create_model = lambda name, **kw: TinyBackbone(kw.get("num_classes", 8))
    return m


def main():
    tmp = Path(tempfile.mkdtemp(prefix="skin_smoke_"))
    try:
        data_root = tmp / "data"
        n_unique, n_expected = build_fake_data(data_root)
        print(f"synthetic dataset: {n_unique} unique images in {data_root}")

        sys.modules["timm"] = fake_timm()

        ns = {"__name__": "__main__"}
        cells = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]

        for i, cell in enumerate(cells):
            src = "".join(cell["source"])
            if any(mk in src for mk in SKIP_MARKERS):
                print(f"  cell {i:>2}: SKIPPED (needs Colab/Kaggle)")
                continue

            # Shrink the run to CPU size right after Config is defined.
            if "CFG = Config()" in src:
                src = src.replace("CFG = Config()", (
                    "CFG = Config()\n"
                    f"CFG.data_root = {str(data_root)!r}\n"
                    f"CFG.work_dir = {str(tmp / 'work')!r}\n"
                    "CFG.img_size, CFG.batch_size, CFG.num_workers = 32, 8, 0\n"
                    "CFG.scratch_epochs, CFG.ft_epochs, CFG.early_stop_patience = 2, 2, 5\n"
                    "CFG.test_frac, CFG.val_frac = 0.2, 0.2"
                ))
            src = re.sub(r"^\s*[!%].*$", "pass", src, flags=re.M)

            try:
                exec(compile(src, f"<cell {i}>", "exec"), ns)
                print(f"  cell {i:>2}: ok")
            except Exception as e:
                print(f"\nFAILED at code cell {i}:\n{'-'*60}\n{src[:700]}\n{'-'*60}")
                import traceback
                traceback.print_exc()
                return 1

        # Post-conditions the notebook is supposed to guarantee.
        merged, train_df, val_df, test_df = (ns["merged"], ns["train_df"],
                                             ns["val_df"], ns["test_df"])
        assert merged["image_id"].is_unique, "dedup failed"
        assert len(merged) == n_expected, f"expected {n_expected} rows, got {len(merged)}"
        for a, b in [(train_df, val_df), (train_df, test_df), (val_df, test_df)]:
            assert not (set(a.lesion_id) & set(b.lesion_id)), "lesion leak between splits"
        assert (REPO and (Path(ns["CFG"].work_dir) / "export" / "skin_lesion_model.pt").exists()), \
            "export missing"

        print("\nPASS - dedup, lesion-grouped splits, all 3 models, evaluation, "
              "Grad-CAM and export all ran clean.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
