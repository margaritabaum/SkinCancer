# Skin Lesion Classification — PyTorch CNNs on ISIC 2019 + HAM10000

8-class dermatoscopic skin lesion classification, built for Google Colab. Three models in
sequence so the effect of each change is visible: a from-scratch CNN, the same CNN with
regularization and augmentation, and a fine-tuned ImageNet backbone.

**→ [`notebooks/skin_cancer_cnn_pytorch.ipynb`](notebooks/skin_cancer_cnn_pytorch.ipynb)**

## Quick start

1. Open the notebook in Google Colab (`File > Upload notebook`, or push this repo to GitHub
   and use Colab's GitHub tab).
2. `Runtime > Change runtime type > T4 GPU`.
3. Get a Kaggle API token: kaggle.com → Settings → API → *Create New Token*. The notebook
   prompts you to upload the resulting `kaggle.json`.
4. `Runtime > Run all`. Expect roughly 1.5–3 hours on a free T4 for all three models.

Configuration lives in one `Config` dataclass in section 0.1. It auto-detects the GPU and
picks the image size, batch size and backbone that fit — 224px / EfficientNet-B0 on a T4,
300–384px / EfficientNet-B3 on an L4 or A100.

## The models

| # | Model | Purpose |
|---|-------|---------|
| A | Baseline CNN, no regularization, no augmentation | Reference point; shows what overfitting looks like here |
| B | Same CNN + BatchNorm + Dropout + augmentation | Isolates the effect of regularization |
| C | Fine-tuned EfficientNet (ImageNet pretrained) | The accuracy model — discriminative LRs, stochastic depth |

## Two things this notebook does that most tutorials get wrong

**ISIC 2019 already contains HAM10000.** The ISIC 2019 training set (25,331 images) is
HAM10000 + BCN20000 + MSK. Concatenating both datasets duplicates ~10,000 images and scatters
copies of the same image across train and test. Section 2.4 deduplicates by ISIC image ID and
reports the overlap explicitly.

**One lesion can appear as several images.** HAM10000's `lesion_id` groups repeat photographs
of the same physical lesion. A plain `train_test_split` puts near-identical images on both
sides of the split. Section 3 uses `StratifiedGroupKFold` to keep lesion groups intact while
stratifying by class, and asserts that no lesion crosses a split boundary.

Both of these inflate reported accuracy if ignored. The numbers this notebook produces will
look *lower* than many published tutorials, and that is the point.

## Metrics

NV is ~50% of the dataset, so a constant "always predict NV" classifier scores ~50% accuracy.
The notebook reports balanced accuracy, macro-F1, macro AUC, melanoma recall, and
malignant-vs-benign sensitivity/specificity. Section 9.2 sweeps the melanoma decision
threshold, because a missed melanoma and an over-referred nevus are not equally costly.
Section 9.3 runs Grad-CAM to check the model is looking at the lesion rather than at surgical
ink or a ruler — a well-documented shortcut in dermatoscopic datasets.

## On skin tone

This dataset pairing does **not** provide skin tone diversity. ISIC and HAM10000 come from
European and Australian clinics, are overwhelmingly Fitzpatrick I–III, and ship no Fitzpatrick
labels at all — so per-skin-tone performance cannot be measured here, and no fairness claim
should be made from these results.

The datasets that do carry darker skin are **Fitzpatrick17k** (~16k images with FST I–VI
labels), **DDI** (656 biopsy-confirmed images, FST V–VI enriched), and **PAD-UFES-20** (~2.3k
smartphone images). They are *clinical* photographs, not dermatoscopy, so mixing them in is a
genuine domain shift that needs handling — a separate held-out evaluation set is the cleanest
approach. Section 2.6 of the notebook marks exactly where they plug in.

## Development

The notebook's source of truth is the percent-format Python file, which diffs cleanly in git.
Edit it, then regenerate the `.ipynb`:

```bash
python tools/py_to_ipynb.py notebooks/skin_cancer_cnn_pytorch.py
```

Validate the whole pipeline on synthetic data in about a minute, on CPU, before spending Colab
GPU time on it:

```bash
python tools/smoke_test.py
```

It fabricates HAM/ISIC metadata with a deliberate overlap, runs every code cell with the
Kaggle- and Colab-only cells stubbed, and asserts that deduplication and lesion-grouped
splitting hold.

## Layout

```
notebooks/skin_cancer_cnn_pytorch.py      source of truth (percent format)
notebooks/skin_cancer_cnn_pytorch.ipynb   generated — this is the Colab notebook
tools/py_to_ipynb.py                      .py -> .ipynb converter
tools/smoke_test.py                       end-to-end synthetic-data test
```

## Disclaimer

Research and coursework code. Not a medical device, not validated for clinical use, and not
suitable for diagnosis.
