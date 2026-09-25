# HAM10000 — from-scratch CNN, three configurations

A controlled progression on a 7-class HAM10000 dermatoscopic classifier: the original
baseline, a tuned version, and a version that adds patient metadata and seed ensembling.
All three use the same data, the same lesion-grouped split and the same held-out test set,
so every difference in the metrics traces to a named change rather than to a different
split.

> **Not a medical device.** This is coursework. It is not validated for clinical use and
> must not be used to make decisions about a real lesion.

## Results

Test set: 1,010 images from 742 held-out lesions.

| Metric | Baseline | Tuned | + metadata & ensemble |
|---|---|---|---|
| Accuracy | 0.592 | 0.717 | **0.788** |
| Balanced accuracy | 0.643 | 0.629 | 0.642 |
| Macro F1 | 0.461 | 0.584 | **0.621** |
| Weighted F1 | 0.636 | 0.738 | **0.796** |
| Cohen's kappa | 0.404 | 0.529 | **0.621** |
| Macro ROC-AUC | 0.893 | 0.939 | **0.953** |
| Malignant sensitivity | 0.823 | **0.847** | 0.764 |
| Malignant specificity | 0.677 | 0.786 | **0.872** |
| Melanomas missed as benign | 27 / 116 | **25 / 116** | 29 / 116 |

The third column is three seeds ensembled with test-time augmentation, using patient
metadata. Per-lesion pooling is **not** applied there — see the caveats below.

Cohen's kappa is the number to read. It corrects for chance agreement, so 0.404 → 0.621
across the three configurations shows the model learned signal rather than learning to
answer "nevus" more often.

**Balanced accuracy barely moved** — 0.643 → 0.629 → 0.642 — while accuracy climbed nearly
20 points. That gap is the honest story of this project: most of the gain is on the
majority class and on precision, not on rare-class recall. Dermatofibroma has 93 training
images and no amount of reweighting conjures signal from that. If the argument is "detect
rare malignancies", lead with balanced accuracy and the gain is small; if it is overall
classification quality, lead with accuracy and kappa.

## What changed, and why

| # | Change | Effect |
|---|---|---|
| 1 | Input 71×71 → 128×128 | Source images are 600×450. 71×71 discards ~98% of the pixels, and with them the pigment networks, dots and streaks the diagnosis rests on. Largest single win. |
| 2 | 4 conv blocks → 5; 256 → 512 channels | A 128px input needs one more downsampling stage to reach the same final feature-map size. 1.18M → 4.72M parameters. |
| 3 | Class weights: 53× spread → 7.6× | Raw inverse frequency gave `nv` weight 0.05 against `df`'s 2.67, so a nevus mistake was nearly free. The model sprayed rare-class guesses: 156 of 669 test nevi came back as melanoma. Taking the square root keeps the ordering and compresses the spread. |
| 4 | Select on macro F1, not balanced accuracy | Balanced accuracy is the mean of per-class *recall* and contains no notion of precision, so it rewards a model that buys recall with false positives. |
| 5 | 30 → 60 epochs, weight decay 1e-4 → 5e-4 | The baseline never converged — validation score was still rising at epoch 30 and the patience-7 early stop never fired. |
| 6 | `RandomResizedCrop` + `RandomErasing` added | Scale variation, and less reliance on any single image region. |
| 7 | Test-time augmentation | Average the prediction over all four flips; a mole has no canonical orientation. +0.3 accuracy points. |
| 8 | Per-lesion pooling | HAM10000 photographs many lesions more than once. Worth +2.1 accuracy points on a single model — but it slightly *hurts* an ensemble (see below). |
| 9 | Patient metadata branch | `age`, `sex` and `localization` sit unused in the CSV and are genuinely predictive. A small MLP encodes them, concatenated with the image features before the classifier. |
| 10 | Three-seed ensemble | Averaged probabilities from seeds 1-3, which also quantifies run-to-run noise. |

Deliberately unchanged: the lesion-grouped 80/10/10 split, `random_state=28`, the label
mapping, cross-entropy loss and AdamW. The lesion-grouped split in particular is the thing
most HAM10000 projects get wrong, and it is why these numbers are trustworthy.

## Per-class, final ensemble

| Lesion | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Melanocytic nevi | 0.932 | 0.867 | 0.899 | 669 |
| Melanoma | 0.475 | 0.733 | 0.576 | 116 |
| Benign keratosis | 0.609 | 0.505 | 0.552 | 111 |
| Basal cell carcinoma | 0.603 | 0.636 | 0.619 | 55 |
| Actinic keratoses | 0.621 | 0.562 | 0.590 | 32 |
| Vascular lesions | 0.615 | 0.571 | 0.593 | 14 |
| Dermatofibroma | 0.412 | 0.538 | 0.467 | 13 |

## Caveats worth stating in any write-up

**Per-lesion pooling reverses sign.** +2.1 points on a single model, −0.7 on the ensemble
(0.788 → 0.781). An ensemble already smooths its predictions; pooling on top over-smooths.

**Seed variance is large.** The three seeds scored 0.759, 0.766 and 0.795 individually — a
3.6-point spread, and seed 3 alone beat the ensemble. Quoting seed 3 would be
cherry-picking. By the same logic, the single-seed 0.717 may have been an unlucky draw.

**Metadata and ensembling are confounded.** The per-seed mean (0.773) against the earlier
single run (0.717) suggests roughly +5 from metadata and +1.5 from ensembling, but epochs
also moved 60 → 50 and the images were pre-cached by box downsampling. A clean attribution
needs a metadata-off run at otherwise identical settings.

**Sensitivity regressed.** Malignant sensitivity fell 0.847 → 0.764 and melanomas called
benign rose 25 → 29, bought with a large specificity gain. For a screening tool that is
the wrong direction, and it is the one metric that got worse.

## Files

| File | Purpose |
|---|---|
| `model.py` | The CNN, written with every layer named in `__init__` and the order spelled out in `forward` |
| `prep_71px.py` | Decodes HAM10000 at 71×71 and builds the lesion-grouped split (baseline) |
| `prep_256px.py` | Decodes at 256×256 and records per-image lesion codes for pooling |
| `train_baseline.py` | Trains the original notebook architecture; reproduces the 0.592 row |
| `train_improved.py` | Trains the tuned model; reproduces the 0.717 row |
| `model_meta.py` | The CNN with the added patient-metadata branch |
| `train_metadata_ensemble.py` | Trains three seeds with metadata and ensembles them; reproduces the 0.788 row |

Raw run logs and metric dumps are in [`../results/`](../results/).

## Reproducing

Point the `read_csv` path at your `HAM10000_metadata.csv`, put all 10,015 JPEGs in an
`images/` folder beside the scripts, then:

```bash
python prep_256px.py                # decode once, cached as x256.npy
python train_improved.py            # single tuned model -> the 0.717 row
python train_metadata_ensemble.py   # 3 seeds with metadata -> the 0.788 row
```

`train_metadata_ensemble.py` expects a 128px cache (`x128.npy`); build it once from
`x256.npy` with an exact 2×2 box downsample. Training the images at their final size
rather than resizing every epoch cut the run from 84 s/epoch to 16 s/epoch.

Runs on CUDA, MPS or CPU — the device is selected automatically.

## One bug worth knowing about

Computing channel normalization statistics as `array.mean(axis=(0,1,2))` on a **float32**
array of ~52M values silently returns the wrong answer: once the running sum passes the
24-bit mantissa limit (~16.7M), adding another 0.7 changes nothing, and every channel
converges to the same wrong value. Keep the data `uint8` (NumPy then accumulates in
float64) or pass `dtype=np.float64` explicitly. `train_improved.py` asserts the channels
differ, so this cannot pass silently again.
