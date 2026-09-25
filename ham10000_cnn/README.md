# HAM10000 — from-scratch CNN, baseline vs. tuned

A controlled before/after on a 7-class HAM10000 dermatoscopic classifier. Both models use
the same data, the same lesion-grouped split and the same seed, so every difference in the
metrics traces to a named change rather than to a different test set.

> **Not a medical device.** This is coursework. It is not validated for clinical use and
> must not be used to make decisions about a real lesion.

## Results

Test set: 1,010 images from 742 held-out lesions.

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| Accuracy | 0.592 | **0.717** | +12.5 |
| Balanced accuracy | 0.643 | 0.629 | −1.4 |
| Macro F1 | 0.461 | **0.584** | +12.3 |
| Weighted F1 | 0.636 | **0.738** | +10.2 |
| Cohen's kappa | 0.404 | **0.529** | +12.5 |
| Macro ROC-AUC | 0.893 | **0.939** | +4.6 |
| Malignant sensitivity | 0.823 | **0.847** | +2.4 |
| Malignant specificity | 0.677 | **0.786** | +10.9 |
| Melanomas missed as benign | 27 / 116 | **25 / 116** | −2 |

Cohen's kappa is the number to read. It corrects for chance agreement, so 0.404 → 0.529
shows the model learned signal rather than learning to answer "nevus" more often.

Balanced accuracy fell 1.4 points. That is the deliberate, predicted cost of softening the
class weights: majority-class precision is bought with a little rare-class recall. If the
question is "detect rare malignancies", balanced accuracy is the metric to lead with and
this trade is arguable. If the question is overall classification quality, macro F1 and
kappa both moved decisively.

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
| 8 | Per-lesion pooling | HAM10000 photographs many lesions more than once. Averaging a lesion's photos into one prediction is +2.1 accuracy points and matches how a clinician actually works. |

Deliberately unchanged: the lesion-grouped 80/10/10 split, `random_state=28`, the label
mapping, cross-entropy loss and AdamW. The lesion-grouped split in particular is the thing
most HAM10000 projects get wrong, and it is why these numbers are trustworthy.

## Per-class, tuned model

| Lesion | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Melanocytic nevi | 0.941 | 0.765 | 0.844 | 669 |
| Melanoma | 0.362 | 0.690 | 0.475 | 116 |
| Benign keratosis | 0.561 | 0.495 | 0.526 | 111 |
| Basal cell carcinoma | 0.562 | 0.745 | 0.641 | 55 |
| Actinic keratoses | 0.431 | 0.688 | 0.530 | 32 |
| Vascular lesions | 0.667 | 0.714 | 0.690 | 14 |
| Dermatofibroma | 0.500 | 0.308 | 0.381 | 13 |

Melanoma precision of 0.362 is the main weakness that remains: roughly two in three
melanoma calls are false alarms. For a screening tool that errs the safe way, but it is
the obvious target for further work.

## Files

| File | Purpose |
|---|---|
| `model.py` | The CNN, written with every layer named in `__init__` and the order spelled out in `forward` |
| `prep_71px.py` | Decodes HAM10000 at 71×71 and builds the lesion-grouped split (baseline) |
| `prep_256px.py` | Decodes at 256×256 and records per-image lesion codes for pooling |
| `train_baseline.py` | Trains the original notebook architecture; reproduces the 0.592 row |
| `train_improved.py` | Trains the tuned model; reproduces the 0.717 row |

Raw run logs and metric dumps are in [`../results/`](../results/).

## Reproducing

Point the `read_csv` path at your `HAM10000_metadata.csv`, put all 10,015 JPEGs in an
`images/` folder beside the scripts, then:

```bash
python prep_256px.py       # decode once, cached as x256.npy
python train_improved.py   # ~40 min on an M-series GPU (MPS), longer on CPU
```

Runs on CUDA, MPS or CPU — the device is selected automatically.

## One bug worth knowing about

Computing channel normalization statistics as `array.mean(axis=(0,1,2))` on a **float32**
array of ~52M values silently returns the wrong answer: once the running sum passes the
24-bit mantissa limit (~16.7M), adding another 0.7 changes nothing, and every channel
converges to the same wrong value. Keep the data `uint8` (NumPy then accumulates in
float64) or pass `dtype=np.float64` explicitly. `train_improved.py` asserts the channels
differ, so this cannot pass silently again.
