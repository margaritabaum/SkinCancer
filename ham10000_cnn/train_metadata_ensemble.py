"""CNN + patient metadata, trained over several seeds and ensembled.

Adds to train_improved.py:
  - age / sex / localization fed through a metadata branch
  - N seeds trained independently, their probabilities averaged
  - test-time augmentation and per-lesion pooling, as before

Run: python train3.py
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, f1_score, cohen_kappa_score)
from model_meta import CNNWithMetadata

SP = os.path.dirname(os.path.abspath(__file__))
RES = 128
EPOCHS = 50
BATCH_SIZE = 64
SEEDS = [1, 2, 3]

names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions ',
         'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions',
         'Dermatofibroma']

METADATA = os.environ.get("HAM10000_METADATA",
                          os.path.join(SP, "HAM10000_metadata.csv"))


def build_metadata(idx_train):
    """age / sex / localization as a float matrix, in the same row order as x128.

    The category vocabulary is fixed from the whole file (that is only column
    layout, not label information), but age is standardized on TRAIN rows only.
    """
    df = pd.read_csv(METADATA).dropna(subset=["age"]).reset_index(drop=True)

    age = df["age"].values.astype(np.float32)
    age_mean, age_std = age[idx_train].mean(), age[idx_train].std()
    age_z = ((age - age_mean) / age_std).reshape(-1, 1)

    sex = pd.get_dummies(df["sex"], prefix="sex")
    loc = pd.get_dummies(df["localization"], prefix="loc")

    meta = np.hstack([age_z, sex.values.astype(np.float32),
                      loc.values.astype(np.float32)]).astype(np.float32)
    cols = ["age_z"] + list(sex.columns) + list(loc.columns)
    return meta, cols


class SkinDataset(Dataset):
    def __init__(self, x, meta, idx, y, tf):
        self.x, self.meta, self.idx, self.y, self.tf = x, meta, idx, y, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = torch.from_numpy(np.array(self.x[j])).permute(2, 0, 1).float().div(255)
        return self.tf(img), torch.from_numpy(self.meta[j]), int(self.y[j])


def main():
    x = np.load(os.path.join(SP, "x128.npy"), mmap_mode="r")
    s = np.load(os.path.join(SP, "split256.npz"))
    y, is_train, is_val, is_test = s["y"], s["is_train"], s["is_val"], s["is_test"]
    lesion_codes = s["lesion_codes"]

    idx_train = np.where(is_train)[0]
    idx_val = np.where(is_val)[0]
    idx_test = np.where(is_test)[0]

    meta, meta_cols = build_metadata(idx_train)
    print(f"metadata features: {meta.shape[1]} ({', '.join(meta_cols[:5])}, ...)", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "| resolution:", RES, "| seeds:", SEEDS, flush=True)

    # uint8 data + float64 accumulator: a float32 mean over 50M values saturates
    sample = np.asarray(x[idx_train[::10]])
    MEAN = (sample.mean(axis=(0, 1, 2), dtype=np.float64) / 255).astype(np.float32)
    STD = (sample.std(axis=(0, 1, 2), dtype=np.float64) / 255).astype(np.float32)
    del sample
    assert MEAN.std() > 1e-3, f"channels should differ, got {MEAN}"
    print("channel means:", MEAN.round(3), "| stds:", STD.round(3))

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(RES, scale=(0.6, 1.0), ratio=(0.85, 1.18), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    eval_tf = transforms.Normalize(MEAN, STD)

    def loader(idx, tf, shuffle=False, drop_last=False):
        return DataLoader(SkinDataset(x, meta, idx, y, tf), batch_size=BATCH_SIZE,
                          shuffle=shuffle, drop_last=drop_last, num_workers=4,
                          persistent_workers=True)

    train_loader = loader(idx_train, train_tf, shuffle=True, drop_last=True)
    val_loader = loader(idx_val, eval_tf)
    test_loader = loader(idx_test, eval_tf)

    counts = np.bincount(y[idx_train], minlength=7)
    weights = np.sqrt(len(idx_train) / (7 * np.maximum(counts, 1)))
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=0.05)

    @torch.no_grad()
    def predict(model, loader, tta=False):
        model.eval()
        probs, trues = [], []
        for xb, mb, yb in loader:
            xb, mb = xb.to(device), mb.to(device)
            p = model(xb, mb).softmax(1)
            if tta:
                for flip in (lambda t: t.flip(-1), lambda t: t.flip(-2),
                             lambda t: t.flip(-1).flip(-2)):
                    p = p + model(flip(xb), mb).softmax(1)
                p = p / 4
            probs.append(p.cpu())
            trues.append(yb)
        return torch.cat(probs).numpy(), torch.cat(trues).numpy()

    seed_test_probs, seed_summaries = [], []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CNNWithMetadata(num_meta=meta.shape[1], num_classes=7).to(device)
        if seed == SEEDS[0]:
            print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        best_f1, best_state, patience, since_best = -1.0, None, 12, 0

        for epoch in range(EPOCHS):
            model.train()
            running, seen = 0.0, 0
            for xb, mb, yb in train_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb, mb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * yb.size(0)
                seen += yb.size(0)
            scheduler.step()

            vp, vt = predict(model, val_loader)
            vpred = vp.argmax(1)
            f1 = f1_score(vt, vpred, average="macro")
            flag = ""
            if f1 > best_f1:
                best_f1, since_best = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                flag = "  <- best"
            else:
                since_best += 1
            print(f"seed {seed} | epoch {epoch+1:>2}/{EPOCHS} | train loss {running/seen:.4f} "
                  f"| val acc {accuracy_score(vt, vpred):.4f} | macroF1 {f1:.4f}{flag}", flush=True)
            if since_best >= patience:
                print(f"seed {seed} | early stop at epoch {epoch+1}", flush=True)
                break

        model.load_state_dict(best_state)
        tp, y_true = predict(model, test_loader, tta=True)
        seed_test_probs.append(tp)
        acc = accuracy_score(y_true, tp.argmax(1))
        seed_summaries.append((seed, best_f1, acc))
        print(f"\nseed {seed} done | best val macroF1 {best_f1:.4f} | test acc (TTA) {acc:.4f}\n",
              flush=True)
        torch.save({"state_dict": model.state_dict(), "classes": names, "img_size": RES,
                    "mean": MEAN, "std": STD, "meta_cols": meta_cols},
                   os.path.join(SP, f"skin_cnn_meta_seed{seed}.pt"))

    # ---------------- ensemble ----------------
    def report(tag, y_true, prob):
        pred = prob.argmax(1)
        print("\n" + "=" * 64)
        print(tag)
        print(f"  accuracy          : {accuracy_score(y_true, pred):.4f}")
        print(f"  balanced accuracy : {balanced_accuracy_score(y_true, pred):.4f}")
        print(f"  macro F1          : {f1_score(y_true, pred, average='macro'):.4f}")
        print(f"  weighted F1       : {f1_score(y_true, pred, average='weighted'):.4f}")
        print(f"  Cohen's kappa     : {cohen_kappa_score(y_true, pred):.4f}")
        print(f"  macro ROC-AUC     : {roc_auc_score(y_true, prob, multi_class='ovr', average='macro'):.4f}")
        mal = [1, 3, 4]
        t = np.isin(y_true, mal).astype(int); p = np.isin(pred, mal).astype(int)
        tp_ = int(((t == 1) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
        print(f"  malignant sensitivity: {tp_/(tp_+fn):.4f}  specificity: {tn/(tn+fp):.4f}")
        print(f"  melanomas called benign: {int(((y_true==1) & ~np.isin(pred, mal)).sum())}"
              f" of {int((y_true==1).sum())}")
        return accuracy_score(y_true, pred)

    print("\nper-seed test accuracy (TTA):")
    for seed, vf1, acc in seed_summaries:
        print(f"  seed {seed}: {acc:.4f}   (best val macroF1 {vf1:.4f})")

    ens = np.mean(seed_test_probs, axis=0)
    report(f"TEST - ensemble of {len(SEEDS)} seeds, + TTA", y_true, ens)

    codes = lesion_codes[idx_test]
    pooled = ens.copy()
    for c in np.unique(codes):
        m = codes == c
        pooled[m] = ens[m].mean(axis=0)
    report(f"TEST - ensemble + TTA + per-lesion pooling", y_true, pooled)

    print()
    print(classification_report(y_true, pooled.argmax(1), target_names=names,
                                digits=3, zero_division=0))
    cm = confusion_matrix(y_true, pooled.argmax(1))
    print("confusion matrix (rows = true, cols = predicted):")
    for i, row in enumerate(cm):
        print(f"{names[i]:<32} " + " ".join(f"{v:>5d}" for v in row))

    json.dump({"seeds": [{"seed": s, "val_macro_f1": f, "test_acc_tta": a}
                         for s, f, a in seed_summaries],
               "ensemble_pooled_accuracy": float(accuracy_score(y_true, pooled.argmax(1))),
               "confusion_matrix": cm.tolist(), "meta_cols": meta_cols},
              open(os.path.join(SP, "results_v3.json"), "w"), indent=2)
    print("\nsaved results_v3.json")


if __name__ == "__main__":
    main()
