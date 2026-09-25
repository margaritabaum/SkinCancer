"""Improved training run for the notebook's from-scratch CNN.

Changes vs the notebook, each one targeting a loss of accuracy identified in the code:
  1. 128x128 input instead of 71x71            (cell 22)
  2. explicit 5-block CNN, 512 channels         (cell 36)
  3. sqrt class weights, ~7x spread not 53x     (cell 39)
  4. model selected on macro F1, not balanced accuracy (cell 41)
  5. 60 epochs so the cosine schedule completes (cell 39)
  6. test-time augmentation + per-lesion pooling (new)
"""
import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, f1_score, cohen_kappa_score)
from model import CNN

SP = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1)
np.random.seed(1)

names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions ',
         'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions',
         'Dermatofibroma']
RES = 128
EPOCHS = 60
BATCH_SIZE = 64

x = np.load(os.path.join(SP, "x256.npy"), mmap_mode="r")
s = np.load(os.path.join(SP, "split256.npz"))
y, is_train, is_val, is_test = s["y"], s["is_train"], s["is_val"], s["is_test"]
lesion_codes = s["lesion_codes"]

idx_train = np.where(is_train)[0]
idx_val = np.where(is_val)[0]
idx_test = np.where(is_test)[0]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device, "| resolution:", RES, flush=True)

# Channel statistics from the TRAINING split only.
# Keep the data uint8 and force a float64 accumulator: averaging 52M float32
# values saturates the 24-bit mantissa and silently returns the same wrong
# number for every channel.
sample = np.asarray(x[idx_train[::10]])
MEAN = (sample.mean(axis=(0, 1, 2), dtype=np.float64) / 255).astype(np.float32)
STD = (sample.std(axis=(0, 1, 2), dtype=np.float64) / 255).astype(np.float32)
del sample
assert MEAN.std() > 1e-3, f"channels should differ on skin images, got {MEAN}"
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
eval_tf = transforms.Compose([
    transforms.Resize((RES, RES), antialias=True),
    transforms.Normalize(MEAN, STD),
])


class SkinDataset(Dataset):
    def __init__(self, x, idx, y, tf):
        self.x, self.idx, self.y, self.tf = x, idx, y, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = torch.from_numpy(np.asarray(self.x[j])).permute(2, 0, 1).float().div(255)
        return self.tf(img), int(self.y[j])


train_loader = DataLoader(SkinDataset(x, idx_train, y, train_tf), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True, num_workers=0)
val_loader = DataLoader(SkinDataset(x, idx_val, y, eval_tf), batch_size=BATCH_SIZE,
                        num_workers=0)
test_loader = DataLoader(SkinDataset(x, idx_test, y, eval_tf), batch_size=BATCH_SIZE,
                         num_workers=0)

model = CNN(num_classes=7, dropout=0.4).to(device)
print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

# --- fix 3: sqrt inverse-frequency weights (notebook used the raw ratio, 53x) ---
counts = np.bincount(y[idx_train], minlength=7)
weights = np.sqrt(len(idx_train) / (7 * np.maximum(counts, 1)))
weights = weights / weights.mean()
print(f"weight spread: {weights.max()/weights.min():.1f}x  (notebook was 53x)")
weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


@torch.no_grad()
def evaluate(model, loader, tta=False):
    model.eval()
    total_loss, probs, trues = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item() * yb.size(0)
        p = out.softmax(1)
        if tta:  # average over the 4 flips a lesion photo is invariant to
            for f in (lambda t: t.flip(-1), lambda t: t.flip(-2),
                      lambda t: t.flip(-1).flip(-2)):
                p = p + model(f(xb)).softmax(1)
            p = p / 4
        probs.append(p.cpu())
        trues.append(yb.cpu())
    probs = torch.cat(probs).numpy()
    trues = torch.cat(trues).numpy()
    return total_loss / len(trues), probs.argmax(1), trues, probs


history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bacc": [], "val_f1": []}
best_f1, best_state, patience, since_best = -1.0, None, 12, 0

for epoch in range(EPOCHS):
    model.train()
    running, seen = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * yb.size(0)
        seen += yb.size(0)
    scheduler.step()

    train_loss = running / seen
    val_loss, val_preds, val_true, _ = evaluate(model, val_loader)
    acc = accuracy_score(val_true, val_preds)
    bacc = balanced_accuracy_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds, average="macro")
    for k, v in zip(history, (train_loss, val_loss, acc, bacc, f1)):
        history[k].append(float(v))

    flag = ""
    if f1 > best_f1:                      # --- fix 4: select on macro F1 ---
        best_f1, since_best = f1, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        flag = "  <- best so far"
    else:
        since_best += 1

    print(f"epoch {epoch+1:>2}/{EPOCHS} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
          f"| acc {acc:.4f} | bacc {bacc:.4f} | macroF1 {f1:.4f}{flag}", flush=True)
    if since_best >= patience:
        print(f"stopping early - no improvement for {patience} epochs")
        break

model.load_state_dict(best_state)
print(f"\nbest validation macro F1: {best_f1:.4f}")

# ---------------- final test evaluation ----------------
def report(tag, y_true, y_pred, y_prob):
    print("\n" + "=" * 64)
    print(f"{tag}")
    print(f"  accuracy          : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  balanced accuracy : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  macro F1          : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  weighted F1       : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Cohen's kappa     : {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"  macro ROC-AUC     : {roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'):.4f}")
    mal = [1, 3, 4]
    t = np.isin(y_true, mal).astype(int); p = np.isin(y_pred, mal).astype(int)
    tp = int(((t == 1) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
    print(f"  malignant sensitivity: {tp/(tp+fn):.4f}  specificity: {tn/(tn+fp):.4f}")
    print(f"  melanoma missed as benign: {int(((y_true==1) & ~np.isin(y_pred, mal)).sum())}"
          f" of {int((y_true==1).sum())}")
    return accuracy_score(y_true, y_pred)


_, pred_plain, y_true, prob_plain = evaluate(model, test_loader)
report("TEST - single view", y_true, pred_plain, prob_plain)

_, _, _, prob_tta = evaluate(model, test_loader, tta=True)   # --- fix 6a: TTA ---
report("TEST - + test-time augmentation", y_true, prob_tta.argmax(1), prob_tta)

# --- fix 6b: pool the several photos of one lesion into one prediction ---
codes = lesion_codes[idx_test]
prob_pooled = prob_tta.copy()
for c in np.unique(codes):
    m = codes == c
    prob_pooled[m] = prob_tta[m].mean(axis=0)
report("TEST - + per-lesion pooling", y_true, prob_pooled.argmax(1), prob_pooled)

print()
print(classification_report(y_true, prob_pooled.argmax(1), target_names=names,
                            digits=3, zero_division=0))
cm = confusion_matrix(y_true, prob_pooled.argmax(1))
print("confusion matrix (rows = true, cols = predicted):")
for i, row in enumerate(cm):
    print(f"{names[i]:<32} " + " ".join(f"{v:>5d}" for v in row))

torch.save({"state_dict": model.state_dict(), "classes": names, "img_size": RES,
            "mean": MEAN, "std": STD}, os.path.join(SP, "skin_cnn_v2.pt"))
json.dump({"history": history,
           "test_accuracy": float(accuracy_score(y_true, prob_pooled.argmax(1))),
           "confusion_matrix": cm.tolist()},
          open(os.path.join(SP, "results_v2.json"), "w"), indent=2)
print("\nsaved skin_cnn_v2.pt and results_v2.json")
