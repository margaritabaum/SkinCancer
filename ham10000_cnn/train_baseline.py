"""Replicates cells 31-47 of 'Skin_Cancer (1).ipynb' and prints the test metrics.

Same model, loss, weights, optimizer, schedule, early stopping and evaluation as
the notebook. Device is MPS instead of CUDA.
"""
import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, f1_score, cohen_kappa_score)

SP = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1)
np.random.seed(1)

names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions ',
         'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions',
         'Dermatofibroma']

x = np.load(os.path.join(SP, "x71.npy"))
s = np.load(os.path.join(SP, "split.npz"))
y, is_train, is_val, is_test = s["y"], s["is_train"], s["is_val"], s["is_test"]

x_train, y_train = x[is_train], y[is_train]
x_val, y_val = x[is_val], y[is_val]
x_test, y_test = x[is_test], y[is_test]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device, flush=True)

MEAN = (x_train.mean(axis=(0, 1, 2)) / 255).astype(np.float32)
STD = (x_train.std(axis=(0, 1, 2)) / 255).astype(np.float32)
print("channel means:", MEAN.round(3), "| stds:", STD.round(3))

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Normalize(MEAN, STD)


class SkinDataset(Dataset):
    def __init__(self, x, y, tf=None):
        self.x, self.y, self.tf = x, y, tf

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).permute(2, 0, 1).float().div(255)
        if self.tf is not None:
            img = self.tf(img)
        return img, int(self.y[i])


BATCH_SIZE = 64
train_loader = DataLoader(SkinDataset(x_train, y_train, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(SkinDataset(x_val, y_val, eval_tf), batch_size=BATCH_SIZE)
test_loader = DataLoader(SkinDataset(x_test, y_test, eval_tf), batch_size=BATCH_SIZE)


def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SkinCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


model = SkinCNN(num_classes=7).to(device)
print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

counts = np.bincount(y_train, minlength=7)
weights = len(y_train) / (7 * np.maximum(counts, 1))
weights = weights / weights.mean()
weights = torch.tensor(weights, dtype=torch.float32, device=device)
for i in range(7):
    print(f"{names[i]:<32} {counts[i]:>5} images   weight {weights[i]:.2f}")

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
EPOCHS = 30
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


@torch.no_grad()
def evaluate(model, loader, want_probs=False):
    model.eval()
    total_loss, preds, trues, probs = 0.0, [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item() * yb.size(0)
        preds.append(out.argmax(1).cpu())
        trues.append(yb.cpu())
        if want_probs:
            probs.append(out.softmax(1).cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    if want_probs:
        return total_loss / len(trues), preds, trues, torch.cat(probs).numpy()
    return total_loss / len(trues), preds, trues


history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bacc": []}
best_bacc, best_state, patience, since_best = -1.0, None, 7, 0

for epoch in range(EPOCHS):
    model.train()
    running, seen = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * yb.size(0)
        seen += yb.size(0)
    scheduler.step()

    train_loss = running / seen
    val_loss, val_preds, val_true = evaluate(model, val_loader)
    acc = accuracy_score(val_true, val_preds)
    bacc = balanced_accuracy_score(val_true, val_preds)
    for k, v in zip(history, (train_loss, val_loss, acc, bacc)):
        history[k].append(float(v))

    flag = ""
    if bacc > best_bacc:
        best_bacc, since_best = bacc, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        flag = "  <- best so far"
    else:
        since_best += 1

    print(f"epoch {epoch+1:>2}/{EPOCHS} | train loss {train_loss:.4f} | "
          f"val loss {val_loss:.4f} | acc {acc:.4f} | balanced acc {bacc:.4f}{flag}",
          flush=True)
    if since_best >= patience:
        print(f"stopping early - no improvement for {patience} epochs")
        break

model.load_state_dict(best_state)
print(f"\nbest validation balanced accuracy: {best_bacc:.4f}")

# ---------------- final test evaluation ----------------
test_loss, y_pred, y_true, y_prob = evaluate(model, test_loader, want_probs=True)
acc = accuracy_score(y_true, y_pred)
bacc = balanced_accuracy_score(y_true, y_pred)

print("\n" + "=" * 62)
print(f"test loss           : {test_loss:.4f}")
print(f"accuracy            : {acc:.4f}")
print(f"balanced accuracy   : {bacc:.4f}")
print(f"macro F1            : {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"weighted F1         : {f1_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Cohen's kappa       : {cohen_kappa_score(y_true, y_pred):.4f}")
print(f"macro ROC-AUC (ovr) : {roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'):.4f}")
print()
print(classification_report(y_true, y_pred, target_names=names, digits=3, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("confusion matrix (rows = true, cols = predicted):")
print(pd_fmt := "\n".join(
    f"{names[i]:<32} " + " ".join(f"{v:>5d}" for v in row) for i, row in enumerate(cm)))

# clinically framed: melanoma + basal cell carcinoma + actinic keratoses = malignant/pre-malignant
mal = {1, 3, 4}
t_mal = np.isin(y_true, list(mal)).astype(int)
p_mal = np.isin(y_pred, list(mal)).astype(int)
prob_mal = y_prob[:, [1, 3, 4]].sum(1)
tp = int(((t_mal == 1) & (p_mal == 1)).sum()); fn = int(((t_mal == 1) & (p_mal == 0)).sum())
fp = int(((t_mal == 0) & (p_mal == 1)).sum()); tn = int(((t_mal == 0) & (p_mal == 0)).sum())
print(f"\nmalignant/pre-malignant vs benign (collapsed to 2 classes)")
print(f"  sensitivity (recall) : {tp/(tp+fn):.4f}   ({tp}/{tp+fn})")
print(f"  specificity          : {tn/(tn+fp):.4f}   ({tn}/{tn+fp})")
print(f"  ROC-AUC              : {roc_auc_score(t_mal, prob_mal):.4f}")
print(f"  melanoma missed as benign: "
      f"{int(((y_true==1) & ~np.isin(y_pred, list(mal))).sum())} of {int((y_true==1).sum())}")

torch.save({"state_dict": model.state_dict(), "classes": names, "img_size": 71,
            "mean": MEAN, "std": STD}, os.path.join(SP, "skin_cnn.pt"))
json.dump({"history": history, "test": {"loss": test_loss, "accuracy": acc,
           "balanced_accuracy": bacc}, "confusion_matrix": cm.tolist()},
          open(os.path.join(SP, "results.json"), "w"), indent=2)
print("\nsaved skin_cnn.pt and results.json")
