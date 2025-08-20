import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, classification_report
from string import ascii_lowercase, digits
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

# ===================== PLOTS (import or fallback) =====================
# Try to import custom plotting utilities; if unavailable, define fallbacks inline
try:
    from plots_utils import (
        plot_training_curves,
        plot_confusion,
        plot_roc_pr,
        plot_domain_distributions,
    )
    _HAS_PLOTS_UTILS = True
except Exception:
    _HAS_PLOTS_UTILS = False
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix, ConfusionMatrixDisplay,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )

    # Utility: make sure directory exists
    def _ensure_dir(d="plots"):
        os.makedirs(d, exist_ok=True); return d

    # Fallback for training curve plots
    def plot_training_curves(history, outdir="plots", title_prefix=""):
        outdir=_ensure_dir(outdir)
        plt.figure()
        if "train_loss" in history: plt.plot(history["train_loss"], label="train")
        if "val_loss" in history:   plt.plot(history["val_loss"],   label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title_prefix} Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_loss.png")); plt.close()
        if ("train_acc" in history) or ("val_acc" in history):
            plt.figure()
            if "train_acc" in history: plt.plot(history["train_acc"], label="train")
            if "val_acc" in history:   plt.plot(history["val_acc"],   label="val")
            plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} Accuracy")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_accuracy.png")); plt.close()

    # Fallback for confusion matrix
    def plot_confusion(y_true, y_prob, threshold=0.5, outdir="plots", title="Confusion Matrix"):
        outdir=_ensure_dir(outdir)
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","Malicious"])
        plt.figure(); disp.plot(values_format="d"); plt.title(title); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title.lower().replace(' ','_')}.png")); plt.close()

    # Fallback for ROC + PR curves
    def plot_roc_pr(y_true, y_prob, outdir="plots", title_prefix="Model"):
        outdir=_ensure_dir(outdir)
        y_true=np.array(y_true); y_prob=np.array(y_prob)
        fpr,tpr,_=roc_curve(y_true,y_prob); roc_auc=auc(fpr,tpr)
        plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{title_prefix} ROC")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir,f"{title_prefix.lower().replace(' ','_')}_roc.png")); plt.close()
        prec,rec,_=precision_recall_curve(y_true,y_prob); ap=average_precision_score(y_true,y_prob)
        plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title_prefix} Precision-Recall")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir,f"{title_prefix.lower().replace(' ','_')}_pr.png")); plt.close()

# ---------------- CONFIG ----------------
MAX_LENGTH = 32                      # max characters per domain
BATCH_SIZE = 64                      # batch size
EPOCHS = 10                          # max epochs
LR = 3e-4                            # learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character vocabulary (lowercase letters, digits, dash, dot)
CHARS = list(ascii_lowercase + digits + "-.")
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for padding
VOCAB_SIZE = len(CHAR2IDX) + 1

# Model hyperparameters
EMBED_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.3

# Outputs & labels
PLOTS_OUTDIR = "plots"
TITLE_PREFIX = "Baseline"

# Sampler & training strategy
POS_TARGET = 0.25           # target malicious fraction in balanced batches
EARLY_STOP_PATIENCE = 2     # stop if no val F1 improvement
THRESHOLD_METRIC = "f1"     # threshold tuning metric: f1 / f05 / f2

# ---- Load Dataset ----
benign = pd.read_csv("top-1m.csv", names=["rank","domain"], usecols=[1]); benign["label"]=0
mal    = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"]); mal["label"]=1
df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Optional distribution plots if utils available
if _HAS_PLOTS_UTILS:
    try:
        plot_domain_distributions(df["domain"].astype(str).tolist(), outdir=PLOTS_OUTDIR, title_prefix="All Domains")
    except Exception:
        pass

# ---- Preprocess ----
def encode_domain(domain):
    """Map each character to an integer index and pad/truncate to MAX_LENGTH."""
    s=str(domain).lower()
    v=[CHAR2IDX.get(c,0) for c in s][:MAX_LENGTH]
    v += [0]*(MAX_LENGTH-len(v))   # pad with 0
    return v

df["encoded"] = df["domain"].apply(encode_domain)
X = np.array(df["encoded"].tolist(), dtype=np.int64)
y = df["label"].values

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,stratify=y_temp,random_state=42)

# ---- Dataset class ----
class DomainDataset(Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)

# Wrap splits into Dataset objects
train_ds, val_ds, test_ds = DomainDataset(X_train,y_train), DomainDataset(X_val,y_val), DomainDataset(X_test,y_test)

# Show data sizes & label distributions
print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))
print("Train label dist:", np.bincount(y_train))
print("Val   label dist:", np.bincount(y_val))
print("Test  label dist:", np.bincount(y_test))

# ---- WeightedRandomSampler to balance classes ----
count0, count1 = np.bincount(y_train)
w0 = (1.0 - POS_TARGET) / max(count0, 1)
w1 = POS_TARGET / max(count1, 1)
class_weights = np.array([w0, w1], dtype=np.float64)
sample_weights = class_weights[y_train]
print("Using WeightedRandomSampler with class weights (target pos frac =",
      POS_TARGET, "):", class_weights)

# DataLoaders
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ---- Model: BiLSTM ----
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        lengths = (x != 0).sum(dim=1)  # actual lengths (before padding)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat((h_n[0], h_n[1]), dim=1)   # concat forward/backward
        h = self.dropout(h)
        return self.fc(h)  # logits (B,2)

model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)

criterion = nn.CrossEntropyLoss()                     # sampler balances data → no weights needed
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# ---- Helpers ----
def batch_metrics_from_logits(logits, y_true):
    """Return preds, probs, and loss for a batch."""
    loss = criterion(logits, y_true).item()
    probs = F.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    return preds, probs, loss

def evaluate_loader(dataloader, threshold=0.5):
    """Evaluate model on loader at given threshold."""
    model.eval()
    preds_all, probs_all, y_all = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            preds, probs_pos, loss = batch_metrics_from_logits(logits, yb)
            preds_all.extend(preds); probs_all.extend(probs_pos); y_all.extend(yb.cpu().numpy()); total_loss += loss
    preds_thr = (np.array(probs_all) >= threshold).astype(int)
    acc = accuracy_score(y_all, preds_thr)
    f1  = f1_score(y_all, preds_thr)
    return total_loss/max(1,len(dataloader)), acc, f1, np.array(y_all), np.array(probs_all)

def find_best_threshold(y_true, y_prob, metric="f1"):
    """Grid search over thresholds to maximize F1 (or F0.5/F2)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.append(thr, 1.0)
    if metric == "f05":
        beta2 = 0.5**2; f = (1+beta2)*(prec*rec)/(beta2*prec+rec+1e-12)
    elif metric == "f2":
        beta2 = 2**2; f = (1+beta2)*(prec*rec)/(beta2*prec+rec+1e-12)
    else:
        f = 2*prec*rec/(prec+rec+1e-12)
    idx = np.argmax(f)
    return float(thr[idx]), float(f[idx]), float(prec[idx]), float(rec[idx])

# ---- Training Loop with Early Stopping ----
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_f1 = -1.0; best_state = None; epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # gradient clipping
        optimizer.step()
        running_loss += loss.item()
        running_correct += (logits.argmax(dim=1) == yb).sum().item()
        running_total += yb.size(0)
    train_loss = running_loss/max(1,len(train_dl))
    train_acc = running_correct/max(1,running_total)

    # Evaluate validation set
    val_loss, val_acc, val_f1, y_val_true, y_val_prob = evaluate_loader(val_dl, threshold=0.5)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Validation -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc);   history["val_acc"].append(val_acc)

    # Early stopping based on tuned threshold F1
    thr_star, f_star, p_star, r_star = find_best_threshold(y_val_true, y_val_prob, metric=THRESHOLD_METRIC)
    print(f"VAL best-{THRESHOLD_METRIC}: F1={f_star:.4f} @ thr={thr_star:.3f} (P={p_star:.3f}, R={r_star:.3f})")

    if f_star > best_val_f1 + 1e-4:
        best_val_f1 = f_star
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered (no val F1 improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state)

# Save training curves
try:
    plot_training_curves(history, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX)
    print(f"Saved training curves to '{PLOTS_OUTDIR}/'.")
except Exception as e:
    print(f"[WARN] Could not save training curves: {e}")

# ---- Tune threshold on validation, then test ----
_, _, _, y_val_true, y_val_prob = evaluate_loader(val_dl, threshold=0.5)
best_thr, best_f1, best_p, best_r = find_best_threshold(y_val_true, y_val_prob, metric=THRESHOLD_METRIC)
print(f"\nChosen threshold from VAL ({THRESHOLD_METRIC}): {best_thr:.3f}  "
      f"=> F1={best_f1:.4f}, P={best_p:.4f}, R={best_r:.4f}")

# === Save artifacts for inference (weights + config) ===
torch.save(model.state_dict(), "dga_bilstm_model.pth")
infer_cfg = {
    "max_length": int(MAX_LENGTH),
    "char2idx": CHAR2IDX,
    "threshold": float(best_thr),
    "embed_dim": int(EMBED_DIM),
    "hidden_dim": int(HIDDEN_DIM),
    "vocab_size": int(len(CHAR2IDX) + 1),
    "dropout": float(DROPOUT),
}
with open("inference_config.json", "w") as f:
    json.dump(infer_cfg, f)

# Also save threshold separately
with open("chosen_threshold.txt", "w") as f:
    f.write(f"{best_thr:.6f}\n")

# ---- Final Test ----
test_loss, test_acc, test_f1_05, y_test_true, y_test_prob = evaluate_loader(test_dl, threshold=best_thr)
y_test_pred = (y_test_prob >= best_thr).astype(int)
print("\nTEST @ tuned threshold")
print(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1_05:.4f}")
print(classification_report(y_test_true, y_test_pred, target_names=['Benign','Malicious'], digits=4))

# Confusion counts (TN, FP, FN, TP)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"Confusion (Test @ thr={best_thr:.3f}): TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ---- Plots ----
try:
    os.makedirs(PLOTS_OUTDIR, exist_ok=True)
    plot_confusion(y_test_true, y_test_prob, threshold=best_thr, outdir=PLOTS_OUTDIR, title=f"{TITLE_PREFIX} Confusion Matrix")
    plot_roc_pr(y_test_true, y_test_prob, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX)
    print(f"Saved evaluation plots to '{PLOTS_OUTDIR}/'.")
except Exception as e:
    print(f"[WARN] Could not save evaluation plots: {e}")
