import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from string import ascii_lowercase, digits
from tqdm import tqdm

# ===================== PLOTS (import or fallback) =====================
# Tries to import shared helpers; if unavailable, defines minimal versions inline.
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

    def _ensure_dir(d="plots"):
        os.makedirs(d, exist_ok=True)
        return d

    def plot_training_curves(history, outdir="plots", title_prefix=""):
        outdir = _ensure_dir(outdir)
        # Loss
        plt.figure()
        if "train_loss" in history: plt.plot(history["train_loss"], label="train")
        if "val_loss" in history: plt.plot(history["val_loss"], label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"{title_prefix} Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_loss.png"))
        plt.close()
        # Accuracy
        has_acc = ("train_acc" in history) or ("val_acc" in history)
        if has_acc:
            plt.figure()
            if "train_acc" in history: plt.plot(history["train_acc"], label="train")
            if "val_acc" in history: plt.plot(history["val_acc"], label="val")
            plt.xlabel("Epoch"); plt.ylabel("Accuracy")
            plt.title(f"{title_prefix} Accuracy")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_accuracy.png"))
            plt.close()

    def plot_confusion(y_true, y_prob, threshold=0.5, outdir="plots", title="Confusion Matrix"):
        outdir = _ensure_dir(outdir)
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malicious"])
        plt.figure()
        disp.plot(values_format="d")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title.lower().replace(' ', '_')}.png"))
        plt.close()

    def plot_roc_pr(y_true, y_prob, outdir="plots", title_prefix="Model"):
        outdir = _ensure_dir(outdir)
        y_true = np.array(y_true); y_prob = np.array(y_prob)
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"{title_prefix} ROC")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_roc.png"))
        plt.close()
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{title_prefix} Precision-Recall")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_pr.png"))
        plt.close()

# ---------------- CONFIG ----------------
MAX_LENGTH = 32
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARS = list(ascii_lowercase + digits + "-.")
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = padding
VOCAB_SIZE = len(CHAR2IDX) + 1
EMBED_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.5
PLOTS_OUTDIR = "plots"
TITLE_PREFIX = "Baseline"  # used in filenames/titles
# ----------------------------------------

# ---- Load Dataset ----
benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
benign["label"] = 0
mal = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
mal["label"] = 1
df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: quick EDA plots of domain distributions (once at start)
if _HAS_PLOTS_UTILS:
    try:
        plot_domain_distributions(df["domain"].astype(str).tolist(), outdir=PLOTS_OUTDIR, title_prefix="All Domains")
    except Exception:
        pass

# ---- Preprocess ----
def encode_domain(domain):
    domain = str(domain).lower()
    vec = [CHAR2IDX.get(c, 0) for c in domain][:MAX_LENGTH]
    vec += [0] * (MAX_LENGTH - len(vec))
    return vec

df["encoded"] = df["domain"].apply(encode_domain)
X = np.array(df["encoded"].tolist(), dtype=np.int64)
y = df["label"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ---- Dataset class ----
class DomainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

train_ds = DomainDataset(X_train, y_train)
val_ds = DomainDataset(X_val, y_val)
test_ds = DomainDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---- Model ----
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (2, batch, hidden_dim) → concatenate forward & backward
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        h_n = self.dropout(h_n)
        out = self.fc(h_n)  # logits shape: (batch, 2)
        return out

model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)

# ---- Loss with class weights ----
class_counts = np.bincount(y_train)
weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- Metrics helpers ----
def batch_metrics_from_logits(logits, y_true):
    """
    logits: torch.Tensor (B,2)
    y_true: torch.Tensor (B,)
    returns: preds (np.int64), probs_pos (np.float32), loss (float)
    """
    loss = criterion(logits, y_true).item()
    probs = F.softmax(logits, dim=-1)  # (B,2)
    probs_pos = probs[:, 1].detach().cpu().numpy()
    preds = probs.argmax(dim=1).detach().cpu().numpy()
    return preds, probs_pos, loss

# ---- Validation ----
def validate():
    model.eval()
    preds_all, probs_all, y_all = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            preds, probs_pos, loss = batch_metrics_from_logits(logits, yb)
            preds_all.extend(preds)
            probs_all.extend(probs_pos)
            y_all.extend(yb.cpu().numpy())
            total_loss += loss
    acc = accuracy_score(y_all, preds_all)
    f1 = f1_score(y_all, preds_all)
    val_loss = total_loss / max(1, len(val_dl))
    print(f"Validation -> Loss: {val_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    return val_loss, acc, f1

# ---- Training Loop with history ----
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

def train_model():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += yb.size(0)

        train_loss = running_loss / max(1, len(train_dl))
        train_acc = running_correct / max(1, running_total)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, _val_f1 = validate()

        # log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    # Save training curves
    try:
        plot_training_curves(history, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX)
        print(f"Saved training curves to '{PLOTS_OUTDIR}/'.")
    except Exception as e:
        print(f"[WARN] Could not save training curves: {e}")

train_model()

# ---- Test ----
model.eval()
preds_all, probs_all, y_all = [], [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        preds, probs_pos, _ = batch_metrics_from_logits(logits, yb)
        preds_all.extend(preds)
        probs_all.extend(probs_pos)
        y_all.extend(yb.cpu().numpy())

acc = accuracy_score(y_all, preds_all)
f1 = f1_score(y_all, preds_all)
print(f"Test -> Acc: {acc:.4f}, F1: {f1:.4f}")

# ---- Plots: Confusion Matrix, ROC, PR ----
try:
    os.makedirs(PLOTS_OUTDIR, exist_ok=True)
    plot_confusion(y_all, probs_all, threshold=0.5, outdir=PLOTS_OUTDIR, title=f"{TITLE_PREFIX} Confusion Matrix")
    plot_roc_pr(y_all, probs_all, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX)
    print(f"Saved evaluation plots to '{PLOTS_OUTDIR}/'.")
except Exception as e:
    print(f"[WARN] Could not save evaluation plots: {e}")

# ---- Optional: Save model ----
# torch.save(model.state_dict(), "dga_bilstm_model.pth")
