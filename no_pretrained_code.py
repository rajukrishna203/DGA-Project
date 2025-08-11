import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from string import ascii_lowercase, digits
from tqdm import tqdm

# ---------------- CONFIG ----------------
MAX_LENGTH = 32
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARS = list(ascii_lowercase + digits + "-.")
CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}  # 0 = padding
VOCAB_SIZE = len(CHAR2IDX) + 1
EMBED_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.5
# ----------------------------------------

# ---- Load Dataset ----
benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
benign["label"] = 0
mal = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
mal["label"] = 1
df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Preprocess ----
def encode_domain(domain):
    domain = domain.lower()
    vec = [CHAR2IDX.get(c, 0) for c in domain][:MAX_LENGTH]
    vec += [0]*(MAX_LENGTH-len(vec))
    return vec

df["encoded"] = df["domain"].apply(encode_domain)
X = np.array(df["encoded"].tolist(), dtype=np.int64)
y = df["label"].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

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
        out = self.fc(h_n)
        return out

model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)

# ---- Loss with class weights ----
class_counts = np.bincount(y_train)
weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- Training Loop ----
def validate():
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            y_all.extend(yb.cpu().numpy())
    acc = accuracy_score(y_all, preds_all)
    f1 = f1_score(y_all, preds_all)
    print(f"Validation -> Acc: {acc:.4f}, F1: {f1:.4f}")

def train_model():
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dl):.4f}")
        validate()

train_model()

# ---- Test ----
model.eval()
preds_all, y_all = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        preds_all.extend(preds)
        y_all.extend(yb.cpu().numpy())
acc = accuracy_score(y_all, preds_all)
f1 = f1_score(y_all, preds_all)
print(f"Test -> Acc: {acc:.4f}, F1: {f1:.4f}")

# torch.save(model.state_dict(), "dga_bilstm_model.pth")
