# classifier_app.py
import os, json, re, io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from urllib.parse import urlparse
from torch.nn.utils.rnn import pack_padded_sequence

# ----------------------- Styling & Page Setup -----------------------
st.set_page_config(page_title="DGA URL Checker", page_icon="🔎", layout="centered")

# Theme variables for colors and background
ACCENT = "#2563eb"   # main accent color (blue)
blur_px = 0          # blur applied to background overlay
BG_CSS_URL = (       # background image URL
    "https://imgs.search.brave.com/4ka_gfrVlQtyQUhDN-3AMeQQ3bEmvts6yZEy0SKZ7ms/"
    "rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRpYS5pc3RvY2twaG90by5jb20vaWQvMTcyNjM0ODQwNy92ZWN0b3Iv"
    "bWFsaWNpb3VzLWF0dGFjaG1lbnRzLWlzb2xhdGVkLWNhcnRvb24tdmVjdG9yLWlsbHVzdHJhdGlvbnMuanBnP3M9NjEy"
    "eDYxMiZ3PTAmaz0yMCZjPXJES2ZRSFRueUFXZGdVT21yclJzdzBiUWRLTnc3RG1mU2x4enpVM3pYR3M9"
)

# Custom CSS for Streamlit app styling
CUSTOM_CSS = f"""
<style>
:root {{
  --accent: {ACCENT};
  --overlay: 0.82;
  --panel-bg: rgba(255,255,255,0.98);
  --panel-border: #e5e7eb;
  --text-strong: #0f172a;
  --bg-url: url('{BG_CSS_URL}');
}}
...
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------- Hero Banner -----------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
      <span style="font-size:28px;">🔒</span>
      <span style="font-size:30px; font-weight:800;">URL Maliciousness Checker</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Detect malicious or benign domains using a trained BiLSTM model.")

# ----------------------- Paths & Loading -----------------------
def find_file(candidates):
    """Return first existing file path from a list of candidate paths."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

# Define candidate paths for config and model weights
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_CFG = [
    os.path.join(HERE, "inference_config.json"),
    os.path.join(HERE, "artifacts", "inference_config.json"),
    "inference_config.json", "artifacts/inference_config.json",
]
CANDIDATE_WEIGHTS = [
    os.path.join(HERE, "dga_bilstm_model.pth"),
    os.path.join(HERE, "artifacts", "dga_bilstm_model.pth"),
    "dga_bilstm_model.pth", "artifacts/dga_bilstm_model.pth",
]

# Try to resolve paths
CFG_PATH = find_file(CANDIDATE_CFG)
WTS_PATH = find_file(CANDIDATE_WEIGHTS)

# Sidebar inputs to override config/weights path if not auto-detected
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("If the app can’t find your model/config, point to them here:")
    cfg_in = st.text_input("Path to inference_config.json", CFG_PATH or "")
    wts_in = st.text_input("Path to dga_bilstm_model.pth", WTS_PATH or "")
    if cfg_in: CFG_PATH = cfg_in
    if wts_in: WTS_PATH = wts_in

# ----------------------- Model & Helpers -----------------------
class BiLSTMClassifier(nn.Module):
    """Bi-directional LSTM classifier for domain strings."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)  # *2 for bidirectional
    def forward(self, x):
        lengths = (x != 0).sum(dim=1)  # sequence lengths (excluding padding)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x)
        # Concatenate forward and backward final hidden states
        h = torch.cat((h_n[0], h_n[1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

@st.cache_resource(show_spinner=False)
def load_model_and_cfg(cfg_path, weights_path):
    """Load model config JSON and corresponding PyTorch model weights."""
    if not cfg_path or not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing inference_config.json at: {cfg_path}")
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weight file at: {weights_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create model
    model = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    # Load weights
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, device

def clean_domain(s: str) -> str:
    """Extract and normalize domain name from a URL or raw string."""
    s = (s or "").strip()
    if not s:
        return s
    if "://" not in s:
        s = "http://" + s
    netloc = urlparse(s).netloc.lower()
    netloc = re.sub(r"^www\d*\.", "", netloc)  # strip leading www or www2
    return netloc

def encode_domain(domain: str, char2idx: dict, max_len: int):
    """Convert domain string to integer indices padded to max_len."""
    ids = [char2idx.get(c, 0) for c in domain.lower()][:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def predict(domain: str, model, cfg, device):
    """Run model prediction for a single domain and return malicious probability."""
    x = encode_domain(domain, cfg["char2idx"], cfg["max_length"]).to(device)
    with torch.no_grad():
        prob_mal = F.softmax(model(x), dim=-1)[0, 1].item()
    return prob_mal

# ----------------------- Load Model -----------------------
if not CFG_PATH or not WTS_PATH:
    st.error("❌ Could not find `inference_config.json` or `dga_bilstm_model.pth`.")
    st.stop()

try:
    model, cfg, device = load_model_and_cfg(CFG_PATH, WTS_PATH)
except Exception as e:
    st.error(f"Model load failed: `{e}`")
    st.stop()

# Sidebar threshold control
with st.sidebar:
    st.markdown("### 🧪 Decision threshold")
    thr_default = float(cfg.get("threshold", 0.5))
    thr = st.slider("Malicious threshold", 0.0, 1.0, value=thr_default, step=0.001)
    st.caption(f"Default from validation: **{thr_default:.3f}**")

# ----------------------- Single URL card -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Single Check")
url = st.text_input("URL or domain", placeholder="e.g., http://phishy-site.biz or example.com")

if st.button("Classify", type="primary", use_container_width=True) and url.strip():
    dom = clean_domain(url)                  # normalize input
    prob = predict(dom, model, cfg, device)  # run model inference

    st.write(f"**Parsed domain:** `{dom}`")
    st.write(f"**Malicious probability:** `{prob:.4f}` | **Threshold:** `{thr:.3f}`")

    # Animated probability bar
    bar_html = f"""
    <div class="progress-holder">
        <div class="progress-bar" style="width: {prob*100:.1f}%;"></div>
    </div>
    <div class="progress-label">{prob*100:.1f}% malicious</div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

    # Badge display based on threshold
    if prob >= thr:
        st.markdown('<span class="badge badge-bad">🛑 Malicious</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-ok">✅ Benign</span>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Batch CSV card (optional) -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Batch (CSV)")
st.caption("Upload a CSV with a **`domain`** column. We’ll score each row and let you download the results.")

up = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
if up is not None:
    try:
        df = pd.read_csv(up)
        if "domain" not in df.columns:
            st.warning("CSV must include a `domain` column.")
        else:
            # Clean and predict each domain
            clean = df["domain"].astype(str).map(clean_domain)
            probs = [predict(d, model, cfg, device) for d in clean]
            labels = (np.array(probs) >= thr).astype(int)

            # Build output dataframe with predictions
            out = df.copy()
            out["domain_clean"] = clean
            out["prob_malicious"] = np.round(probs, 6)
            out["prediction"] = np.where(labels == 1, "Malicious", "Benign")

            st.dataframe(out.head(20), use_container_width=True, hide_index=True)

            # Allow user to download scored CSV
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download scored CSV",
                buf.getvalue().encode("utf-8"),
                file_name="scored_domains.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Could not read CSV: `{e}`")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Footer -----------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Tip: If you deployed on a server, keep the **model** and **config** files in a mounted `artifacts/` folder "
    "and point to them from the sidebar."
)
