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

# Theme variables
ACCENT = "#2563eb"   # accent color
blur_px = 0          # keep text crisp
BG_CSS_URL = (
    "https://imgs.search.brave.com/4ka_gfrVlQtyQUhDN-3AMeQQ3bEmvts6yZEy0SKZ7ms/"
    "rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRpYS5pc3RvY2twaG90by5jb20vaWQvMTcyNjM0ODQwNy92ZWN0b3Iv"
    "bWFsaWNpb3VzLWF0dGFjaG1lbnRzLWlzb2xhdGVkLWNhcnRvb24tdmVjdG9yLWlsbHVzdHJhdGlvbnMuanBnP3M9NjEy"
    "eDYxMiZ3PTAmaz0yMCZjPXJES2ZRSFRueUFXZGdVT21yclJzdzBiUWRLTnc3RG1mU2x4enpVM3pYR3M9"
)

# High-contrast CSS
CUSTOM_CSS = f"""
<style>
:root {{
  --accent: {ACCENT};
  --overlay: 0.82;                 /* darker bg overlay */
  --panel-bg: rgba(255,255,255,0.98);
  --panel-border: #e5e7eb;
  --text-strong: #0f172a;
  --bg-url: url('{BG_CSS_URL}');
}}

/* Big background image with dark overlay */
.stApp {{
  background-image:
    linear-gradient(rgba(10,15,25,var(--overlay)), rgba(10,15,25,var(--overlay))),
    var(--bg-url);
  background-size: cover;
  background-position: center center;
  background-attachment: fixed;
}}

/* Make the main content a solid white panel for readability */
.block-container {{
  max-width: 1100px;
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  box-shadow: 0 10px 28px rgba(0,0,0,0.15);
  border-radius: 18px;
  padding: 28px 28px 22px 28px;
  margin-top: 22px;
  backdrop-filter: blur({blur_px}px);
}}

/* Headings & labels darker */
h1,h2,h3,h4,h5, label, .stCaption, .stMarkdown, p, small, span {{
  color: var(--text-strong) !important;
  text-shadow: none !important;
}}

/* Glass card (still used for inner sections if you want the look) */
.card {{
  background: #ffffff;
  border: 1px solid #eef2f7;
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}}
.card:hover {{ transform: none; }}

/* Inputs & uploader: pure white */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div[data-baseweb="select"] > div,
.stFileUploader > div {{
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
  box-shadow: none !important;
}}
/* Slider accent */
.stSlider [role="slider"] {{
  background: var(--accent) !important;
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 28%, transparent) !important;
}}

/* Primary button */
.stButton > button[kind="primary"], .stButton > button {{
  background: var(--accent) !important;
  border: 1px solid #1d4ed8 !important;
  color: #fff !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  padding: .6rem 1rem !important;
}}
.stButton > button:hover {{ filter: brightness(0.95); }}

/* Probability bar */
.progress-holder {{
  background: #f8fafc;
  border-radius: 12px; padding: 6px;
  border: 1px solid #e2e8f0; margin-top: 8px;
}}
.progress-bar {{
  height: 18px; border-radius: 8px;
  background: linear-gradient(90deg, var(--accent), #ff8e53);
  width: 0%; transition: width .9s ease;
}}
.progress-label {{
  font-weight:700; margin-top:6px;
  color:#0f172a;
}}

/* Badges */
.badge {{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.55rem .9rem; border-radius:999px; font-weight:700; letter-spacing:.2px;
  border:1px solid;
}}
.badge-ok  {{ background:#E8F5E9; color:#1B5E20; border-color:#C8E6C9; }}
.badge-bad {{ background:#FFEBEE; color:#B00020; border-color:#FFCDD2; }}
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
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

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

CFG_PATH = find_file(CANDIDATE_CFG)
WTS_PATH = find_file(CANDIDATE_WEIGHTS)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("If the app can’t find your model/config, point to them here:")
    cfg_in = st.text_input("Path to inference_config.json", CFG_PATH or "")
    wts_in = st.text_input("Path to dga_bilstm_model.pth", WTS_PATH or "")
    if cfg_in: CFG_PATH = cfg_in
    if wts_in: WTS_PATH = wts_in

# ----------------------- Model & Helpers -----------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        lengths = (x != 0).sum(dim=1)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[0], h_n[1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

@st.cache_resource(show_spinner=False)
def load_model_and_cfg(cfg_path, weights_path):
    if not cfg_path or not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing inference_config.json at: {cfg_path}")
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weight file at: {weights_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, device

def clean_domain(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if "://" not in s:
        s = "http://" + s
    netloc = urlparse(s).netloc.lower()
    netloc = re.sub(r"^www\d*\.", "", netloc)
    return netloc

def encode_domain(domain: str, char2idx: dict, max_len: int):
    ids = [char2idx.get(c, 0) for c in domain.lower()][:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def predict(domain: str, model, cfg, device):
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
    dom = clean_domain(url)
    prob = predict(dom, model, cfg, device)

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
            clean = df["domain"].astype(str).map(clean_domain)
            probs = [predict(d, model, cfg, device) for d in clean]
            labels = (np.array(probs) >= thr).astype(int)
            out = df.copy()
            out["domain_clean"] = clean
            out["prob_malicious"] = np.round(probs, 6)
            out["prediction"] = np.where(labels == 1, "Malicious", "Benign")

            st.dataframe(out.head(20), use_container_width=True, hide_index=True)

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
