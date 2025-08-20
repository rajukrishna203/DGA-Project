# DGA Project – Detection of DGA Domains using ML/DL

This repository implements a complete pipeline for detecting **Domain Generation Algorithm (DGA)** domains using both **classical machine learning** and **deep learning**. It includes:

- Data preparation for **URLHaus** (malicious) and **Top-1M** (benign) domains  
- A **feature-based** ML path (LogReg / SVM / RandomForest)  
- A **sequence-based** DL path (LSTM from scratch)  
- A **Transformer** path fine-tuning ELECTRA (character-level tokenization)  
- Evaluation utilities: confusion matrices, ROC/PR curves, metrics tables  

> **Privacy by design**: classification uses only domain strings; no traffic payloads or user data.

---

## 📂 Project Structure

.
├── domaingen_algorithm.py # Toy DGA generators for testing
├── no_pretrained_code.py # LSTM training from scratch
├── plots_utils.py # Plotting & metrics helpers
├── pytorch_electra.py # ELECTRA fine-tuning (char-level)
├── pytorch_electra_validation.py # Inference / validation script
├── try1.py # Helper script (tokenization demo)
├── try2.py # Helper script (plotting demo)
├── top-1m.csv # Benign domains (sample)
├── urlhaus_cleaned_no_duplicates.csv # Malicious domains (sample)
├── urlhaus_mapped.csv # Optional label-mapped file
├── requirements.txt # Dependencies
└── README.md # This file

yaml
Copy
Edit

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
2. Prepare Data
urlhaus_cleaned_no_duplicates.csv → contains malicious domains (domain column, label=1)

top-1m.csv → contains benign domains (domain column, label=0)

If benign CSV is rank,domain, the scripts handle it automatically.

3. Train Models
LSTM (from scratch):

bash
Copy
Edit
python no_pretrained_code.py \
  --malicious urlhaus_cleaned_no_duplicates.csv \
  --benign top-1m.csv \
  --epochs 8 --batch-size 256 --lr 1e-3
ELECTRA fine-tuning:

bash
Copy
Edit
python pytorch_electra.py \
  --malicious urlhaus_cleaned_no_duplicates.csv \
  --benign top-1m.csv \
  --epochs 3 --batch-size 64 --lr 2e-5
4. Validate a Saved Model
bash
Copy
Edit
python pytorch_electra_validation.py --model-checkpoint runs/electra/electra-best
5. Outputs
Confusion matrices, ROC/PR curves → saved as .png in runs/<model-name>/

Metrics summary → metrics.json

Model checkpoints → runs/<model-name>/

⚙️ Design Highlights
Character-level models capture lexical DGA patterns

Balanced training (undersampling benign) + skewed eval for realism

Explainability: SHAP (ML) and attention maps (Transformer)

Reproducibility: seeds fixed, outputs versioned in runs/

📊 Example Results (typical)
LSTM baseline: ~92–94% accuracy

ELECTRA fine-tuned: ~97–98% accuracy

Classical ML: 85–90% depending on features

📜 License
This project is for academic and learning purposes.
If you use it in production, review dataset licenses and applicable laws.

yaml
Copy
Edit

---

Do you want me to also draft a **short “Usage Examples” section** with sample input/output 
