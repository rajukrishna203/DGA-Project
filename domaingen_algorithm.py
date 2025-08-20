import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BENIGN_PATH = "top-1m.csv"                       # path to benign domains (Tranco/Alexa top list)
MALICIOUS_PATH = "urlhaus_cleaned_no_duplicates.csv"  # path to malicious domains dataset
PLOTS_OUTDIR = "plots"                           # folder to save plots/reports
RANDOM_STATE = 42                                # fixed seed for reproducibility
N_ESTIMATORS = 100                               # number of trees in Random Forest
# ---------------------------------------

os.makedirs(PLOTS_OUTDIR, exist_ok=True)         # create plots folder if it doesn't exist

# Step 1: Load datasets and label them
benign = pd.read_csv(BENIGN_PATH, names=["rank", "domain"], usecols=[1])  # load benign domains
benign["is_dga"] = 0   # label = 0 → legitimate

malicious = pd.read_csv(MALICIOUS_PATH, usecols=["domain"])  # load malicious domains
malicious["is_dga"] = 1  # label = 1 → DGA

# Merge datasets and shuffle
data = pd.concat([benign, malicious]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Step 2: Feature extraction function
def extract_features(domain: str):
    """Extract simple lexical features from a domain string."""
    d = str(domain).lower()
    return {
        'length': len(d),                                   # total length of domain
        'num_digits': sum(c.isdigit() for c in d),          # count of digits
        'num_special_chars': sum(not c.isalnum() for c in d), # count of non-alphanumeric chars
        'num_vowels': sum(c in 'aeiou' for c in d),         # count of vowels
    }

# Step 3: Extract features for each domain
features = data['domain'].apply(extract_features).tolist()
features_df = pd.DataFrame(features)   # convert list of dicts → dataframe

# Step 4: Prepare the dataset
X = features_df                        # feature matrix
y = data['is_dga'].astype(int)         # labels (0=benign, 1=dga)

# Step 5: Split dataset into train and test sets (80/20 split, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Step 6: Train Random Forest model
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# Step 7: Predictions + metrics
y_pred = model.predict(X_test)                  # predicted labels
y_proba = model.predict_proba(X_test)[:, 1]     # predicted probabilities for class=1 (DGA)

# Print and save classification report
report = classification_report(y_test, y_pred, target_names=["Legitimate", "DGA"])
print(report)
with open(os.path.join(PLOTS_OUTDIR, "rf_classification_report.txt"), "w") as f:
    f.write(report)

# ---------------- PLOTS ----------------

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "DGA"])
plt.figure(figsize=(6, 5))
disp.plot(values_format="d")
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, "rf_confusion_matrix.png"))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)   # area under ROC
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")   # diagonal line (random baseline)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest — ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, "rf_roc.png"))
plt.close()

# Precision–Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)   # area under PR curve
plt.figure()
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest — Precision–Recall")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, "rf_pr.png"))
plt.close()

# Feature Importances (which features matter most in RF)
importances = model.feature_importances_
order = np.argsort(importances)[::-1]   # sort descending
feat_names = X.columns.to_numpy()[order]
feat_vals = importances[order]
plt.figure()
plt.bar(range(len(feat_vals)), feat_vals)
plt.xticks(range(len(feat_vals)), feat_names, rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, "rf_feature_importances.png"))
plt.close()

# Per-feature distributions (to visualize separation of benign vs dga per feature)
full_feats = X.copy()
full_feats["is_dga"] = y.to_numpy()
for col in X.columns:
    plt.figure()
    legit_vals = full_feats.loc[full_feats["is_dga"] == 0, col].to_numpy()
    dga_vals   = full_feats.loc[full_feats["is_dga"] == 1, col].to_numpy()
    bins = 30
    plt.hist(legit_vals, bins=bins, alpha=0.6, label="Legitimate")
    plt.hist(dga_vals, bins=bins, alpha=0.6, label="DGA")
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel("Count")
    plt.title(f"Distribution — {col.replace('_', ' ').title()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, f"dist_{col}.png"))
    plt.close()

# Step 8: Function to classify a new domain
def classify_domain(domain: str):
    """Classify a new domain string using the trained Random Forest model."""
    feats = extract_features(domain)
    feats_df = pd.DataFrame([feats])
    prediction = model.predict(feats_df)
    return "DGA" if prediction[0] == 1 else "Legitimate"

# Example usage (interactive input)
try:
    new_domain = input("Enter a domain to classify (or press Enter to skip): ").strip()
    if new_domain:
        classification = classify_domain(new_domain)
        print(f"The domain '{new_domain}' is classified as: {classification}")
    else:
        print("Skipped interactive classification.")
except EOFError:
    # Handles cases where input() is not available (e.g., non-interactive runs)
    pass

print(f"All plots and report saved in: {os.path.abspath(PLOTS_OUTDIR)}")
