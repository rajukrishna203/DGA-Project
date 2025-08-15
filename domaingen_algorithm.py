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
BENIGN_PATH = "top-1m.csv"
MALICIOUS_PATH = "urlhaus_cleaned_no_duplicates.csv"
PLOTS_OUTDIR = "plots"
RANDOM_STATE = 42
N_ESTIMATORS = 100
# ---------------------------------------

os.makedirs(PLOTS_OUTDIR, exist_ok=True)

# Step 1: Load datasets and label them
benign = pd.read_csv(BENIGN_PATH, names=["rank", "domain"], usecols=[1])
benign["is_dga"] = 0

malicious = pd.read_csv(MALICIOUS_PATH, usecols=["domain"])
malicious["is_dga"] = 1

# Merge datasets
data = pd.concat([benign, malicious]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Step 2: Feature extraction function
def extract_features(domain: str):
    d = str(domain).lower()
    return {
        'length': len(d),
        'num_digits': sum(c.isdigit() for c in d),
        'num_special_chars': sum(not c.isalnum() for c in d),
        'num_vowels': sum(c in 'aeiou' for c in d),
    }

# Step 3: Extract features for each domain in the dataset
features = data['domain'].apply(extract_features).tolist()
features_df = pd.DataFrame(features)

# Step 4: Prepare the dataset
X = features_df
y = data['is_dga'].astype(int)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Step 6: Train model
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# Step 7: Predictions + metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

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
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest — ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, "rf_roc.png"))
plt.close()

# Precision–Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure()
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest — Precision–Recall")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, "rf_pr.png"))
plt.close()

# Feature Importances
importances = model.feature_importances_
order = np.argsort(importances)[::-1]
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

# Per-feature distributions
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
    feats = extract_features(domain)
    feats_df = pd.DataFrame([feats])
    prediction = model.predict(feats_df)
    return "DGA" if prediction[0] == 1 else "Legitimate"

# Example usage
try:
    new_domain = input("Enter a domain to classify (or press Enter to skip): ").strip()
    if new_domain:
        classification = classify_domain(new_domain)
        print(f"The domain '{new_domain}' is classified as: {classification}")
    else:
        print("Skipped interactive classification.")
except EOFError:
    pass

print(f"All plots and report saved in: {os.path.abspath(PLOTS_OUTDIR)}")
