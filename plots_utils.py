# plots_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

def _ensure_dir(d="plots"):
    os.makedirs(d, exist_ok=True)
    return d

def plot_training_curves(history, outdir="plots", title_prefix=""):
    """
    history: dict with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
             each value is a list (one per epoch).
    """
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
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","Malicious"])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title.lower().replace(' ','_')}.png"))
    plt.close()

def plot_roc_pr(y_true, y_prob, outdir="plots", title_prefix="Model"):
    outdir = _ensure_dir(outdir)
    y_true = np.array(y_true); y_prob = np.array(y_prob)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title_prefix} ROC")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_roc.png"))
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_pr.png"))
    plt.close()

def plot_domain_distributions(domains, outdir="plots", title_prefix="Domains"):
    """
    domains: list/iterable of raw domain strings (no labels).
    Makes two plots: length distribution & character histogram.
    """
    outdir = _ensure_dir(outdir)

    # Length distribution
    lens = [len(d) for d in domains]
    plt.figure()
    plt.hist(lens, bins=50)
    plt.xlabel("Length"); plt.ylabel("Count")
    plt.title(f"{title_prefix} Length Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_length_dist.png"))
    plt.close()

    # Character histogram
    from collections import Counter
    c = Counter("".join(domains))
    chars, counts = zip(*sorted(c.items(), key=lambda x: x[0]))
    plt.figure()
    plt.bar(chars, counts)
    plt.xlabel("Character"); plt.ylabel("Frequency")
    plt.title(f"{title_prefix} Character Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_char_freq.png"))
    plt.close()
