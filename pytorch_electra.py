import os
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import matplotlib.pyplot as plt

# Optional seaborn for nicer confusion matrix plots; falls back if missing
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# Hugging Face transformers imports
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    ElectraConfig
)

# Optional Optuna for hyperparameter optimization (script works without it)
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

# Dataset utilities
from datasets import Dataset, DatasetDict, ClassLabel
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# ---------------- CONFIG ----------------
MODEL_NAME   = "google/electra-small-discriminator"  # Pretrained Electra backbone
OUTPUT_DIR   = "./electra_finetune"                  # Where fine-tuned model is saved
PLOTS_OUTDIR = "plots"                               # Save plots here
TITLE_PREFIX = "ELECTRA"                             # Used in plot titles
NUM_LABELS   = 2                                     # Binary classification (benign/malicious)
EPOCHS       = 5                                     # Default epochs if HPO not used
DEFAULT_LR   = 3e-5                                  # Default learning rate
DEFAULT_BS   = 32                                    # Default batch size
MAX_LEN      = 32                                    # Max token length for domain names
# ----------------------------------------

os.makedirs(PLOTS_OUTDIR, exist_ok=True)

# ---- Tokenizer ----
# Loads pretrained Electra tokenizer
tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

# ---- Metrics (HuggingFace evaluate) ----
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

# Function to compute accuracy, F1, and ROC-AUC during training/eval
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)                    # predicted class indices
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()  # softmax → probabilities
    proba = probs[:, 1]                                   # positive class probability
    acc   = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1    = f1_metric.compute(predictions=preds, references=labels)["f1"]
    rocau = roc_auc_score(labels, proba)
    return {"accuracy": acc, "f1": f1, "roc_auc": rocau}

# ---- Model Init ----
# Initializes Electra with classification head (2 labels)
def model_init():
    config = ElectraConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model  = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    for p in model.parameters():
        p.requires_grad = True   # ensure all layers are trainable
    return model

# ---- Tokenization function ----
# Applies padding & truncation to domain strings
def tokenize_fn(batch):
    return tokenizer(batch["domain"], padding="max_length", truncation=True, max_length=MAX_LEN)

# ---- Data Preparation ----
# Load benign dataset (top 1M Alexa) and malicious dataset (URLHaus)
benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
benign["label"] = 0
mal    = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
mal["label"] = 1

# Merge benign + malicious and shuffle
df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
# Ensure labels are treated as categorical (0=benign, 1=malicious)
dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["benign", "malicious"]))
# Train/val/test split (80/10/10)
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
val_test = dataset["test"].train_test_split(test_size=0.5, stratify_by_column="label")
dataset = DatasetDict({
    "train":      dataset["train"],
    "validation": val_test["train"],
    "test":       val_test["test"],
}).map(tokenize_fn, batched=True)   # tokenize all splits

# ---- TrainingArguments ----
# Handles training config (epochs, batch size, logging, eval strategy)
def build_training_args():
    try:  # newer transformers versions
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",    # evaluate every epoch
            save_strategy="epoch",          # save best model each epoch
            load_best_model_at_end=True,    # restore best checkpoint
            metric_for_best_model="f1",
            greater_is_better=True,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=DEFAULT_BS,
            per_device_eval_batch_size=DEFAULT_BS,
            learning_rate=DEFAULT_LR,
            logging_dir="./logs",
            logging_steps=50,
            report_to="none",               # disable wandb/tensorboard
        )
    except TypeError:  # fallback for older versions
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=DEFAULT_BS,
            per_device_eval_batch_size=DEFAULT_BS,
            learning_rate=DEFAULT_LR,
            logging_dir="./logs",
            logging_steps=50,
            report_to="none",
        )

training_args = build_training_args()

# ---- Trainer ----
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ---- Hyperparameter Optimization (Optuna if available) ----
if HAS_OPTUNA:
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7),
        }
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10,
    )
    print("Best run:", best_run)
    for k, v in best_run.hyperparameters.items():
        setattr(trainer.args, k, v)
else:
    print(f"[INFO] Optuna not installed — training with defaults "
          f"(lr={DEFAULT_LR}, batch_size={DEFAULT_BS}, epochs={EPOCHS}).")

# ---- Train model ----
trainer.train()

# ---- Collect training logs & plot curves ----
logs = getattr(trainer.state, "log_history", [])
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
for step in logs:
    if "loss" in step and "epoch" in step and "eval_loss" not in step:
        history["train_loss"].append(step["loss"])
    if "accuracy" in step and "eval_accuracy" not in step:
        history["train_acc"].append(step["accuracy"])
    if "eval_loss" in step:
        history["val_loss"].append(step["eval_loss"])
    if "eval_accuracy" in step:
        history["val_acc"].append(step["eval_accuracy"])

# Plot loss & accuracy curves
def _plot_training_curves(history, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX):
    # ---- Loss curve ----
    plt.figure()
    if history["train_loss"]:
        plt.plot(history["train_loss"], label="train")
    if history["val_loss"]:
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower()}_loss.png"))
    plt.close()

    # ---- Accuracy curve ----
    if history["train_acc"] or history["val_acc"]:
        plt.figure()
        if history["train_acc"]:
            plt.plot(history["train_acc"], label="train")
        if history["val_acc"]:
            plt.plot(history["val_acc"], label="val")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} Accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix.lower()}_accuracy.png"))
        plt.close()

_plot_training_curves(history)
print(f"Saved training curves to '{PLOTS_OUTDIR}/'.")

# ---- Final Evaluation ----
results = trainer.evaluate(dataset["test"])
print(results)

# Get predictions on test set
pred   = trainer.predict(dataset["test"])
logits = pred.predictions
labels = pred.label_ids
probs  = F.softmax(torch.tensor(logits), dim=-1).numpy()
y_prob = probs[:, 1]                    # positive class probabilities
y_true = labels                         # true labels
y_pred = np.argmax(logits, axis=-1)     # predicted labels

# ---- Confusion Matrix ----
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malicious"],
                yticklabels=["Benign", "Malicious"])
else:  # fallback to plain matplotlib
    plt.imshow(cm, interpolation='nearest')
    plt.xticks([0, 1], ["Benign", "Malicious"])
    plt.yticks([0, 1], ["Benign", "Malicious"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'{TITLE_PREFIX} Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, f"{TITLE_PREFIX.lower()}_confusion_matrix.png"))
plt.close()

# ---- ROC Curve ----
roc_auc = roc_auc_score(y_true, y_prob)
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{TITLE_PREFIX} ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, f"{TITLE_PREFIX.lower()}_roc.png"))
plt.close()

# ---- Precision–Recall Curve ----
prec, rec, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure()
plt.plot(rec, prec, label=f'AP = {ap:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title(f'{TITLE_PREFIX} Precision-Recall')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, f"{TITLE_PREFIX.lower()}_pr.png"))
plt.close()

print(f"Saved evaluation plots to '{PLOTS_OUTDIR}/'.")

# ---- Save final model & tokenizer ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
