import os
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import matplotlib.pyplot as plt

# Optional seaborn (pretty heatmap); will fall back to matplotlib if missing
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    ElectraConfig
)

# Optional Optuna (HPO); script runs fine without it
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

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
MODEL_NAME   = "google/electra-small-discriminator"
OUTPUT_DIR   = "./electra_finetune"
PLOTS_OUTDIR = "plots"
TITLE_PREFIX = "ELECTRA"
NUM_LABELS   = 2
EPOCHS       = 5         # used if HPO is disabled
DEFAULT_LR   = 3e-5      # used if HPO is disabled
DEFAULT_BS   = 32
MAX_LEN      = 32
# ----------------------------------------

os.makedirs(PLOTS_OUTDIR, exist_ok=True)

# ---- Tokenizer ----
tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

# ---- Metrics ----
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    proba = probs[:, 1]
    acc   = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1    = f1_metric.compute(predictions=preds, references=labels)["f1"]
    rocau = roc_auc_score(labels, proba)
    return {"accuracy": acc, "f1": f1, "roc_auc": rocau}

# ---- Model Init ----
def model_init():
    config = ElectraConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model  = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    for p in model.parameters():
        p.requires_grad = True
    return model

# ---- Tokenize ----
def tokenize_fn(batch):
    return tokenizer(batch["domain"], padding="max_length", truncation=True, max_length=MAX_LEN)

# ---- Data ----
benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
benign["label"] = 0
mal    = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
mal["label"] = 1

df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["benign", "malicious"]))
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
val_test = dataset["test"].train_test_split(test_size=0.5, stratify_by_column="label")
dataset = DatasetDict({
    "train":      dataset["train"],
    "validation": val_test["train"],
    "test":       val_test["test"],
}).map(tokenize_fn, batched=True)

# ---- TrainingArguments (version-compatible) ----
def build_training_args():
    # Try new API (evaluation_strategy); if it fails, fallback to eval_strategy (older transformers)
    try:
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",
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
    except TypeError:
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

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ---- Optional HPO ----
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

# ---- Train ----
trainer.train()

# ---- Curves (loss/acc) ----
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

def _plot_training_curves(history, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX):
    # Loss curve
    plt.figure()
    if history["train_loss"]:
        plt.plot(history["train_loss"], label="train")
    if history["val_loss"]:
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower()}_loss.png"))
    plt.close()

    # Accuracy curve
    if history["train_acc"] or history["val_acc"]:
        plt.figure()
        if history["train_acc"]:
            plt.plot(history["train_acc"], label="train")
        if history["val_acc"]:
            plt.plot(history["val_acc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix.lower()}_accuracy.png"))
        plt.close()

_plot_training_curves(history)
print(f"Saved training curves to '{PLOTS_OUTDIR}/'.")

# ---- Evaluate & plots ----
results = trainer.evaluate(dataset["test"])
print(results)

pred   = trainer.predict(dataset["test"])
logits = pred.predictions
labels = pred.label_ids
probs  = F.softmax(torch.tensor(logits), dim=-1).numpy()
y_prob = probs[:, 1]
y_true = labels
y_pred = np.argmax(logits, axis=-1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malicious"],
                yticklabels=["Benign", "Malicious"])
else:
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

# ROC
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

# Precision–Recall
prec, rec, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure()
plt.plot(rec, prec, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'{TITLE_PREFIX} Precision-Recall')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUTDIR, f"{TITLE_PREFIX.lower()}_pr.png"))
plt.close()

print(f"Saved evaluation plots to '{PLOTS_OUTDIR}/'.")

# ---- Save model ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
