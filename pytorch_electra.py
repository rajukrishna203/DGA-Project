import os
import numpy as np
import torch
import evaluate
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    ElectraConfig
)
import optuna
from datasets import Dataset, DatasetDict, ClassLabel
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_NAME = "google/electra-small-discriminator"
OUTPUT_DIR = "./electra_finetune"
NUM_LABELS = 2
EPOCHS = 5
# ----------------------------------------

# ---- Tokenizer ----
tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

# ---- Metrics ----
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}

# ---- Model Init for Hyperparameter Search ----
def model_init():
    config = ElectraConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    for param in model.parameters():
        param.requires_grad = True  # Full fine-tuning
    return model

# ---- Tokenize function ----
def tokenize_fn(batch):
    return tokenizer(batch["domain"], padding="max_length", truncation=True, max_length=32)

# ---- Load dataset ----
benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
benign["label"] = 0
mal = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
mal["label"] = 1

df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to Dataset with ClassLabel
features = {"domain": df["domain"].dtype, "label": ClassLabel(num_classes=2, names=["benign", "malicious"])}
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["benign", "malicious"]))

# Split with stratification
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
val_test = dataset["test"].train_test_split(test_size=0.5, stratify_by_column="label")
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": val_test["train"],
    "test": val_test["test"]
}).map(tokenize_fn, batched=True)

# ---- Training arguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---- Hyperparameter Search ----
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7)
    }

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=10
)
print("Best run:", best_run)

# ---- Train final model with best params ----
for k, v in best_run.hyperparameters.items():
    setattr(trainer.args, k, v)

trainer.train()

# ---- Evaluate on test ----
results = trainer.evaluate(dataset["test"])
print(results)

# ---- Save final model ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
import os
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    ElectraConfig
)
import optuna
from datasets import Dataset, DatasetDict, ClassLabel
import pandas as pd

# ===================== PLOTS (import or fallback) =====================
# Tries to import shared helpers; if unavailable, defines minimal versions inline.
try:
    from plots_utils import (
        plot_training_curves,
        plot_confusion,
        plot_roc_pr,
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
MODEL_NAME = "google/electra-small-discriminator"
OUTPUT_DIR = "./electra_finetune"
PLOTS_OUTDIR = "plots"
TITLE_PREFIX = "ELECTRA"
NUM_LABELS = 2
EPOCHS = 5
# ----------------------------------------

# ---- Tokenizer ----
tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

# ---- Metrics ----
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}

# ---- Model Init for Hyperparameter Search ----
def model_init():
    config = ElectraConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    for param in model.parameters():
        param.requires_grad = True  # Full fine-tuning
    return model

# ---- Tokenize function ----
def tokenize_fn(batch):
    return tokenizer(batch["domain"], padding="max_length", truncation=True, max_length=32)

# ---- Load dataset ----
benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
benign["label"] = 0
mal = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
mal["label"] = 1

df = pd.concat([benign, mal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to Dataset with ClassLabel
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["benign", "malicious"]))

# Split with stratification
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
val_test = dataset["test"].train_test_split(test_size=0.5, stratify_by_column="label")
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": val_test["train"],
    "test": val_test["test"]
}).map(tokenize_fn, batched=True)

# ---- Training arguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",      # <-- fixed param name (was eval_strategy)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---- Hyperparameter Search ----
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7)
    }

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=10
)
print("Best run:", best_run)

# ---- Train final model with best params ----
for k, v in best_run.hyperparameters.items():
    setattr(trainer.args, k, v)

trainer.train()

# ===================== TRAINING CURVES =====================
# Build history dict from trainer.state.log_history
logs = getattr(trainer.state, "log_history", [])
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for step in logs:
    # training loss & accuracy (when present)
    if "loss" in step and "epoch" in step and "eval_loss" not in step:
        history["train_loss"].append(step["loss"])
    if "accuracy" in step and "eval_accuracy" not in step:
        history["train_acc"].append(step["accuracy"])
    # eval loss & accuracy
    if "eval_loss" in step:
        history["val_loss"].append(step["eval_loss"])
    if "eval_accuracy" in step:
        history["val_acc"].append(step["eval_accuracy"])

try:
    os.makedirs(PLOTS_OUTDIR, exist_ok=True)
    plot_training_curves(history, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX)
    print(f"Saved training curves to '{PLOTS_OUTDIR}/'.")
except Exception as e:
    print(f"[WARN] Could not save training curves: {e}")

# ---- Evaluate on test ----
results = trainer.evaluate(dataset["test"])
print(results)

# ===================== TEST PLOTS: CONFUSION / ROC / PR =====================
# Get logits on the test set to compute class-1 probabilities
pred = trainer.predict(dataset["test"])
logits = pred.predictions  # shape: (N, 2)
labels = pred.label_ids    # shape: (N,)
probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
y_prob = probs[:, 1].tolist()
y_true = labels.tolist()

try:
    plot_confusion(y_true, y_prob, threshold=0.5, outdir=PLOTS_OUTDIR, title=f"{TITLE_PREFIX} Confusion Matrix")
    plot_roc_pr(y_true, y_prob, outdir=PLOTS_OUTDIR, title_prefix=TITLE_PREFIX)
    print(f"Saved evaluation plots to '{PLOTS_OUTDIR}/'.")
except Exception as e:
    print(f"[WARN] Could not save evaluation plots: {e}")

# ---- Save final model ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
