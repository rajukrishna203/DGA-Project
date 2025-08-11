import os
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

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
    proba = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    roc_auc = roc_auc_score(labels, proba)
    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}

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
predictions = trainer.predict(dataset["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=-1)
y_proba = predictions.predictions[:, 1]

# ---- Confusion Matrix ----
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# ---- Plot Confusion Matrix Heatmap ----
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ---- ROC-AUC ----
roc_auc = roc_auc_score(y_true, y_proba)
print("Test ROC-AUC:", roc_auc)

# ---- Plot ROC Curve ----
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# ---- Save final model ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
