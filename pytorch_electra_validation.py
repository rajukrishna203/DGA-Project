import os
import platform
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    ElectraConfig,
    EarlyStoppingCallback,
)

# Optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


# ---------------- SPEED/QUALITY CONFIG ----------------
MODEL_NAME        = "google/electra-small-discriminator"  # pretrained Electra backbone
OUTPUT_DIR        = "./electra_finetune"                  # save trained model here
PLOTS_DIR         = "plots"                               # save plots here
TITLE_PREFIX      = "ELECTRA"                             # used in plot titles
NUM_LABELS        = 2                                     # binary classification
MAX_LEN           = 32                                    # max token length for domains
RANDOM_STATE      = 42                                    # reproducibility seed

# Data capping (useful to reduce runtime)
BALANCE_CLASSES   = True          # keep benign & malicious balanced
CAP_PER_CLASS     = 100_000       # cap number of samples per class
USE_FULL_DATA     = False         # set True to train on full dataset

# Training profile
EPOCHS            = 5             # number of training epochs
LR_DEFAULT        = 3e-5          # learning rate
BATCH_DEFAULT     = 32            # batch size
SAVE_TOTAL_LIMIT  = 1             # keep only best checkpoint

# Regularization & scheduler
WEIGHT_DECAY              = 0.01
LABEL_SMOOTHING_FACTOR    = 0.0   # can try >0 for slight smoothing
LR_SCHEDULER_TYPE         = "linear"
WARMUP_RATIO              = 0.06  # warmup fraction

# Optionally freeze some encoder layers (currently off)
N_FREEZE_LAYERS           = 0

# Optuna hyperparameter optimization
USE_OPTUNA         = False
N_TRIALS           = 3
# ------------------------------------------------------


def _bf16_supported():
    """Check if current GPU supports bf16 (Ampere+)."""
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


def build_training_args(num_workers: int):
    """Construct TrainingArguments with proper compatibility and perf knobs."""
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and _bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    common = dict(
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",                 # save model every epoch
        load_best_model_at_end=True,           # reload best checkpoint at end
        metric_for_best_model="f1",            # track f1 for best model
        greater_is_better=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_DEFAULT,
        per_device_eval_batch_size=BATCH_DEFAULT,
        learning_rate=LR_DEFAULT,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",                      # disable W&B/TensorBoard
        dataloader_num_workers=num_workers,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
    )
    try:
        # Some transformers versions use eval_strategy
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        # Older versions use evaluation_strategy
        return TrainingArguments(evaluation_strategy="epoch", **common)


def compute_metrics(eval_pred):
    """Compute accuracy, F1, and ROC-AUC on validation/test sets."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    proba = probs[:, 1]
    acc   = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
    f1    = evaluate.load("f1").compute(predictions=preds, references=labels)["f1"]
    try:
        rocau = roc_auc_score(labels, proba)
    except Exception:
        rocau = float("nan")
    return {"accuracy": acc, "f1": f1, "roc_auc": rocau}


def model_init():
    """Initialize Electra model with classification head."""
    cfg = ElectraConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)

    # Optionally freeze first N encoder layers
    try:
        if N_FREEZE_LAYERS > 0:
            for i, layer in enumerate(model.electra.encoder.layer):
                if i < N_FREEZE_LAYERS:
                    for p in layer.parameters():
                        p.requires_grad = False
    except Exception:
        pass

    return model


def tokenize_fn_factory(tokenizer, max_len):
    """Return a tokenize function bound to the tokenizer and max_len."""
    def tokenize_fn(batch):
        return tokenizer(batch["domain"], padding="max_length", truncation=True, max_length=max_len)
    return tokenize_fn


def save_training_curves(history, outdir=PLOTS_DIR, title=TITLE_PREFIX):
    """Save loss and accuracy training/validation curves as PNGs."""
    os.makedirs(outdir, exist_ok=True)
    # ---- Loss curve ----
    plt.figure()
    if history["train_loss"]: plt.plot(history["train_loss"], label="train")
    if history["val_loss"]:   plt.plot(history["val_loss"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title} Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title.lower()}_loss.png")); plt.close()
    # ---- Accuracy curve ----
    if history["train_acc"] or history["val_acc"]:
        plt.figure()
        if history["train_acc"]: plt.plot(history["train_acc"], label="train")
        if history["val_acc"]:   plt.plot(history["val_acc"],   label="val")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title} Accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title.lower()}_accuracy.png")); plt.close()


def main():
    # ---- Environment hygiene ----
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    # MacOS: avoid multiprocessing for DataLoader
    is_macos = platform.system().lower() == "darwin"
    num_workers = 0 if is_macos else max(2, min(8, os.cpu_count() or 4))

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)
    tokenize_fn = tokenize_fn_factory(tokenizer, MAX_LEN)

    # ---- Load and (optionally) downsample data ----
    benign = pd.read_csv("top-1m.csv", names=["rank", "domain"], usecols=[1])
    benign["label"] = 0
    mal    = pd.read_csv("urlhaus_cleaned_no_duplicates.csv", usecols=["domain"])
    mal["label"] = 1

    if USE_FULL_DATA:
        benign_ds, mal_ds = benign, mal
    else:
        if BALANCE_CLASSES:
            # Ensure balanced benign/malicious, apply cap
            n = min(len(mal), len(benign))
            if CAP_PER_CLASS:
                n = min(n, CAP_PER_CLASS)
            benign_ds = benign.sample(n=n, random_state=RANDOM_STATE)
            mal_ds    = mal.sample(n=n, random_state=RANDOM_STATE) if len(mal) >= n else mal
        else:
            # No balancing, just apply cap on benign
            n = min(len(benign), CAP_PER_CLASS) if CAP_PER_CLASS else len(benign)
            benign_ds = benign.sample(n=n, random_state=RANDOM_STATE)
            mal_ds    = mal

    # Merge and shuffle dataset
    df = pd.concat([benign_ds, mal_ds]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["benign", "malicious"]))
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    val_test = dataset["test"].train_test_split(test_size=0.5, stratify_by_column="label")
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }).map(
        tokenize_fn, batched=True,
        remove_columns=[c for c in dataset["train"].column_names if c not in ["input_ids","attention_mask","label"]]
    )

    # ---- Training arguments ----
    training_args = build_training_args(num_workers)

    # ---- Trainer ----
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,  # future-proof for tokenizer
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0)],  # stop early if no progress
    )

    # ---- Optional Optuna HPO ----
    if USE_OPTUNA and HAS_OPTUNA:
        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
            }
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=N_TRIALS,
        )
        print("Best run:", best_run)
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)
    else:
        print("[INFO] Optuna disabled — training with defaults.")

    # ---- Train ----
    trainer.train()

    # ---- Collect training logs and save curves ----
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

    save_training_curves(history)
    print(f"Saved training curves to '{PLOTS_DIR}/'.")

    # ==== Quick overfitting check ====
    train_eval = trainer.evaluate(dataset["train"])
    val_eval   = trainer.evaluate(dataset["validation"])

    def _fmt(d):
        return {k: round(float(v), 4) for k, v in d.items()
                if k.startswith(("eval_accuracy","eval_f1","eval_roc_auc","eval_loss"))}

    print("TRAIN  :", _fmt(train_eval))
    print("VAL    :", _fmt(val_eval))
    print("GAPS   :", {
        "loss_gap(val-train)": round(float(val_eval["eval_loss"] - train_eval["eval_loss"]), 4),
        "acc_gap(train-val)":  round(float(train_eval["eval_accuracy"] - val_eval["eval_accuracy"]), 4),
        "f1_gap(train-val)":   round(float(train_eval["eval_f1"] - val_eval["eval_f1"]), 4),
    })

    # ---- Evaluate on test set ----
    results = trainer.evaluate(dataset["test"])
    print(results)

    # ---- Test predictions ----
    pred   = trainer.predict(dataset["test"])
    logits = pred.predictions
    labels = pred.label_ids
    probs  = F.softmax(torch.tensor(logits), dim=-1).numpy()
    y_prob = probs[:, 1]
    y_true = labels
    y_pred = np.argmax(logits, axis=-1)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(6,5))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign","Malicious"],
                    yticklabels=["Benign","Malicious"])
    else:
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","Malicious"])
        disp.plot(values_format="d")
        plt.title(f"{TITLE_PREFIX} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{TITLE_PREFIX} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{TITLE_PREFIX.lower()}_confusion_matrix.png"))
    plt.close()

    # ---- ROC Curve ----
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{TITLE_PREFIX} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{TITLE_PREFIX.lower()}_roc.png"))
    plt.close()

    # ---- Precision-Recall Curve ----
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{TITLE_PREFIX} Precision-Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{TITLE_PREFIX.lower()}_pr.png"))
    plt.close()

    print(f"Saved evaluation plots to '{PLOTS_DIR}/'.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    # Entry point — important for multiprocessing safety on macOS
    main()
