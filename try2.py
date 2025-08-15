import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from transformers import (
    AutoTokenizer,
    ElectraConfig,
    TFElectraForSequenceClassification
)

# ── CONFIG ──────────────────────────────────────────────────────────────────────
ALEXA_CSV           = "top-1m.csv"
URLHAUS_CSV         = "urlhaus_cleaned_no_duplicates.csv"
MODEL_DIR           = "dga_electra_model"
PLOTS_OUTDIR        = "plots"
BENIGN_SAMPLE_SIZE  = 5000
TEST_SIZE           = 0.2
VAL_SIZE            = 0.1
RANDOM_STATE        = 42
MAX_LENGTH          = 32
BATCH_SIZE          = 16
EPOCHS              = 5
LEARNING_RATE       = 2e-5
PATIENCE            = 2
ELECTRA_CHECKPOINT  = "google/electra-small-discriminator"
DROPOUT_PROB        = 0.3
# ────────────────────────────────────────────────────────────────────────────────

os.makedirs(PLOTS_OUTDIR, exist_ok=True)

def load_alexa(fn: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(
        fn, names=["rank","domain"], header=None, skiprows=1, usecols=[0,1]
    ).iloc[:n].copy()
    df["label"] = 0
    return df[["domain","label"]]

def load_urlhaus(fn: str) -> pd.DataFrame:
    df = pd.read_csv(fn, usecols=["domain"]).rename(columns={"domain":"domain"})
    df["label"] = 1
    return df[["domain","label"]]

def tokenize_domains(tokenizer, texts: list) -> dict:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

def build_model_with_dropout() -> TFElectraForSequenceClassification:
    # 1) load base config, overriding both body & attention dropout
    cfg = ElectraConfig.from_pretrained(
        ELECTRA_CHECKPOINT,
        num_labels=2,
        hidden_dropout_prob=DROPOUT_PROB,
        attention_probs_dropout_prob=DROPOUT_PROB
    )
    # 2) load the pretrained ELECTRA + our modified config
    model = TFElectraForSequenceClassification.from_pretrained(
        ELECTRA_CHECKPOINT,
        config=cfg
    )
    # 3) (optional) freeze the ELECTRA body so only the classification head trains
    model.electra.trainable = False
    # 4) compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def train_and_save(model, tokenizer, X_tr, y_tr, X_val, y_val):
    tr_enc  = tokenize_domains(tokenizer, X_tr)
    val_enc = tokenize_domains(tokenizer, X_val)

    history = model.fit(
        x=dict(tr_enc),  # pass both input_ids and attention_mask
        y=np.array(y_tr),
        validation_data=(dict(val_enc), np.array(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=PATIENCE,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    return history

# ── PLOTS ───────────────────────────────────────────────────────────────────────
def plot_learning_curves(history: tf.keras.callbacks.History):
    epochs = range(1, len(history.history.get("loss", [])) + 1)

    # Loss
    plt.figure()
    if "loss" in history.history:
        plt.plot(epochs, history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "electra_tf_loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    if "accuracy" in history.history:
        plt.plot(epochs, history.history["accuracy"], label="Train Acc")
    if "val_accuracy" in history.history:
        plt.plot(epochs, history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "electra_tf_accuracy.png"))
    plt.close()

def plot_eval_curves(y_true, logits, prefix="ELECTRA_TF"):
    # probs for positive class
    probs = tf.nn.softmax(tf.convert_to_tensor(logits), axis=1).numpy()
    y_prob = probs[:, 1]
    y_pred = np.argmax(logits, axis=1)

    # Save classification report
    report_txt = classification_report(y_true, y_pred, target_names=["benign","malicious"])
    with open(os.path.join(PLOTS_OUTDIR, "electra_tf_classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.xticks([0,1], ["benign","malicious"])
    plt.yticks([0,1], ["benign","malicious"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center')
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.title(f"{prefix} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "electra_tf_confusion_matrix.png"))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} — ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "electra_tf_roc.png"))
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{prefix} — Precision-Recall"); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "electra_tf_pr.png"))
    plt.close()

    return report_txt

# ── EVAL ────────────────────────────────────────────────────────────────────────
def evaluate(model, tokenizer, X_test, y_test):
    test_enc = tokenize_domains(tokenizer, X_test)
    outputs  = model.predict(dict(test_enc), verbose=0)
    logits   = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    report_txt = plot_eval_curves(np.array(y_test), logits, prefix="ELECTRA_TF")
    print("\nTest set classification report:")
    print(report_txt)

def main():
    # 1) load and label
    benign = load_alexa(ALEXA_CSV, BENIGN_SAMPLE_SIZE)
    mal    = load_urlhaus(URLHAUS_CSV)
    df     = pd.concat([benign, mal], ignore_index=True)
    df     = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 2) train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["domain"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )
    # 3) train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    # 4) load or train
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print("🔄 loading existing model…")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model     = TFElectraForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        print("🚀 training new model…")
        tokenizer = AutoTokenizer.from_pretrained(ELECTRA_CHECKPOINT)
        model     = build_model_with_dropout()
        history   = train_and_save(model, tokenizer, X_tr, y_tr, X_val, y_val)
        plot_learning_curves(history)

    # 5) final evaluation (+ plots)
    evaluate(model, tokenizer, X_test, y_test)
    print(f"Saved plots and report to: {os.path.abspath(PLOTS_OUTDIR)}")

if __name__ == "__main__":
    main()
