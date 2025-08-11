import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    ElectraConfig,
    TFElectraForSequenceClassification
)

# ── CONFIG ──────────────────────────────────────────────────────────────────────
ALEXA_CSV           = "top-1m.csv"
URLHAUS_CSV         = "urlhaus_cleaned_no_duplicates.csv"
MODEL_DIR           = "dga_electra_model"
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

from transformers import ElectraConfig, TFElectraForSequenceClassification

from transformers import ElectraConfig, TFElectraForSequenceClassification

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

    # 3) freeze the ELECTRA body so only the classification head trains
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
        x=tr_enc["input_ids"],
        y=np.array(y_tr),
        validation_data=(val_enc["input_ids"], np.array(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    return history

def plot_learning_curves(history):
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history.history["loss"],     label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val   Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history.history["accuracy"],     label="Train Acc")
    plt.plot(epochs, history.history["val_accuracy"], label="Val   Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.show()

def evaluate(model, tokenizer, X_test, y_test):
    test_enc = tokenize_domains(tokenizer, X_test)
    logits   = model.predict(test_enc["input_ids"]).logits
    y_pred   = np.argmax(logits, axis=1)
    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred, target_names=["benign","malicious"]))

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
    if os.path.isdir(MODEL_DIR):
        print("🔄 loading existing model…")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model     = TFElectraForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        print("🚀 training new model…")
        tokenizer = AutoTokenizer.from_pretrained(ELECTRA_CHECKPOINT)
        model     = build_model_with_dropout()
        history   = train_and_save(model, tokenizer, X_tr, y_tr, X_val, y_val)
        # ─── toggle this plotting on/off ──────────────────────
        # plot_learning_curves(history)

    # 5) final evaluation
    evaluate(model, tokenizer, X_test, y_test)

if __name__ == "__main__":
    main()
