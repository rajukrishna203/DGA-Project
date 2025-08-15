import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification

# ---------------- Configuration ----------------
ALEXA_CSV          = "top-1m.csv"
URLHAUS_CSV        = "urlhaus_cleaned_no_duplicates.csv"
MODEL_DIR          = "dga_model"
PLOTS_OUTDIR       = "plots"
BENIGN_SAMPLE_SIZE = 5000
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
MAX_LENGTH         = 32
BATCH_SIZE         = 16
EPOCHS             = 2
LEARNING_RATE      = 2e-5
PATIENCE           = 2
# ------------------------------------------------

os.makedirs(PLOTS_OUTDIR, exist_ok=True)

def load_alexa(filename: str, n_samples: int) -> pd.DataFrame:
    """
    Load the Alexa Top-1M CSV and sample the top n_samples benign domains.
    """
    df = pd.read_csv(
        filename,
        names=["rank", "domain"],
        header=None,
        skiprows=1,
        usecols=[0, 1]
    )
    df = df.iloc[:n_samples].copy()
    df["label"] = 0
    return df[["domain", "label"]]

def load_urlhaus(filename: str) -> pd.DataFrame:
    """
    Load the URLhaus malicious domains.
    """
    df = pd.read_csv(filename, usecols=["domain"]).rename(columns={"domain": "domain"})
    df["label"] = 1
    return df

def tokenize_domains(tokenizer, texts: list) -> dict:
    """
    Tokenize a list of domain strings using the provided tokenizer.
    Returns a dict suitable for TF model inputs (input_ids, attention_mask).
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

def build_model() -> TFDebertaV2ForSequenceClassification:
    """
    Initialize and compile the DeBERTa sequence classification model.
    """
    model = TFDebertaV2ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=2
    )
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def train_and_save(model, tokenizer, X_train, y_train) -> tf.keras.callbacks.History:
    """
    Train the model with early stopping and save both model and tokenizer.
    """
    train_enc = tokenize_domains(tokenizer, X_train)
    history = model.fit(
        x=dict(train_enc),  # pass both input_ids and attention_mask
        y=np.array(y_train),
        validation_split=0.1,
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
    # Save tokenizer and model
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    return history

def plot_learning_curves(history: tf.keras.callbacks.History):
    """
    Save training and validation loss & accuracy plots from history.
    """
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "deberta_loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    if 'accuracy' in history.history:
        plt.plot(epochs, history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        plt.plot(epochs, history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "deberta_accuracy.png"))
    plt.close()

def evaluate(model, tokenizer, X_test, y_test):
    """
    Evaluate the model on the test set, print & save classification report,
    and save confusion matrix, ROC, and PR plots.
    """
    test_enc = tokenize_domains(tokenizer, X_test)
    outputs = model.predict(dict(test_enc), verbose=0)
    # TF Transformers returns a TFSequenceClassifierOutput; with .logits in eager mode
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    y_pred = np.argmax(logits, axis=1)

    # Classification report (print + save)
    report_txt = classification_report(y_test, y_pred, target_names=["benign","malicious"])
    print("\nTest set classification report:")
    print(report_txt)
    with open(os.path.join(PLOTS_OUTDIR, "deberta_classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Probabilities for positive class (malicious=1)
    probs = tf.nn.softmax(tf.convert_to_tensor(logits), axis=1).numpy()
    y_prob = probs[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    # simple heatmap using matplotlib
    plt.imshow(cm, interpolation='nearest')
    plt.xticks([0,1], ["benign", "malicious"])
    plt.yticks([0,1], ["benign", "malicious"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('DeBERTa — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "deberta_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("DeBERTa — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "deberta_roc.png"))
    plt.close()

    # Precision–Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("DeBERTa — Precision–Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTDIR, "deberta_pr.png"))
    plt.close()

def main():
    # Load data
    benign_df = load_alexa(ALEXA_CSV, BENIGN_SAMPLE_SIZE)
    mal_df    = load_urlhaus(URLHAUS_CSV)
    df = pd.concat([benign_df, mal_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['domain'].tolist(),
        df['label'].tolist(),
        test_size=TEST_SIZE,
        stratify=df['label'],
        random_state=RANDOM_STATE
    )

    # Load or train
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print("Loading existing model from", MODEL_DIR)
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)
        model     = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        print("Training new model...")
        tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
        model     = build_model()
        history   = train_and_save(model, tokenizer, X_train, y_train)
        plot_learning_curves(history)

    # Evaluate
    evaluate(model, tokenizer, X_test, y_test)
    print(f"Saved plots and report to: {os.path.abspath(PLOTS_OUTDIR)}")

if __name__ == "__main__":
    main()
