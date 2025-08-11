import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification

# Configuration
ALEXA_CSV = "top-1m.csv"
URLHAUS_CSV = "urlhaus_cleaned_no_duplicates.csv"
MODEL_DIR = "dga_model"
BENIGN_SAMPLE_SIZE = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
PATIENCE = 2


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


def train_and_save(
    model, tokenizer, X_train, y_train
) -> tf.keras.callbacks.History:
    """
    Train the model with early stopping and save both model and tokenizer.
    """
    train_enc = tokenize_domains(tokenizer, X_train)
    history = model.fit(
        x=train_enc["input_ids"],
        y=np.array(y_train),
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]
    )
    # Save tokenizer and model
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    return history


def plot_learning_curves(history: tf.keras.callbacks.History):
    """
    Plot training and validation loss & accuracy from history.
    """
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], label='Train Acc')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def evaluate(model, tokenizer, X_test, y_test):
    """
    Evaluate the model on the test set and print classification report.
    """
    test_enc = tokenize_domains(tokenizer, X_test)
    logits = model.predict(test_enc["input_ids"]).logits
    y_pred = np.argmax(logits, axis=1)
    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred, target_names=["benign","malicious"]))


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
    if os.path.isdir(MODEL_DIR):
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


if __name__ == "__main__":
    main()
