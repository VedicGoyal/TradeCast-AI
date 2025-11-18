import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_finbert(model, tokenizer, df, max_len=128):
    texts = df['Sentence'].tolist()
    labels = df['Sentiment'].tolist()
    
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    
    preds = model.predict(encodings)
    logits = preds.logits
    y_pred = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, y_pred)
    prec = precision_score(labels, y_pred)
    rec = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    cm = confusion_matrix(labels, y_pred)
    
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("Confusion Matrix:\n", cm)

    return acc, prec, rec, f1, cm



def load_kaggle_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['Sentence', 'Sentiment']]
    df['Sentiment'] = df['Sentiment'].astype(int)
    return df


def tokenize_texts(texts, tokenizer, max_len=128):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )


def build_tf_dataset(encodings, labels, batch_size=16, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    ))
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.batch(batch_size)


def build_finbert_model():
    model = TFBertForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=2
    )
    return model


def train_model(model, train_ds, val_ds, epochs=3, lr=2e-5):
    steps = len(train_ds)
    optimizer, schedule = create_optimizer(
        init_lr=lr,
        num_warmup_steps=int(0.1 * steps),
        num_train_steps=epochs * steps
    )
    model.compile(
        optimizer=optimizer,
        loss=model.compute_loss,
        metrics=['accuracy']
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return model


def save_model(model, tokenizer, path="./finbert_finetuned"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_finbert_pipeline(csv_path, output_path="./finbert_finetuned"):
    df = load_kaggle_dataset(csv_path)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    print("\n--- Evaluating Base FinBERT Before Training ---")
    base_model = TFBertForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=2
    )
    evaluate_finbert(base_model, tokenizer, val_df)

    train_enc = tokenize_texts(train_df['Sentence'], tokenizer)
    val_enc = tokenize_texts(val_df['Sentence'], tokenizer)

    train_labels = tf.convert_to_tensor(train_df['Sentiment'].values)
    val_labels = tf.convert_to_tensor(val_df['Sentiment'].values)

    train_ds = build_tf_dataset(train_enc, train_labels)
    val_ds = build_tf_dataset(val_enc, val_labels, shuffle=False)

    model = build_finbert_model()

    model = train_model(model, train_ds, val_ds)

    print("\n--- Evaluating Fine-Tuned FinBERT After Training ---")
    evaluate_finbert(model, tokenizer, val_df)

    save_model(model, tokenizer, output_path)

    print("Model saved at:", output_path)



train_finbert_pipeline("Sentiment_Analysis_Dataset.csv", "./finbert_finetuned")
