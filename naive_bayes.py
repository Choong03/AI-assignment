import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Global objects
vectorizer = None
clf = None
X_train_texts = None
y_train_labels = None
X_test = None
y_test = None
swear_words = []


def train_model(dataset_path="sentiment_dataset.csv"):
    """Train Naive Bayes model from dataset of individual words and cache split"""
    global vectorizer, clf, X_train_texts, y_train_labels, X_test, y_test, swear_words

    df = pd.read_csv(dataset_path)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    # Collect swear words (from negative class)
    swear_words = df[df["label"] == "negative"]["text"].str.lower().tolist()

    # Train/test split (stratified for stability)
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train primary model (Naive Bayes)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # Save split for later evaluation
    X_train_texts = X_train
    y_train_labels = y_train
    X_test = X_test_split
    y_test = y_test_split


def censor_bad_words(text: str) -> str:
    """Replace swear words from dataset with ###"""
    words = text.split()
    censored = []
    for w in words:
        if w.lower() in swear_words:
            censored.append("#" * len(w))
        else:
            censored.append(w)
    return " ".join(censored)


def predict_sentiment(text: str) -> str:
    """Predict sentiment of input text"""
    global vectorizer, clf
    if vectorizer is None or clf is None:
        raise ValueError("Model not trained. Call train_model() first.")

    X_input = vectorizer.transform([text])
    return clf.predict(X_input)[0]


def evaluate_models():
    """Train multiple models on cached split and return metrics (Accuracy, Precision, Recall, F1)."""
    global vectorizer, X_train_texts, y_train_labels, X_test, y_test
    if vectorizer is None or X_train_texts is None:
        raise ValueError("Model not trained. Call train_model() first.")

    # Vectorize train and test using the same vectorizer
    X_train_vec = vectorizer.transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test)

    models = [
        ("MultinomialNB", MultinomialNB()),
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("LinearSVC", LinearSVC())
    ]

    results = []
    for name, model in models:
        model.fit(X_train_vec, y_train_labels)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        results.append({
            "model": name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return results
