import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Global objects
vectorizer = None
clf = None
X_test = None
y_test = None
swear_words = []


def train_model(dataset_path="sentiment_dataset.csv"):
    """Train Naive Bayes model from dataset of individual words"""
    global vectorizer, clf, X_test, y_test, swear_words

    df = pd.read_csv(dataset_path)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    # Collect swear words (from negative class)
    swear_words = df[df["label"] == "negative"]["text"].str.lower().tolist()

    # Train/test split
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # Save test set for accuracy evaluation
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


def evaluate_accuracy() -> str:
    """Compute accuracy and classification report on dataset test split"""
    global vectorizer, clf, X_test, y_test
    if vectorizer is None or clf is None:
        raise ValueError("Model not trained. Call train_model() first.")

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=2)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)
    return report


# Example usage
if __name__ == "__main__":
    train_model("sentiment_dataset.csv")
    evaluate_accuracy()

    # Try predictions
    sample_text = "I hate this product"
    print("\nOriginal:", sample_text)
    print("Censored:", censor_bad_words(sample_text))
    print("Predicted Sentiment:", predict_sentiment(sample_text))
