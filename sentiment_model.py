import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class SentimentModel:
    def __init__(self, dataset_path="sentiment_dataset.csv"):
        # Load dataset
        df = pd.read_csv(dataset_path)
        self.vectorizer = CountVectorizer()
        self.X = df["text"].astype(str).tolist()
        self.y = df["label"].astype(str).tolist()

        # Train model
        X_vec = self.vectorizer.fit_transform(self.X)
        self.clf = MultinomialNB()
        self.clf.fit(X_vec, self.y)

        # Prepare bad word list
        bad_words = df[df["label"] == "negative"]["text"].tolist()
        self.bad_word_tokens = [w for phrase in bad_words for w in phrase.split()]

    def censor_bad_words(self, text: str) -> str:
        words = text.split()
        censored = []
        for w in words:
            if w.lower() in self.bad_word_tokens:
                censored.append("#" * len(w))
            else:
                censored.append(w)
        return " ".join(censored)

    def predict(self, text: str) -> str:
        X_input = self.vectorizer.transform([text])
        return self.clf.predict(X_input)[0]

    def test_accuracy(self):
        test_data = [
            ("fuck you", "negative"),
            ("shit happens", "negative"),
            ("I am happy", "positive"),
            ("awesome work", "positive"),
            ("idiot person", "negative"),
            ("great job", "positive"),
            ("bitch move", "negative"),
            ("love this", "positive"),
            ("average service", "neutral"),
            ("this is okay", "neutral"),
        ]

        X_test = [text for text, label in test_data]
        y_true = [label for text, label in test_data]
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.clf.predict(X_test_vec)

        results = []
        for text, pred in zip(X_test, y_pred):
            results.append({
                "input": text,
                "censored": self.censor_bad_words(text),
                "predicted": pred
            })

        accuracy = accuracy_score(y_true, y_pred)
        return results, accuracy
