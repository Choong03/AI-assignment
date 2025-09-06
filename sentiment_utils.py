import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class SentimentModel:
    def __init__(self, dataset_path="sentiment_dataset.csv"):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()
        self._train()

    def _train(self):
        # Train on full dataset
        X = self.df["text"].astype(str).tolist()
        y = self.df["label"].astype(str).tolist()

        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)

        # Evaluate accuracy on same dataset
        y_pred = self.clf.predict(X_vec)
        self.dataset_accuracy = accuracy_score(y, y_pred)

        # Collect bad words from negative class
        bad_words = self.df[self.df["label"] == "negative"]["text"].tolist()
        self.bad_word_tokens = [w.lower() for phrase in bad_words for w in phrase.split()]

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

    def get_accuracy(self) -> float:
        """Return accuracy measured on the dataset itself"""
        return self.dataset_accuracy
