import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SentimentModel:
    def __init__(self, dataset_path="sentiment_dataset.csv"):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()
        self._train()

    def _train(self):
        # Split dataset into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.df["text"].astype(str), 
            self.df["label"].astype(str), 
            test_size=0.2, 
            random_state=42
        )

        # Train model
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_train_vec, y_train)

        # Evaluate accuracy
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.clf.predict(X_test_vec)
        self.test_accuracy_score = accuracy_score(y_test, y_pred)

        # Bad words from negative samples
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

    def add_training_example(self, text: str, label: str):
        """Add new training example and retrain"""
        new_row = pd.DataFrame([[text, label]], columns=["text", "label"])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.dataset_path, index=False)  # save back to CSV
        self._train()

    def get_accuracy(self) -> float:
        return self.test_accuracy_score
