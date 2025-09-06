import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ====================================
# 1. Load dataset
# ====================================
dataset_path = "sentiment_dataset.csv"
df = pd.read_csv(dataset_path)

X = df["text"].astype(str).tolist()
y = df["label"].astype(str).tolist()

# Train/test split for accuracy testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================
# 2. Train Naive Bayes
# ====================================
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Accuracy on test set
X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)
dataset_accuracy = accuracy_score(y_test, y_pred)

# ====================================
# 3. Censoring function (swear words only)
# ====================================
swear_words = [
    "fuck", "shit", "bitch", "bastard", "asshole",
    "idiot", "stupid", "dumb", "crap", "slut", "whore"
]

def censor_bad_words(text):
    words = text.split()
    censored = []
    for w in words:
        if w.lower() in swear_words:
            censored.append("#" * len(w))
        else:
            censored.append(w)
    return " ".join(censored)

# ====================================
# 4. Streamlit UI
# ====================================
st.title("💬 Sentiment Chatbox with Dataset Accuracy")
st.write("This app predicts sentiment, censors swear words, and shows accuracy on the dataset.")

# User input
user_input = st.text_input("Enter your message:")

if user_input:
    clean_input = censor_bad_words(user_input)
    X_input = vectorizer.transform([user_input])
    sentiment = clf.predict(X_input)[0]

    st.subheader("🔍 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# ====================================
# 5. Accuracy Section
# ====================================
st.subheader("📊 Accuracy on Dataset")
st.success(f"✅ Accuracy: {dataset_accuracy:.2f}")
