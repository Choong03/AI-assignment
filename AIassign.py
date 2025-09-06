import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ====================================
# 1. Load dataset from CSV
# ====================================
dataset_path = "sentiment_dataset.csv"  # file must be in same repo folder
df = pd.read_csv(dataset_path)

X = df["text"].tolist()
y = df["label"].tolist()

# ====================================
# 2. Train Naive Bayes
# ====================================
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

clf = MultinomialNB()
clf.fit(X_vec, y)

# ====================================
# 3. Censoring function
# ====================================
bad_words = df[df["label"] == "negative"]["text"].tolist()
bad_word_tokens = [w for phrase in bad_words for w in phrase.split()]

def censor_bad_words(text):
    words = text.split()
    censored = []
    for w in words:
        if w.lower() in bad_word_tokens:
            censored.append("#" * len(w))
        else:
            censored.append(w)
    return " ".join(censored)

# ====================================
# 4. Streamlit UI
# ====================================
st.title("💬 Sentiment Chatbox with Censorship")
st.write("This app censors bad words and predicts sentiment using Naive Bayes.")

# User input box
user_input = st.text_input("Enter your message:")

if user_input:
    clean_input = censor_bad_words(user_input)
    X_input = vectorizer.transform([user_input])
    sentiment = clf.predict(X_input)[0]

    st.subheader("🔎 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# ====================================
# 5. Accuracy Test Section
# ====================================
st.subheader("📊 Accuracy Test Results")

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
    ("this is okay", "neutral")
]

X_test = [text for text, label in test_data]
y_true = [label for text, label in test_data]

X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)

for text, pred in zip(X_test, y_pred):
    st.write(f"Input: `{text}` → Censored: `{censor_bad_words(text)}` → Predicted: **{pred}**")

st.write("### ✅ Overall Accuracy:", accuracy_score(y_true, y_pred))
