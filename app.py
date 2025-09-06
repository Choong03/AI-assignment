import streamlit as st
from sentiment_model import SentimentModel

# Load model
model = SentimentModel("sentiment_dataset.csv")

st.title("💬 Sentiment Chatbox with Module")
st.write("This app censors bad words and predicts sentiment using Naive Bayes.")

# User input
user_input = st.text_input("Enter your message:")

if user_input:
    clean_input = model.censor_bad_words(user_input)
    sentiment = model.predict(user_input)

    st.subheader("🔎 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# Accuracy test
st.subheader("📊 Accuracy Test Results")
results, accuracy = model.test_accuracy()

for r in results:
    st.write(f"Input: `{r['input']}` → Censored: `{r['censored']}` → Predicted: **{r['predicted']}**")

st.write("### ✅ Overall Accuracy:", accuracy)
