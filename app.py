import streamlit as st
from sentiment_utils import SentimentModel

# Load model
model = SentimentModel("sentiment_dataset.csv")

st.title("💬 Sentiment Chatbox with Dataset Accuracy")
st.write("This app predicts sentiment, censors bad words, and shows accuracy on the dataset.")

# User input for prediction
user_input = st.text_input("Enter your message:")
if user_input:
    clean_input = model.censor_bad_words(user_input)
    sentiment = model.predict(user_input)
    st.subheader("🔎 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# Accuracy based on dataset
st.subheader("📊 Accuracy on Dataset")
st.write("### ✅ Accuracy:", round(model.get_accuracy(), 2))
