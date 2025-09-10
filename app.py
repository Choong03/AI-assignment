import streamlit as st
from naive_bayes import train_model, censor_bad_words, predict_sentiment, evaluate_report

# Train model on startup
train_model()

# Streamlit page setup
st.set_page_config(page_title="Sentiment Chatbox (Naive Bayes)", layout="centered")
st.title("💬 Sentiment Chatbox (Naive Bayes)")
st.write("This app censors swear words, predicts sentiment, and shows evaluation metrics.")

# Input box
user_input = st.text_input("Enter your message:")

if user_input:
    clean_input = censor_bad_words(user_input)
    sentiment = predict_sentiment(user_input)

    st.subheader("🔎 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# Accuracy + report
if st.checkbox("Show accuracy and classification report"):
    acc, report = evaluate_report()
    st.write(f"✅ Model Accuracy: {acc:.2%}")
    st.text("Classification Report:\n" + report)
