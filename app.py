import streamlit as st
from naive_bayes import train_model, censor_bad_words, predict_sentiment, evaluate_accuracy

# Train model once when app starts
train_model("sentiment_dataset.csv")

st.title("💬 Sentiment Chatbox (Naive Bayes)")
st.write("This app censors swear words, predicts sentiment, and checks accuracy against the dataset.")

# User input
user_input = st.text_input("Enter your message:")

if user_input:
    clean_input = censor_bad_words(user_input)
    sentiment = predict_sentiment(user_input)

    st.subheader("🔍 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# Show accuracy (optional)
if st.checkbox("Show accuracy on dataset"):
    acc = evaluate_accuracy()
    st.success(f"✅ Accuracy on dataset: {acc:.2f}")
