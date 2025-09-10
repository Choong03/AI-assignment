import streamlit as st
from naive_bayes import train_model, censor_bad_words, predict_sentiment, evaluate_models

# Train model once when app starts
train_model("sentiment_dataset.csv")

st.title("💬 Sentiment Chatbox (NLP Models)")
st.write("This app censors swear words, predicts sentiment, and compares multiple models using Accuracy, Precision, Recall, and F1.")

# User input
user_input = st.text_input("Enter your message:")

if user_input:
    clean_input = censor_bad_words(user_input)
    sentiment = predict_sentiment(user_input)

    st.subheader("🔍 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# Model evaluation metrics (always shown)
st.subheader("📊 Model comparison on dataset split")
try:
    results = evaluate_models()
    # Convert to a simple table
    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df)
except Exception as e:
    st.warning(f"Unable to evaluate models: {e}")
