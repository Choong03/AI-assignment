import streamlit as st
import pandas as pd
import naive_bayes as nb

train_model = nb.train_model
censor_bad_words = nb.censor_bad_words
predict_sentiment = nb.predict_sentiment
evaluate_models = getattr(nb, "evaluate_models", None)

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
if evaluate_models is None:
    st.warning("Model comparison not available in the current server version. Restart or redeploy to load latest code.")
else:
    try:
        results = evaluate_models()
        df = pd.DataFrame(results)
        st.dataframe(df)
    except Exception as e:
        st.warning(f"Unable to evaluate models: {e}")
