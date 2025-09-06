import streamlit as st
from sentiment_utils import SentimentModel

# Load model
model = SentimentModel("sentiment_dataset.csv")

st.title("💬 Sentiment Chatbox with Learning & Accuracy Test")
st.write("This app censors bad words, predicts sentiment, and learns from user input.")

# User input for prediction
user_input = st.text_input("Enter your message:")
if user_input:
    clean_input = model.censor_bad_words(user_input)
    sentiment = model.predict(user_input)
    st.subheader("🔎 Chatbox Response")
    st.write(f"**Censored Text:** {clean_input}")
    st.write(f"**Predicted Sentiment:** {sentiment}")

# Accuracy based on CSV
st.subheader("📊 Accuracy from CSV Test Split")
st.write("### ✅ Current Accuracy:", round(model.get_accuracy(), 2))

# Add new training example
st.subheader("📝 Teach the Model")
new_text = st.text_input("New training sentence:")
new_label = st.selectbox("Select label:", ["positive", "negative", "neutral"])

if st.button("Add to Training Data"):
    if new_text.strip():
        model.add_training_example(new_text, new_label)
        st.success(f"Added '{new_text}' as {new_label}. Model retrained!")
        st.write("### 🔄 New Accuracy:", round(model.get_accuracy(), 2))
