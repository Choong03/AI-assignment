import pandas as pd
import random
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ====================================
# 1. Generate Synthetic Sentiment Dataset
# ====================================
positive_words = ["good", "amazing", "excellent", "fantastic", "love", "happy", "great", "awesome", "wonderful", "enjoy"]
negative_words = ["bad", "terrible", "awful", "hate", "worst", "angry", "horrible", "sad", "disappointing", "poor"]
neutral_words = ["service", "product", "experience", "delivery", "support", "quality", "price", "system", "app", "feature"]

# Expanded swear words list
swear_words_list = [
    "fuck","shit","bitch","bastard","asshole","damn","dick","pussy","cock","cunt","faggot","slut","whore","douche",
    "bollocks","bugger","prick","wanker","twat","arse","jerk","moron","idiot","retard","piss","suck","jackass","motherfucker",
    "hell","bloody","arsehole","dipshit","scumbag","knob","tosser","git","chav","skank","tramp","hag","hoe",
    "jerkwad","fuckface","fuckhead","shithead","butthead","numbnuts","airhead","bonehead","blockhead","loser","dummy",
    "weirdo","freak","nerd","geek","stupid","dumb","lame","trash","garbage","worthless","useless","lazy","slacker","nigger"
]
swear_words_set = set(w.lower() for w in swear_words_list)

def generate_sentence(sentiment="positive"):
    length = random.randint(5, 12)
    sentence = []
    if sentiment == "positive":
        sentence.append(random.choice(["I", "We"]))
        sentence.append(random.choice(["love", "enjoy", "like", "appreciate"]))
        sentence.append(random.choice(neutral_words))
        sentence.extend(random.choices(positive_words, k=length-3))
    elif sentiment == "negative":
        sentence.append(random.choice(["I", "We"]))
        sentence.append(random.choice(["hate", "dislike", "cannot stand", "am upset with"]))
        sentence.append(random.choice(neutral_words))
        # Occasionally add swear words into negative samples
        if random.random() < 0.3:
            sentence.append(random.choice(list(swear_words_set)))
        sentence.extend(random.choices(negative_words, k=length-3))
    else:  # neutral
        sentence.append(random.choice(["The", "This"]))
        sentence.append(random.choice(neutral_words))
        sentence.append("is")
        sentence.append(random.choice(["okay", "normal", "average", "acceptable"]))
        sentence.extend(random.choices(neutral_words, k=length-4))
    return " ".join(sentence)

# Create dataset
data = []
for _ in range(10000):  # 10k samples
    r = random.random()
    if r < 0.33:
        data.append([generate_sentence("positive"), "positive"])
    elif r < 0.66:
        data.append([generate_sentence("negative"), "negative"])
    else:
        data.append([generate_sentence("neutral"), "neutral"])

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("synthetic_sentiment_dataset.csv", index=False)
print("✅ Dataset created with", len(df), "rows and saved as synthetic_sentiment_dataset.csv")

# ====================================
# 2. Train Naive Bayes (Bag-of-Words + TF-IDF)
# ====================================
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

text_clf = Pipeline([
    ("vect", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("clf", MultinomialNB()),
])

text_clf.fit(X_train, y_train)

# ====================================
# 3. Function to censor swear words
# ====================================
def censor_bad_words(text):
    words = text.split()
    censored = []
    for w in words:
        if w.lower() in swear_words_set:
            censored.append("#" * len(w))
        else:
            censored.append(w)
    return " ".join(censored)

# ====================================
# 4. Chatbox Loop (Hybrid AI + Rule-based)
# ====================================
print("=== Chatbox Sentiment Analyzer (type 'exit' to quit) ===")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Step A: censor swear words
    clean_input = censor_bad_words(user_input)
    
    # Step B: predict sentiment with Naive Bayes
    sentiment = text_clf.predict([user_input])[0]
    
    # Rule: if a swear word is found, force negative sentiment
    if any(word.lower() in swear_words_set for word in user_input.split()):
        sentiment = "negative"
    
    print(f"Chatbox: {clean_input}")

    print(f"[ Sentiment: {sentiment} ]\n")
