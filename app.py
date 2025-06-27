import streamlit as st
import pickle
import string

# 🔹 Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔹 Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# 🔹 Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)
    return "🟢 Real News" if result[0] == 1 else "🔴 Fake News"

# 🔹 Streamlit UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or headline and check if it's real or fake.")

user_input = st.text_area("✍️ Enter news text here:")

if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_news(user_input)
        st.success(f"Prediction: {result}")
