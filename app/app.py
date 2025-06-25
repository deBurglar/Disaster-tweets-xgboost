import streamlit as st
import pickle
import numpy as np
from utils import preprocess_text

# Load vectorizer and model
# Corrected paths (no ../, no models/)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")

st.title("🌪️ Disaster Tweet Classifier")
st.write("Enter a tweet and the model will predict whether it's about a real disaster or not.")

# Input box
user_input = st.text_area("✍️ Your Tweet:", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet to classify.")
    else:
        # Preprocess and predict
        processed_text = preprocess_text(user_input)
        vectorized_input = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_input)[0]
        proba = model.predict_proba(vectorized_input)[0][prediction]

        if prediction == 1:
            st.success(f"✅ Disaster-related tweet (Confidence: {proba:.2f})")
        else:
            st.info(f"🟢 Not disaster-related (Confidence: {proba:.2f})")
