"""Streamlit demo: streamlit run app.py"""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("models/fake_news_model.joblib")

st.title("Fake News Detector")
st.caption("Educational demo. Not a fact-checking authority.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run: python src/train_model.py --data data/news.csv")
    st.stop()

model = joblib.load(MODEL_PATH)
text = st.text_area("Paste a news article", height=250)

if st.button("Classify") and text.strip():
    label = model.predict([text])[0]
    (st.error if label == "FAKE" else st.success)(f"Prediction: {label}")
