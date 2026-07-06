import re
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).parent
ICONS = {"World": "🌍", "Sports": "🏅", "Business": "💼", "Sci/Tech": "🔬"}

st.set_page_config(page_title="News Category Classifier", page_icon="📰")


@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "agnews_classification_model.keras")
    scaler = pickle.load(open(BASE_DIR / "scaler.pkl", "rb"))
    encoder = pickle.load(open(BASE_DIR / "label_encoder.pkl", "rb"))
    embeddings = pickle.load(open(BASE_DIR / "embedding_lookup.pkl", "rb"))
    return model, scaler, encoder, embeddings


model, scaler, encoder, embeddings = load_artifacts()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def sentence_vector(text):
    words = [w for w in clean_text(text) if w in embeddings]
    if not words:
        return np.zeros(200)
    vectors = np.array([embeddings[w] for w in words])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # matches gensim's get_mean_vector
    return vectors.mean(axis=0)


st.title("📰 News Category Classifier")
st.caption("Paste a headline or a short article and I'll guess where it belongs: World, Sports, Business, or Sci/Tech.")

text = st.text_area("Your text", height=160, placeholder="e.g. The central bank raised interest rates again on Tuesday...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Paste some text first.")
    else:
        vector = scaler.transform([sentence_vector(text)])
        probs = model.predict(vector, verbose=0)[0]
        pred = encoder.inverse_transform([probs.argmax()])[0]

        st.subheader(f"{ICONS.get(pred, '')} {pred}")
        for label, prob in sorted(zip(encoder.classes_, probs), key=lambda p: -p[1]):
            st.progress(float(prob), text=f"{label} — {prob:.0%}")