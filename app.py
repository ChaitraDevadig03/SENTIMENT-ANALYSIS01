import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Fix for loading legacy h5 models with custom objects
def load_legacy_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        # Try loading with custom_objects if needed
        custom_objects = {}
        return load_model(path, custom_objects=custom_objects, compile=False)

# Load separate models with fix
sentiment_model = load_legacy_model("sentiment_model.h5")
platform_model = load_legacy_model("platform_model.h5")

# Load tokenizer and encoders
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
sentiment_encoder = pickle.load(open("sentiment_encoder.pkl", "rb"))
platform_encoder = pickle.load(open("platform_encoder.pkl", "rb"))

MAX_LEN = 100  # same maxlen as training

st.title("Sentiment & Platform Predictor")

text = st.text_area("Enter text:")

def predict_sentiment(padded_sequence):
    preds = sentiment_model.predict(padded_sequence)
    # Use softmax probabilities to get confidence
    probs = tf.nn.softmax(preds).numpy()
    class_idx = np.argmax(probs, axis=1)[0]
    confidence = probs[0][class_idx]
    label = sentiment_encoder.inverse_transform([class_idx])[0]
    return label, confidence

def predict_platform(padded_sequence):
    preds = platform_model.predict(padded_sequence)
    probs = tf.nn.softmax(preds).numpy()
    class_idx = np.argmax(probs, axis=1)[0]
    label = platform_encoder.inverse_transform([class_idx])[0]
    return label

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        try:
            # Preprocess input text
            sequence = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=MAX_LEN)

            sentiment_label, sentiment_confidence = predict_sentiment(padded)
            platform_label = predict_platform(padded)

            st.success(f"Sentiment: {sentiment_label} (Confidence: {sentiment_confidence:.2f})")
            st.info(f"Platform: {platform_label}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
