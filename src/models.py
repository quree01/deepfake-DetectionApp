# src/models.py
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_audio_model(path):
    """Loads the audio deepfake detection model."""
    print(f"DEBUG: Attempting to load audio model from: {path}")
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"ERROR: Could not load audio model from {path}: {e}")
        st.error(f"Error loading audio model. Please check the file path: {e}")
        return None

@st.cache_resource
def load_video_model(path):
    """Loads the video deepfake detection model."""
    print(f"DEBUG: Attempting to load video model from: {path}")
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"ERROR: Could not load video model from {path}: {e}")
        st.error(f"Error loading video model. Please check the file path: {e}")
        return None