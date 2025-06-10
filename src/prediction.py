# src/prediction.py
import streamlit as st
import numpy as np
import tensorflow as tf

def predict_audio_deepfake(model, preprocessed_input):
    """
    Makes a prediction using the audio deepfake model.
    """
    if model is None:
        st.error("Audio model not loaded. Cannot make prediction.")
        return None
    if preprocessed_input is None:
        return None
    try:
        prediction = model.predict(preprocessed_input)[0][0]
        return prediction
    except Exception as e:
        print(f"DEBUG: Error during audio prediction: {e}")
        st.error(f"Error during audio prediction: {e}")
        return None

def predict_video_deepfake(model, preprocessed_faces_list):
    """
    Makes predictions for a list of preprocessed faces from a video using the video deepfake model
    and aggregates the results.
    """
    if model is None:
        st.error("Video model not loaded. Cannot make prediction.")
        return None
    if not preprocessed_faces_list: # If no faces were passed
        return None, None

    predictions = []
    try:
        # Predict on all preprocessed faces at once if possible (batch prediction)
        # If your model can handle a batch of (N, 224, 224, 3)
        batched_input = np.vstack(preprocessed_faces_list) # Stack list of (1, H, W, C) to (N, H, W, C)
        raw_predictions = model.predict(batched_input)

        # raw_predictions will be shape (N, 1) or (N,) depending on model output
        predictions = raw_predictions.flatten().tolist() # Convert to flat list of floats

        if not predictions:
            return None, None

        overall_prediction = np.mean(predictions) # Aggregate by taking the mean
        return overall_prediction, predictions # Return aggregated and per-face predictions
    except Exception as e:
        print(f"DEBUG: Error during video prediction (batch): {e}")
        st.error(f"Error during video prediction: {e}")
        return None, None