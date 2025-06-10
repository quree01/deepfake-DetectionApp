import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import cv2
# Adjust this import based on whether you downgraded moviepy or changed the import
from moviepy.editor import VideoFileClip # Or from moviepy.video.io.VideoFileClip import VideoFileClip
from mtcnn import MTCNN

# --- ABSOLUTE FIRST STREAMLIT COMMAND ---
# This MUST be the first Streamlit command in your entire script.
st.set_page_config(page_title="Deepfake Detection App", layout="wide")
# --- END OF ABSOLUTE FIRST STREAMLIT COMMAND ---

# --- Configuration (can be anywhere after imports) ---
AUDIO_MODEL_PATH = 'final_model.keras'
VIDEO_MODEL_PATH = 'best_model.keras'

# --- Load Models (Cached for performance) ---
# These functions and their calls must come *after* st.set_page_config
@st.cache_resource
def load_audio_model(path):
    # Added print for debugging file path
    print(f"DEBUG: Attempting to load audio model from: {path}")
    model = tf.keras.models.load_model(path)
    return model

@st.cache_resource
def load_video_model(path):
    # Added print for debugging file path
    print(f"DEBUG: Attempting to load video model from: {path}")
    model = tf.keras.models.load_model(path)
    return model

# Initialize models and detector after page config.
# If these lines are causing the issue, it means their internal initialization
# is somehow triggering Streamlit before st.set_page_config() is processed.
audio_model = load_audio_model(AUDIO_MODEL_PATH)
video_model = load_video_model(VIDEO_MODEL_PATH)

# Initialize MTCNN detector
detector = MTCNN() # This needs to be imported and initialized

# --- Preprocessing Functions ---

# Keep preprocess_audio as discussed, you need to confirm its exact requirements
def preprocess_audio(audio_file_path):
    # This function is highly dependent on how YOUR specific audio model was trained.
    # The example below assumes MFCC features, 40 n_mfcc, and a fixed length.
    try:
        y, sr = librosa.load(audio_file_path, sr=16000) # Assuming 16kHz was used for training
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Pad or truncate MFCCs to a fixed length (e.g., 100 frames)
        # You MUST verify the exact sequence length your audio model expects.
        target_mfcc_length = 100
        if mfccs.shape[1] < target_mfcc_length:
            pad_width = target_mfcc_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :target_mfcc_length]

        # Add batch and channel dimensions if your model expects them (e.g., (1, 40, 100, 1))
        # This depends on your model's input shape.
        mfccs = np.expand_dims(mfccs, axis=0) # Add batch dimension
        mfccs = np.expand_dims(mfccs, axis=-1) # Add channel dimension if it's a CNN expecting (batch, height, width, channels)

        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def preprocess_video(video_file_path):
    # Based on the GitHub repo's main.py, it involves face detection and cropping.
    frames_processed = []
    deepfake_predictions_per_frame = [] # To store predictions for each detected face

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, None # Return None for both preprocessed input and predictions

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        # Process every Nth frame to speed things up for web app
        # You might adjust this based on video length and desired speed
        if frame_count % 5 != 0: # Process every 5th frame, adjust as needed
            continue

        # Convert to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, width, height = face['box']
            # Expand bounding box slightly for better face capture
            margin_x = int(0.1 * width)
            margin_y = int(0.1 * height)
            x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
            x2, y2 = min(frame.shape[1], x + width + margin_x), min(frame.shape[0], y + height + margin_y)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0: # Skip empty face images
                continue

            # Resize to the model's expected input size (e.g., 224x224)
            face_img_resized = cv2.resize(face_img, (224, 224))
            # Convert BGR to RGB if your Keras model expects RGB (common)
            face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
            # Normalize pixel values
            face_img_normalized = face_img_rgb / 255.0

            # Add batch dimension
            processed_face = np.expand_dims(face_img_normalized, axis=0)

            # Predict for each detected face
            prediction_score = predict_video_deepfake(video_model, processed_face)
            if prediction_score is not None:
                deepfake_predictions_per_frame.append(prediction_score)
            
            # Optional: If you only care about one face per frame, break
            # break

    cap.release()
    
    if not deepfake_predictions_per_frame:
        st.warning("No faces detected in the video or error during processing.")
        return None, None

    # Aggregate predictions: e.g., take the average or median
    # For a binary classification, you might consider the proportion of "fake" predictions
    # If the majority of detected faces are fake, classify the video as fake.
    # Or, if any face is confidently fake, classify as fake.
    
    # Simple average for now
    avg_prediction = np.mean(deepfake_predictions_per_frame)
    return avg_prediction, deepfake_predictions_per_frame # Return overall prediction and individual frame predictions

# --- Prediction Functions (slightly modified to return the direct score) ---
def predict_audio_deepfake(model, preprocessed_input):
    if preprocessed_input is None:
        return None
    # Ensure input shape matches model.input_shape before prediction
    # You might need to add a check here if shape doesn't match
    prediction = model.predict(preprocessed_input)[0][0] # Assuming binary classification, single output
    return prediction

def predict_video_deepfake(model, preprocessed_input):
    if preprocessed_input is None:
        return None
    # Ensure input shape matches model.input_shape before prediction
    # You might need to add a check here if shape doesn't match
    prediction = model.predict(preprocessed_input)[0][0] # Assuming binary classification, single output
    return prediction

# --- Streamlit UI (modified for video) ---
st.set_page_config(page_title="Deepfake Detection App", layout="wide")
st.title("Deepfake Audio & Video Detector")

st.markdown("""
Upload an audio or video file to check if it's a deepfake.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Audio Deepfake Detection")
    audio_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        st.write("Processing audio...")

        with st.spinner("Saving audio file..."):
            temp_audio_path = "temp_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

        preprocessed_audio_input = preprocess_audio(temp_audio_path)

        if preprocessed_audio_input is not None:
            with st.spinner("Analyzing audio..."):
                audio_prediction = predict_audio_deepfake(audio_model, preprocessed_audio_input)

            if audio_prediction is not None:
                st.subheader("Audio Analysis Result:")
                if audio_prediction > 0.5:
                    st.error(f"**Likely Deepfake Audio!** (Confidence: {audio_prediction:.2f})")
                else:
                    st.success(f"**Likely Real Audio!** (Confidence: {1 - audio_prediction:.2f})")
                st.write(f"Raw Prediction Score: {audio_prediction:.4f}")
            else:
                st.error("Audio prediction failed.")
        else:
            st.error("Audio preprocessing failed. Please check the file.")

with col2:
    st.header("Video Deepfake Detection")
    video_file = st.file_uploader("Upload a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if video_file is not None:
        st.video(video_file)
        st.write("Processing video...")

        with st.spinner("Saving video file..."):
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getbuffer())

        # Call preprocess_video, which now also returns individual predictions
        overall_video_prediction, frame_predictions = preprocess_video(temp_video_path)

        if overall_video_prediction is not None:
            st.subheader("Video Analysis Result:")
            if overall_video_prediction > 0.5:
                st.error(f"**Likely Deepfake Video!** (Overall Confidence: {overall_video_prediction:.2f})")
            else:
                st.success(f"**Likely Real Video!** (Overall Confidence: {1 - overall_video_prediction:.2f})")
            st.write(f"Raw Overall Prediction Score: {overall_video_prediction:.4f}")

            if frame_predictions:
                st.write("Individual frame prediction scores (first 10):", [f"{p:.2f}" for p in frame_predictions[:10]])
                if len(frame_predictions) > 10:
                    st.write("...")
        else:
            st.error("Video processing or prediction failed. Please check the file.")

st.sidebar.header("About This App")
st.sidebar.markdown("""
This application demonstrates the capability of AI models to detect deepfake audio and video.
It utilizes pre-trained Keras models for analysis.
""")

st.sidebar.header("How it Works")
st.sidebar.markdown("""
1.  **Upload:** Users upload an audio or video file.
2.  **Pre-processing:**
    * **Audio:** The app loads the audio, extracts features (like MFCCs), and prepares it for the audio model.
    * **Video:** The app extracts frames, detects faces using MTCNN, crops faces, and prepares them for the video model.
3.  **Prediction:** The respective Keras model predicts the likelihood of the input being a deepfake.
4.  **Result:** The app displays the confidence score and a clear "Likely Deepfake" or "Likely Real" status.
""")

# --- Cleanup (Optional, but good practice for temp files) ---
import os
def cleanup_temp_files():
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    if os.path.exists("temp_video.mp4"):
        os.remove("temp_video.mp4")

# Streamlit reruns the script, so cleanup should be handled carefully.
# For now, files are simply overwritten. For deployment, consider st.session_state
# or specific cleanup routines.