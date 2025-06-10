import streamlit as st
import tensorflow as tf
import numpy as np
import librosa # For audio processing
import cv2 # For video processing
from moviepy.editor import VideoFileClip 
from mtcnn import MTCNN # For face detection in video frames
import os # For temporary file handling

if 'page_config_set' not in st.session_state:
    st.set_page_config(page_title="Deepfake Detection App", layout="wide")
    st.session_state['page_config_set'] = True
# --- END OF PAGE CONFIG WORKAROUND ---


# --- Configuration ---
# IMPORTANT: Confirm these paths match the actual location and names of your Keras model files.
# For example, if your models are in a 'clmodels' subfolder, use 'models/best_model.keras'
AUDIO_MODEL_PATH = 'best_model.keras'
VIDEO_MODEL_PATH = 'final_resnet50_deepfake.keras'

# --- Load Models (Cached for performance) ---
# Using st.cache_resource to load models only once across Streamlit reruns
@st.cache_resource
def load_audio_model(path):
    print(f"DEBUG: Attempting to load audio model from: {path}") # Debug print to terminal
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"ERROR: Could not load audio model from {path}: {e}")
        st.error(f"Error loading audio model. Please check the file path: {e}")
        return None

@st.cache_resource
def load_video_model(path):
    print(f"DEBUG: Attempting to load video model from: {path}") # Debug print to terminal
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"ERROR: Could not load video model from {path}: {e}")
        st.error(f"Error loading video model. Please check the file path: {e}")
        return None

# Initialize models and detector AFTER page config, but before preprocessing functions are used
audio_model = load_audio_model(AUDIO_MODEL_PATH)
video_model = load_video_model(VIDEO_MODEL_PATH)

# Initialize MTCNN detector globally for reuse
detector = MTCNN()

# --- Preprocessing Functions ---

def preprocess_audio(audio_file_path):
    """
    Processes an audio file to extract MFCC features for the audio deepfake model.
    Assumes the model expects MFCCs (e.g., 40 n_mfcc, 100 fixed length, with channel dimension).
    """
    try:
        y, sr = librosa.load(audio_file_path, sr=16000) # Assuming 16kHz was used for training
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Pad or truncate MFCCs to a fixed length (e.g., 100 frames)
        target_mfcc_length = 100
        if mfccs.shape[1] < target_mfcc_length:
            pad_width = target_mfcc_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :target_mfcc_length]

        # --- CRITICAL CHANGE HERE ---
        # Current mfccs shape after padding/truncating: (n_mfcc, time_steps) -> (40, 100)
        # Model expects: (batch, time_steps, n_mfcc, channels) -> (None, 100, 40, 1)

        # 1. Transpose the MFCCs to get (time_steps, n_mfcc)
        mfccs = mfccs.T # Now shape is (100, 40)

        # 2. Add batch and channel dimensions
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension: (1, 100, 40)
        mfccs = np.expand_dims(mfccs, axis=-1) # Add channel dimension: (1, 100, 40, 1)

        # --- END CRITICAL CHANGE ---

        # Ensure correct data type (float32 is common for TensorFlow models)
        return mfccs.astype(np.float32)

    except Exception as e:
        print(f"DEBUG: Error processing audio: {e}")
        st.error(f"Error processing audio: {e}")
        return None

def preprocess_video(video_file_path):
    """
    Processes a video file, extracts frames, detects faces, and prepares them
    for the video deepfake model. Aggregates predictions per detected face.
    """
    print(f"DEBUG: Entering preprocess_video for: {video_file_path}")
    deepfake_predictions_per_frame = [] # To store predictions for each detected face

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("DEBUG: Video file could not be opened by OpenCV.")
        st.error("Error: Could not open video file. Please check its format or integrity.")
        return None, None # Return None, None if video cannot be opened

    frame_count = 0
    # Process frames for a maximum of 30 seconds of video to prevent excessive processing time
    # Assuming 30 fps, this means about 900 frames max. Adjust as needed.
    max_frames_to_process = 30 * int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 900

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"DEBUG: End of video or frame read error after {frame_count} frames.")
            break # End of video or error reading frame

        frame_count += 1
        # Process every Nth frame to speed things up for web app. Adjust '5' as needed.
        if frame_count % 5 != 0:
            continue

        if frame_count > max_frames_to_process:
            print(f"DEBUG: Reached maximum frames to process ({max_frames_to_process}). Stopping.")
            break

        try:
            # Convert to RGB (MTCNN expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            faces = detector.detect_faces(rgb_frame)

            if not faces:
                # print(f"DEBUG: No faces detected in frame {frame_count}.") # Can be noisy
                continue # Skip frame if no faces are found

            for face in faces:
                x, y, width, height = face['box']
                # Expand bounding box slightly for better face capture
                margin_x = int(0.1 * width)
                margin_y = int(0.1 * height)
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = min(frame.shape[1], x + width + margin_x), min(frame.shape[0], y + height + margin_y)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0: # Skip empty or invalid face images
                    print(f"DEBUG: Empty face image extracted at frame {frame_count}. Skipping.")
                    continue

                # Resize to the model's expected input size (e.g., 224x224 based on your repo)
                face_img_resized = cv2.resize(face_img, (224, 224))
                # Convert BGR (OpenCV default) to RGB if your Keras model expects RGB (common)
                face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
                # Normalize pixel values to [0, 1] as used in your training
                face_img_normalized = face_img_rgb / 255.0

                # Add batch dimension: (1, 224, 224, 3)
                processed_face = np.expand_dims(face_img_normalized, axis=0)

                # Predict for this single detected face
                prediction_score = predict_video_deepfake(video_model, processed_face)
                if prediction_score is not None:
                    deepfake_predictions_per_frame.append(prediction_score)
                else:
                    print(f"DEBUG: Prediction failed for a face in frame {frame_count}.")

        except Exception as e:
            print(f"DEBUG: Exception during frame processing at frame {frame_count}: {e}")
            st.warning(f"Warning: Error processing frame {frame_count}. Skipping: {e}")
            continue # Continue to next frame even if one frame fails

    cap.release()
    print(f"DEBUG: Finished video frame processing. Total frames read: {frame_count}. Faces analyzed: {len(deepfake_predictions_per_frame)}")

    if not deepfake_predictions_per_frame:
        print("DEBUG: No deepfake predictions collected from any detected face.")
        st.warning("Could not detect any clear faces in the video for analysis, or an error occurred during face processing.")
        return None, None # If no predictions were made, return None, None

    # Aggregate predictions: e.g., take the average of all face predictions
    # You could use other aggregation methods if desired (e.g., median, max, count majority)
    avg_prediction = np.mean(deepfake_predictions_per_frame)
    print(f"DEBUG: Returning overall video prediction: {avg_prediction:.4f}")
    return avg_prediction, deepfake_predictions_per_frame # Return overall prediction and individual frame predictions


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
        # Assuming binary classification, single output from model
        prediction = model.predict(preprocessed_input)[0][0]
        return prediction
    except Exception as e:
        print(f"DEBUG: Error during audio prediction: {e}")
        st.error(f"Error during audio prediction: {e}")
        return None

def predict_video_deepfake(model, preprocessed_input):
    """
    Makes a prediction using the video deepfake model on a single preprocessed face.
    """
    if model is None:
        st.error("Video model not loaded. Cannot make prediction.")
        return None
    if preprocessed_input is None:
        return None
    try:
        # Assuming binary classification, single output from model
        prediction = model.predict(preprocessed_input)[0][0]
        return prediction
    except Exception as e:
        print(f"DEBUG: Error during video prediction: {e}")
        st.error(f"Error during video prediction: {e}")
        return None

# --- Streamlit UI Elements ---
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

        # Save uploaded audio to a temporary file
        temp_audio_path = "temp_audio.wav"
        try:
            with st.spinner("Saving audio file..."):
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())

            preprocessed_audio_input = preprocess_audio(temp_audio_path)

            if preprocessed_audio_input is not None:
                with st.spinner("Analyzing audio..."):
                    audio_prediction = predict_audio_deepfake(audio_model, preprocessed_audio_input)

                if audio_prediction is not None:
                    st.subheader("Audio Analysis Result:")
                    # Assuming prediction is a probability of being deepfake (0 to 1)
                    if audio_prediction > 0.5:
                        st.error(f"**Likely Deepfake Audio!** (Confidence: {audio_prediction:.2f})")
                    else:
                        st.success(f"**Likely Real Audio!** (Confidence: {1 - audio_prediction:.2f})")
                    st.write(f"Raw Prediction Score: {audio_prediction:.4f}")
                else:
                    st.error("Audio prediction failed. Check terminal for details.")
            else:
                st.error("Audio preprocessing failed. Please check the file and terminal for details.")
        finally:
            # Clean up the temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


with col2:
    st.header("Video Deepfake Detection")
    video_file = st.file_uploader("Upload a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if video_file is not None:
        st.video(video_file)
        st.write("Processing video...")

        # Save uploaded video to a temporary file
        temp_video_path = "temp_video.mp4"
        try:
            with st.spinner("Saving video file..."):
                with open(temp_video_path, "wb") as f:
                    f.write(video_file.getbuffer())

            # Call preprocess_video, which returns overall prediction and individual frame predictions
            overall_video_prediction, frame_predictions = preprocess_video(temp_video_path)

            if overall_video_prediction is not None:
                st.subheader("Video Analysis Result:")
                # Assuming prediction is a probability of being deepfake (0 to 1)
                if overall_video_prediction > 0.5:
                    st.error(f"**Likely Deepfake Video!** (Overall Confidence: {overall_video_prediction:.2f})")
                else:
                    st.success(f"**Likely Real Video!** (Overall Confidence: {1 - overall_video_prediction:.2f})")
                st.write(f"Raw Overall Prediction Score: {overall_video_prediction:.4f}")

                if frame_predictions:
                    st.write("Individual face prediction scores (first 10, if available):", [f"{p:.2f}" for p in frame_predictions[:10]])
                    if len(frame_predictions) > 10:
                        st.write("...")
            else:
                st.error("Video processing or prediction failed. Check terminal for details (e.g., no faces detected).")
        finally:
            # Clean up the temporary video file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


# --- Sidebar Information ---
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