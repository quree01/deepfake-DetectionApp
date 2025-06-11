# app.py
import streamlit as st
import os

# Import functions from your newly created modules
from src.models import load_audio_model, load_video_model
from src.preprocessing import preprocess_audio, preprocess_video
from src.prediction import predict_audio_deepfake, predict_video_deepfake
from src.utils import cleanup_temp_files

import src.custom_losses
# --- Streamlit Page Configuration (MUST be the absolute first Streamlit command) ---
if 'page_config_set' not in st.session_state:
    st.set_page_config(page_title="Deepfake Detection App", layout="wide")
    st.session_state['page_config_set'] = True
# --- END OF PAGE CONFIG WORKAROUND ---


# --- Configuration ---
AUDIO_MODEL_PATH = 'models/best_model.keras'
VIDEO_MODEL_PATH = 'models/final_faceforensics_resnet50.keras'
TEMP_FILES_DIR = 'temp_files'
os.makedirs(TEMP_FILES_DIR, exist_ok=True) # Ensure the temp directory exists

# --- Load Models (Cached for performance) ---
audio_model = load_audio_model(AUDIO_MODEL_PATH)
video_model = load_video_model(VIDEO_MODEL_PATH)

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

        temp_audio_path = os.path.join(TEMP_FILES_DIR, "temp_audio.wav")
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
            cleanup_temp_files(temp_audio_path=temp_audio_path, temp_video_path=None)


with col2:
    st.header("Video Deepfake Detection")
    video_file = st.file_uploader("Upload a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if video_file is not None:
        

        original_temp_path = os.path.join(TEMP_FILES_DIR, "temp_video_original.mp4")
        playable_temp_path = os.path.join(TEMP_FILES_DIR, "temp_video_playable.mp4")

        # Initialize these outside the try block so finally can access them
        # They will only exist if saving and transcoding were successful,
        # but their paths will always be defined.
        temp_video_path_to_cleanup = None
        playable_video_path_to_cleanup = None

        try:
            # 1. Save original video
            with st.spinner("Saving original video file..."):
                with open(original_temp_path, "wb") as f:
                    f.write(video_file.getbuffer())
                temp_video_path_to_cleanup = original_temp_path # Assign for cleanup

            st.write(f"DEBUG: Original video saved to: {original_temp_path}")
            if os.path.exists(original_temp_path):
                st.write(f"DEBUG: Original file exists. Size: {os.path.getsize(original_temp_path) / (1024*1024):.2f} MB")
            else:
                st.error("DEBUG: Original video file not found on disk after saving!")
                # If file not found, we can't proceed. Raise an exception or just let the outer try catch it.
                raise FileNotFoundError("Original video file could not be saved or found.")


            # 2. Convert video for playback compatibility
            with st.spinner("Converting video for playback compatibility (this may take a moment)..."):
                try:
                    from moviepy.editor import VideoFileClip # Import here for local scope if not global
                    video_clip = VideoFileClip(original_temp_path)
                    video_clip.write_videofile(playable_temp_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                    video_clip.close()
                    playable_video_path_to_cleanup = playable_temp_path # Assign for cleanup
                    st.success("Video converted successfully for playback.")
                    st.video(playable_temp_path) # Play the newly transcoded file!
                except Exception as e:
                    st.error(f"Error during video conversion for playback: {e}. Attempting to play original file instead.")
                    print(f"DEBUG: Video conversion failed: {e}")
                    # Fallback to playing the original if conversion fails
                    if os.path.exists(original_temp_path):
                        st.video(original_temp_path)
                    else:
                        st.error("Original video file not found for fallback playback.")
                    # Don't raise here, allow prediction part to run even if playback conversion fails

            st.write("Processing video for deepfake detection...")

            # 3. Preprocess video for model. Use the ORIGINAL video file for model input.
            preprocessed_faces_list = preprocess_video(original_temp_path)

            if preprocessed_faces_list is not None and preprocessed_faces_list:
                with st.spinner("Analyzing video..."):
                    overall_video_prediction, frame_predictions = predict_video_deepfake(video_model, preprocessed_faces_list)

                if overall_video_prediction is not None:
                    st.subheader("Video Analysis Result:")
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
                    st.error("Video prediction failed. Check terminal for details.")
            else:
                st.error("Video preprocessing failed. Could not detect any clear faces in the video for analysis, or an error occurred during face processing. Please check the file and terminal for details.")

        except Exception as e: # This broad except catches any unhandled errors from above steps
            st.error(f"An unexpected error occurred during video processing: {e}")
            print(f"ERROR: Video processing failed: {e}")

        finally:
            # Clean up both original and playable temp files
            cleanup_temp_files(temp_audio_path=None, temp_video_path=temp_video_path_to_cleanup)
            if playable_video_path_to_cleanup and os.path.exists(playable_video_path_to_cleanup):
                os.remove(playable_video_path_to_cleanup)
                print(f"DEBUG: Removed {playable_video_path_to_cleanup}")


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