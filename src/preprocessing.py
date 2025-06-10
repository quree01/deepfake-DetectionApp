# src/preprocessing.py
import numpy as np
import librosa
import cv2
from mtcnn import MTCNN # MTCNN detector is used here

# Initialize MTCNN detector globally in this module for reuse
# It's instantiated once when this module is imported.
detector = MTCNN()

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

        # Transpose MFCCs to get (time_steps, n_mfcc)
        mfccs = mfccs.T # Now shape is (100, 40)

        # Add batch and channel dimensions: (1, 100, 40, 1)
        mfccs = np.expand_dims(mfccs, axis=0)
        mfccs = np.expand_dims(mfccs, axis=-1)

        return mfccs.astype(np.float32)

    except Exception as e:
        print(f"DEBUG: Error processing audio: {e}")
        # In a real app, you might want to log this or raise a custom exception
        return None

def preprocess_video(video_file_path):
    """
    Processes a video file, extracts frames, detects faces, and prepares them
    for the video deepfake model. Aggregates predictions per detected face.
    """
    print(f"DEBUG: Entering preprocess_video for: {video_file_path}")
    deepfake_predictions_per_frame = []

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("DEBUG: Video file could not be opened by OpenCV.")
        # Streamlit errors should be handled in app.py or raised from here
        return None, None

    frame_count = 0
    # Process frames for a maximum of 30 seconds of video to prevent excessive processing time
    max_frames_to_process = 30 * int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 900

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"DEBUG: End of video or frame read error after {frame_count} frames.")
            break

        frame_count += 1
        if frame_count % 5 != 0: # Process every 5th frame
            continue

        if frame_count > max_frames_to_process:
            print(f"DEBUG: Reached maximum frames to process ({max_frames_to_process}). Stopping.")
            break

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)

            if not faces:
                continue

            for face in faces:
                x, y, width, height = face['box']
                # Expand bounding box slightly
                margin_x = int(0.1 * width)
                margin_y = int(0.1 * height)
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = min(frame.shape[1], x + width + margin_x), min(frame.shape[0], y + height + margin_y)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    print(f"DEBUG: Empty face image extracted at frame {frame_count}. Skipping.")
                    continue

                face_img_resized = cv2.resize(face_img, (224, 224)) # Resize to model's input size
                face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
                face_img_normalized = face_img_rgb / 255.0 # Normalize

                processed_face = np.expand_dims(face_img_normalized, axis=0) # Add batch dimension
                deepfake_predictions_per_frame.append(processed_face)

        except Exception as e:
            print(f"DEBUG: Exception during frame processing at frame {frame_count}: {e}")
            # Do not use st.warning here as it's a module level function
            continue

    cap.release()
    print(f"DEBUG: Finished video frame processing. Total frames read: {frame_count}. Faces analyzed: {len(deepfake_predictions_per_frame)}")

    if not deepfake_predictions_per_frame:
        print("DEBUG: No deepfake predictions collected from any detected face.")
        return None, None # If no predictions were made, return None, None

    # This function should only return preprocessed frames, not predictions
    # Predictions will be made in src/prediction.py
    return deepfake_predictions_per_frame # Return a list of preprocessed faces