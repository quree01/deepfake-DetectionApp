# src/utils.py
import os

def cleanup_temp_files(temp_audio_path=None, temp_video_path=None):
    """
    Removes specified temporary audio and video files if their paths are provided.
    """
    print(f"DEBUG: Cleaning up temporary files.")

    if temp_audio_path is not None: # Add this check
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"DEBUG: Removed {temp_audio_path}")
        else:
            print(f"DEBUG: Audio file not found for cleanup: {temp_audio_path}")

    if temp_video_path is not None: # Add this check
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"DEBUG: Removed {temp_video_path}")
        else:
            print(f"DEBUG: Video file not found for cleanup: {temp_video_path}")