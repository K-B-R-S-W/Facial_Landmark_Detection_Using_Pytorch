"""
Video utilities for webcam handling and frame processing
"""
import cv2

def init_webcam(camera_id=0):
    """Initialize webcam capture"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not initialize webcam")
    return cap

def get_frame(cap):
    """Get a frame from the webcam"""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def show_frame(frame, window_name="Facial Landmarks"):
    """Display frame with landmarks"""
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF  # Return key pressed
