# utils.py
import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

TARGET_SIZE = (200, 200)  # consistent face size

def detect_faces_gray(image_gray):
    """Return list of face bounding boxes (x,y,w,h) in gray image."""
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def extract_face(image_bgr):
    """
    Detect first face and return resized grayscale ROI.
    Returns None if no face found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_gray(gray)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, TARGET_SIZE)
    return roi_resized

def extract_faces_from_gray(gray):
    """Return list of resized gray face ROIs for every detected face."""
    faces_rects = detect_faces_gray(gray)
    rois = []
    for (x, y, w, h) in faces_rects:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, TARGET_SIZE)
        rois.append(roi_resized)
    return rois
