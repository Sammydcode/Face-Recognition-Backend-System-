# capture_faces.py
import cv2
import os
import time
from utils import extract_face

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def capture_for_student(folder_name, count=20, delay=0.5):
    """
    Launch webcam and capture 'count' face images for folder_name.
    folder_name -> subfolder under dataset (e.g. ENG2023001_samuel)
    """
    folder_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit manually.")
    captured = 0
    while captured < count:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        display = frame.copy()
        face = extract_face(frame)
        if face is not None:
            # show box on display - detect for display
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rects = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")\
                          .detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces_rects:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)

            saved_path = os.path.join(folder_path, f"{captured+1}.jpg")
            cv2.imwrite(saved_path, face)  # save grayscale resized face
            captured += 1
            print(f"[INFO] Captured {captured}/{count}")
            time.sleep(delay)  # short delay to vary expressions/poses

        cv2.imshow("Capture (press q to exit)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {captured} face images to {folder_path}")

if __name__ == "__main__":
    name = input("Enter folder name for student (e.g. ENG2023001_samuel): ").strip()
    n = input("Number of images to capture (default 20): ").strip()
    count = int(n) if n.isdigit() else 20
    capture_for_student(name, count=count)
