# train_model.py
import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import extract_faces_from_gray

DATASET_DIR = "database"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)

# Collect faces and labels
print("[INFO] Loading dataset...")
faces = []
labels = []
label_ids = {}
current_id = 0

for person_folder in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_folder)
    if not os.path.isdir(person_path):
        continue
    label = person_folder  # use folder name as label
    if label not in label_ids:
        label_ids[label] = current_id
        current_id += 1
    id_ = label_ids[label]

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(("jpg","jpeg","png")):
            continue
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # ensure face size consistency (capture_faces already does resizing)
        faces.append(img)
        labels.append(id_)

if len(faces) == 0:
    raise ValueError("No face images found in dataset. Run capture_faces.py first.")

faces = np.array(faces, dtype=object)
labels = np.array(labels)

print(f"[INFO] Found {len(faces)} face images for {len(label_ids)} people.")

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42, stratify=labels)

# Create LBPH recognizer
print("[INFO] Training LBPH recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# OpenCV requires list of arrays for train
recognizer.train(list(X_train), np.array(list(y_train)))

# Evaluation
print("[INFO] Evaluating model...")
y_pred = []
for img in X_test:
    label_pred, conf = recognizer.predict(img)
    y_pred.append(label_pred)

acc = accuracy_score(y_test, y_pred)
print(f"\n=== Evaluation Results ===")
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
# map id -> label name in order of label_ids
id_to_label = {v:k for k,v in label_ids.items()}
target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and labels
recognizer.save(MODEL_PATH)
with open(LABELS_PATH, "wb") as f:
    pickle.dump(label_ids, f)
    
print(f"[INFO] Model saved to {MODEL_PATH}")
print(f"[INFO] Label mapping saved to {LABELS_PATH}")
print("[DONE] Training & evaluation complete.")