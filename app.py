import os
import io
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify
from utils import extract_faces_from_gray
from database import init_db, add_student, log_attendance_by_folder, get_attendance_report, get_student_by_folder
from datetime import datetime

# Config
MODEL_PATH = "models/face_recognizer.yml"
LABELS_PATH = "models/labels.pkl"
CONFIDENCE_THRESHOLD = 70  # lower is more confident; adjust after evaluation

app = Flask(__name__)
init_db()

# Load model & labels if exists
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        label_ids = pickle.load(f)  # folder_name -> id
    id_to_label = {v:k for k,v in label_ids.items()}
    print("[INFO] Model and labels loaded.")
else:
    recognizer = None
    label_ids = {}
    id_to_label = {}
    print("[WARN] Model not found. Train model before using /scan.")

# Endpoint: health
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message":"Facial recognition backend running."})

# Endpoint: register metadata for student folder (does not capture images)
@app.route("/students/register", methods=["POST"])
def api_register_student():
    """
    JSON: { "folder_name": "ENG2023001_samuel", "matric_no": "ENG2023001", "name": "Samuel" }
    You must have already created folder under dataset/ with images (via capture_faces.py).
    """
    data = request.get_json()
    if not data or "folder_name" not in data:
        return jsonify({"error":"folder_name required"}), 400
    folder = data["folder_name"]
    matric = data.get("matric_no")
    name = data.get("name")
    folder_path = os.path.join("database", folder)
    print(folder_path)
    if not os.path.isdir(folder_path):
        return jsonify({"error":"dataset folder not found. Capture images first."}), 400

    ok = add_student(folder, matric, name)
    if not ok:
        return jsonify({"warning":"Student already registered in DB."}), 200
    return jsonify({"success":"Student added. Run train_model.py to (re)train model so API can recognize them."})

# Endpoint: scan image (multipart form-data: image file) -> returns first recognized student
@app.route("/scan", methods=["POST"])
def api_scan():
    # check model
    global recognizer, id_to_label
    if recognizer is None:
        return jsonify({"error":"Model not trained. Run train_model.py first."}), 500
    print(request)
    if 'image' not in request.files:
        return jsonify({"error":"No image file"}), 400

    file = request.files['image']
    in_memory = file.read()
    npimg = np.frombuffer(in_memory, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error":"Invalid image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = extract_faces_from_gray(gray)

    # if multiple faces, process each
    for roi in faces:
        try:
            label_pred, conf = recognizer.predict(roi)
        except Exception as e:
            return jsonify({"error":"Prediction failed", "details": str(e)}), 500

        # interpret confidence: LBPH gives lower==better. We invert to percent-ish for response.
        if conf <= CONFIDENCE_THRESHOLD:
            folder_name = id_to_label.get(label_pred)
            if folder_name:
                # Log attendance
                success, info = log_attendance_by_folder(folder_name, course=request.form.get("course"))
                return jsonify({
                    "recognized": True,
                    "folder_name": folder_name,
                    "student_db": get_student_by_folder(folder_name),
                    "confidence": float(conf),
                    "logged": success,
                    "info": info
                })

    return jsonify({"recognized": False, "message":"No known face recognized."})

# Endpoint: get attendance report
@app.route("/reports", methods=["GET"])
def reports():
    course = request.args.get("course")
    date = request.args.get("date")
    rows = get_attendance_report(course=course, date=date)
    # return as list of dicts
    res = []
    for r in rows:
        res.append({
            "attendance_id": r[0],
            "folder_name": r[1],
            "matric_no": r[2],
            "name": r[3],
            "course": r[4],
            "date": r[5],
            "time_in": r[6]
        })
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)
