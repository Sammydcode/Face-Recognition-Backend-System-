# Face-Recognition-Backend-System-

A lightweight **Flask + OpenCV** backend for automating **attendance tracking** using facial recognition.  
This system detects faces, recognizes enrolled students, and logs attendance into a local database.

---

## Features
- Register students (image folders or via API)
- Train a face recognition model (LBPH with OpenCV)
- Real-time or batch attendance logging
- Generate daily/weekly/monthly attendance reports
- Modular utilities for face detection and database operations
- SQLite by default (can be upgraded to PostgreSQL/MySQL)
- RESTful API endpoints for easy frontend or mobile integration

---

## Tech Stack
- **Language:** Python 3.9+  
- **Framework:** Flask  
- **Computer Vision:** OpenCV  
- **Database:** SQLite  
- **Serialization:** Pickle   

---

## Project Structure
```bash
facial-attendance-backend/
├── app.py                 # Main Flask API
├── database.py            # DB initialization & operations
├── utils.py               # Face detection & preprocessing
├── train.py               # Model training script
├── models/
│   ├── face_recognizer.yml
│   └── labels.pkl
├── students/
│   ├── student_001/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
├── requirements.txt
└── README.md
