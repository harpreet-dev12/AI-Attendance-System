import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# ============================
# PATHS (FIXED)
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_UPLOAD_DIR = os.path.join(BASE_DIR, "student_uploads")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "embeddings_insightface.pickle")

print("[DEBUG] BASE_DIR =", BASE_DIR)
print("[DEBUG] STUDENT_UPLOAD_DIR =", STUDENT_UPLOAD_DIR)

# ============================
# INIT INSIGHTFACE
# ============================
print("[INFO] Initializing InsightFace...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ============================
# BUILD EMBEDDINGS
# ============================
embeddings_db = {}

if not os.path.exists(STUDENT_UPLOAD_DIR):
    print("[ERROR] student_uploads folder not found!")
    exit(1)

student_folders = os.listdir(STUDENT_UPLOAD_DIR)
print("[DEBUG] Found student folders:", student_folders)

for student_name in student_folders:
    student_path = os.path.join(STUDENT_UPLOAD_DIR, student_name)

    if not os.path.isdir(student_path):
        continue

    print(f"[INFO] Processing student: {student_name}")
    embeddings_db[student_name] = []

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] Could not read image:", img_path)
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print("[WARN] No face detected in:", img_name)
            continue

        # Take largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding

        embeddings_db[student_name].append(emb)

    print(f"[OK] {student_name}: {len(embeddings_db[student_name])} embeddings")

# ============================
# SAVE EMBEDDINGS
# ============================
if len(embeddings_db) == 0:
    print("[ERROR] No embeddings created! Check student_uploads.")
    exit(1)

os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(embeddings_db, f)

print("\n[SUCCESS] InsightFace embeddings saved to:")
print(EMBEDDINGS_PATH)
print("[INFO] Total registered students:", len(embeddings_db))
