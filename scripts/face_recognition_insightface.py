import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import pandas as pd

from insightface.app import FaceAnalysis

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "embeddings_insightface.pickle")

print("[INFO] Initializing InsightFace...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

print("[INFO] Loading InsightFace embeddings...")
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, "rb") as f:
        EMBEDDINGS_DB = pickle.load(f)
    print(f"[INFO] Loaded {len(EMBEDDINGS_DB)} registered students")
else:
    EMBEDDINGS_DB = {}
    print("[WARN] No InsightFace embeddings found!")


# =========================
# UTILS
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def match_face(face_emb, threshold=0.45):
    best_name = "UNKNOWN"
    best_score = -1

    for name, emb_list in EMBEDDINGS_DB.items():
        for db_emb in emb_list:
            score = cosine_similarity(face_emb, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "UNKNOWN", best_score


# =========================
# MAIN FUNCTION
# =========================
def recognize_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to load class image")

    faces = app.get(img)
    print(f"[INFO] Detected {len(faces)} faces in class photo")

    present_students = []
    unknown_count = 0

    for face in faces:
        emb = face.normed_embedding
        full_id, score = match_face(emb)

        print(f"[MATCH] {full_id} | score={score:.3f}")

        if full_id != "UNKNOWN":
            present_students.append(full_id)
        else:
            unknown_count += 1

    present_students = list(set(present_students))

    # =========================
    # DATE & TIME
    # =========================
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(BASE_DIR, f"attendance_{timestamp}.xlsx")

    rows = []

    # =========================
    # SPLIT ROLL NO & NAME
    # FORMAT EXPECTED: ROLLNO_NAME
    # =========================
    for full_id in present_students:
        if "_" in full_id:
            roll_no, name = full_id.split("_", 1)
        else:
            roll_no = ""
            name = full_id

        rows.append({
            "Roll No": roll_no,
            "Name": name,
            "Status": "Present",
            "Date": current_date,
            "Time": current_time
        })

    for _ in range(unknown_count):
        rows.append({
            "Roll No": "",
            "Name": "UNKNOWN",
            "Status": "Unknown",
            "Date": current_date,
            "Time": current_time
        })

    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)

    return present_students, excel_path
