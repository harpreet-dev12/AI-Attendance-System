import os
import cv2
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROTOTXT = os.path.join(BASE_DIR, "scripts", "deploy.prototxt")
MODEL = os.path.join(BASE_DIR, "scripts", "res10_300x300_ssd_iter_140000.caffemodel")
EMBEDDER = os.path.join(BASE_DIR, "scripts", "openface.nn4.small2.v1.t7")

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "embeddings.pickle")

# ==============================
# LOAD MODELS
# ==============================
print("[INFO] Loading face detector...")
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

print("[INFO] Loading face embedder (OpenFace)...")
embedder_net = cv2.dnn.readNetFromTorch(EMBEDDER)

print("[INFO] Loading embeddings database...")
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, "rb") as f:
        EMBEDDINGS_DB = pickle.load(f)
    print(f"[INFO] Loaded {len(EMBEDDINGS_DB)} registered students")
else:
    EMBEDDINGS_DB = {}
    print("[WARN] No embeddings found!")

# ==============================
# FACE DETECTION
# ==============================
def detect_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        face = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if face.size > 0:
            faces.append(face)

    print(f"[INFO] Detected {len(faces)} faces in class photo")
    return faces

# ==============================
# FACE EMBEDDING (REAL)
# ==============================
def get_face_embedding(face):
    face_blob = cv2.dnn.blobFromImage(
        cv2.resize(face, (96, 96)),
        1.0 / 255,
        (96, 96),
        (0, 0, 0),
        swapRB=True,
        crop=False
    )

    embedder_net.setInput(face_blob)
    vec = embedder_net.forward()
    return vec.flatten()

# ==============================
# MATCH FACE (EUCLIDEAN)
# ==============================
def match_face(face_emb, threshold=0.6):
    best_name = "UNKNOWN"
    best_dist = 999

    for name, emb_list in EMBEDDINGS_DB.items():
        for db_emb in emb_list:
            dist = np.linalg.norm(face_emb - db_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_dist < threshold:
        return best_name, best_dist
    else:
        return "UNKNOWN", best_dist

# ==============================
# MAIN ATTENDANCE
# ==============================
def recognize_faces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Failed to load image")

    faces = detect_faces(image)

    present_students = []
    unknown_count = 0

    for face in faces:
        emb = get_face_embedding(face)
        name, dist = match_face(emb)

        print(f"[MATCH] {name} | dist={dist:.3f}")

        if name != "UNKNOWN":
            present_students.append(name)
        else:
            unknown_count += 1

    present_students = list(set(present_students))

    # ==============================
    # SAVE EXCEL
    # ==============================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(BASE_DIR, f"attendance_{timestamp}.xlsx")

    rows = []
    for name in present_students:
        rows.append({"Name": name, "Status": "Present"})

    for i in range(unknown_count):
        rows.append({"Name": "UNKNOWN", "Status": "Unknown"})

    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)

    return present_students, excel_path
