import os
import cv2
import pickle
from scripts.face_recognition_dl import detect_faces, get_face_embedding, EMBEDDINGS_PATH

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "student_uploads")

embeddings_db = {}

print("[INFO] Building embeddings from dataset...")

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings_db[person_name] = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        faces = detect_faces(image)
        if len(faces) == 0:
            continue

        # take biggest face
        face = faces[0]
        emb = get_face_embedding(face)
        embeddings_db[person_name].append(emb)

    print(f"[OK] {person_name}: {len(embeddings_db[person_name])} embeddings")

os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(embeddings_db, f)

print(f"[SUCCESS] embeddings.pickle saved at:\n{EMBEDDINGS_PATH}")
