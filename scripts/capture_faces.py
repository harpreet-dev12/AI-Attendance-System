import cv2
import os
import time

DATASET_PATH = "dataset"

# Ask for student name
name = input("Enter student name: ").strip()

if name == "":
    print("❌ Name cannot be empty!")
    exit()

person_path = os.path.join(DATASET_PATH, name)

if not os.path.exists(person_path):
    os.makedirs(person_path)
    print(f"[INFO] Created folder: {person_path}")
else:
    print(f"[INFO] Using existing folder: {person_path}")

# Open phone camera (Iriun usually = 1)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Could not open phone camera at index 1. Trying index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Could not open any camera.")
    exit()

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Warm-up camera
print("[INFO] Warming up camera...")
for i in range(20):
    cap.read()

count = 0
MAX_IMAGES = 20

print(f"[INFO] Starting face capture for: {name}")
print("[INFO] Look at camera. Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1
        img_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(img_path, face)

        print(f"[CAPTURED] {img_path}")

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Captured {count}/{MAX_IMAGES}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        # Small delay so images are different
        time.sleep(0.3)

    cv2.imshow("Auto Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[SUCCESS] Captured {count} face images for {name}")
print("[NEXT] Now run: python scripts/train_lbph.py")
