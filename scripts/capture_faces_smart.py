import cv2
import os
import numpy as np
import time

person_name = input("Enter person name: ").strip()
dataset_dir = os.path.join("dataset", person_name)
os.makedirs(dataset_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0
MAX_IMAGES = 25
MIN_FACE_SIZE = 120
MIN_BRIGHTNESS = 70
MAX_BLUR = 100.0  # lower = more strict

print("[INFO] Smart capture started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera read failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))

        # Brightness check
        brightness = np.mean(face_resized)
        if brightness < MIN_BRIGHTNESS:
            cv2.putText(frame, "Too Dark", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue

        # Blur check
        blur = cv2.Laplacian(face_resized, cv2.CV_64F).var()
        if blur < MAX_BLUR:
            cv2.putText(frame, "Blurry", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue

        # Draw green box for GOOD face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Save every 0.6 seconds
        if time.time() % 0.6 < 0.02:
            img_path = os.path.join(dataset_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_resized)
            print(f"[SAVED] {img_path}")
            count += 1
            time.sleep(0.6)

        if count >= MAX_IMAGES:
            break

    cv2.putText(frame, f"Saved: {count}/{MAX_IMAGES}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Smart Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print("[SUCCESS] Smart face capture complete.")
