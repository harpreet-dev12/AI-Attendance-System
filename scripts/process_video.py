import cv2
import datetime

def process_video(video_path):
    print("[INFO] Processing video:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return

    # For now, just simulate attendance
    while True:
        ret, frame = cap.read()
        if not ret:
            break

    cap.release()

    # Dummy attendance entry (for testing)
    now = datetime.datetime.now().strftime("%H:%M:%S")
    with open("attendance.csv", "a") as f:
        f.write(f"Test_User,{now}\n")

    print("[SUCCESS] Video processed, attendance marked.")
