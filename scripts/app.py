import os
import cv2
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

# IMPORT DL FUNCTIONS
from scripts.face_recognition_insightface import recognize_faces

app = Flask(__name__)

# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_UPLOAD_DIR = os.path.join(BASE_DIR, "..", "student_uploads")
TEACHER_UPLOAD_DIR = os.path.join(BASE_DIR, "..", "teacher_uploads")

os.makedirs(STUDENT_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEACHER_UPLOAD_DIR, exist_ok=True)

print("[INFO] Starting AI Attendance System (Deep Learning Mode)")

# ===============================
# HOME
# ===============================
@app.route("/")
def home():
    return render_template("home.html")

# ===============================
# STUDENT PAGE
# ===============================
@app.route("/student", methods=["GET"])
def student_page():
    return render_template("student.html")

# ===============================
# STUDENT REGISTER 
# ===============================
@app.route("/register", methods=["POST"])
def register():
    try:
        # 🔥 Get separate fields
        student_name = request.form.get("student_name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        files = request.files.getlist("images")

        # ✅ Validation
        if not student_name or not roll_no:
            return render_template("student.html", message="❌ Name and Roll No are required.")

        if not files or len(files) < 12:
            return render_template("student.html", message="❌ Please upload at least 12 photos.")

        # 🔥 Unique person ID (used everywhere)
        person_id = f"{roll_no}_{student_name}".replace(" ", "_")

        # 🔥 Create student folder
        student_dir = os.path.join(STUDENT_UPLOAD_DIR, secure_filename(person_id))
        os.makedirs(student_dir, exist_ok=True)

        saved = 0
        for i, file in enumerate(files):
            if file.filename == "":
                continue

            filename = secure_filename(f"{person_id}_{i}.jpg")
            save_path = os.path.join(student_dir, filename)
            file.save(save_path)
            saved += 1

        if saved < 12:
            return render_template(
                "student.html",
                message="❌ Not enough valid images saved. Use clear face photos."
            )

        print(f"[INFO] Registered student: {person_id} | Photos saved: {saved}")

        return render_template(
            "student.html",
            message="✅ Registration completed! 12 photos saved. Now run: python scripts/build_embeddings_insightface.py"
        )

    except Exception as e:
        print("[ERROR] Registration failed:", e)
        return render_template("student.html", message=f"❌ Registration Error: {e}")

# ===============================
# TEACHER PAGE
# ===============================
@app.route("/teacher", methods=["GET"])
def teacher_page():
    return render_template("teacher.html")

# ===============================
# TEACHER ATTENDANCE
# ===============================
@app.route("/upload_attendance", methods=["POST"])
def upload_attendance():
    try:
        file = request.files.get("photo")

        if not file or file.filename == "":
            return render_template("teacher.html", message="❌ Please upload or capture a class photo.")

        # ✅ ALWAYS SAVE FILE FIRST
        filename = secure_filename(file.filename)
        save_path = os.path.join(TEACHER_UPLOAD_DIR, filename)
        file.save(save_path)

        print("[INFO] Saved class photo at:", save_path)

        # ✅ PASS FILE PATH TO AI
        present_students, excel_path = recognize_faces(save_path)

        if not present_students:
            return render_template(
                "result.html",
                message="⚠️ No registered students detected. All faces are UNKNOWN.",
                present_students=[],
                excel_path=excel_path
            )

        return render_template(
            "result.html",
            message="✅ Attendance Completed Successfully!",
            present_students=present_students,
            excel_path=excel_path
        )

    except Exception as e:
        print("[ERROR] Attendance failed:", e)
        return render_template("teacher.html", message=f"❌ Attendance Error: {e}")

# ===============================
# DOWNLOAD EXCEL
# ===============================
@app.route("/download")
def download_excel():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
