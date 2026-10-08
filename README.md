# AI Attendance System

A Python and Flask based attendance system that uses face recognition to identify registered students from a classroom photo and generate an Excel attendance report.

This project is a practical AI automation experiment for reducing manual attendance work in classrooms.

## Features

- Student registration with multiple face images
- Face recognition using InsightFace
- Teacher upload page for classroom photos
- Automatic attendance generation
- Excel report export
- Unknown face handling
- Flask web interface for student and teacher workflows

## Tech Stack

- Python
- Flask
- OpenCV
- InsightFace
- NumPy
- Pandas
- OpenPyXL

## Project Structure

```text
AI-Attendance-System/
├── scripts/
│   ├── app.py
│   ├── build_embeddings_insightface.py
│   ├── face_recognition_insightface.py
│   ├── capture_faces.py
│   └── templates/
├── attendance.csv
├── requirements.txt
└── README.md
```

## How It Works

1. Students register by uploading clear face photos.
2. The system creates face embeddings for registered students.
3. A teacher uploads a classroom photo.
4. The AI model detects and matches faces.
5. The system creates an Excel attendance file with present and unknown students.

## Setup

Clone the repository:

```bash
git clone https://github.com/harpreet-dev12/AI-Attendance-System.git
cd AI-Attendance-System
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python scripts/app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Basic Workflow

1. Open the student page and register a student with at least 12 clear face photos.
2. Build the face embeddings:

```bash
python scripts/build_embeddings_insightface.py
```

3. Open the teacher page and upload a classroom photo.
4. Download the generated Excel attendance report.

## Notes

- Use clear, front-facing student photos for better recognition accuracy.
- Generated files such as uploaded photos, embeddings, and Excel reports should not be committed to GitHub.
- Large local tools such as `ngrok.exe` should be downloaded separately instead of stored in the repository.

## Future Improvements

- Add a cleaner dashboard UI
- Add student database support
- Improve duplicate attendance handling
- Add authentication for teacher access
- Add a setup guide with screenshots

## Author

Built by [Harpreet](https://github.com/harpreet-dev12) as an AI automation project.
