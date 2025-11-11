# Emotion Detection Web App

## Overview
**Emotion Detection Web App** is a Flask-based machine learning web application that detects human emotions from uploaded images. It provides a user-friendly interface for image uploads, emotion analysis, and visualization of prediction results with confidence levels.

This project demonstrates how computer vision and deep learning can be integrated into an accessible web interface.

---

## 🚀 Features
- Upload an image via drag-and-drop or file picker.  
- Real-time emotion prediction with confidence score.  
- Displays prediction statistics:
  - Total predictions made  
  - Top detected emotions  
  - Recent activity history  
- Simple and responsive frontend.  
- JSON API endpoint (`/predict`) for integration with other apps.

---

## 🧩 Tech Stack
**Backend:** Python, Flask  
**Frontend:** HTML, CSS, JavaScript (Vanilla)  
**Machine Learning:** TensorFlow / PyTorch (for emotion model)  
**Database:** SQLite (for logging predictions)

---

## 🗂️ Project Structure
emotion_detection_web_app/
│
├── app.py # Main Flask application
├── static/
│ ├── css/ # Custom CSS styles
│ └── js/ # Frontend logic
├── templates/
│ ├── index.html # Main interface
│ └── layout.html # Base template
├── model_cache/ # Model weights and cache
├── uploads/ # User uploads (excluded from git)
├── logs/ # Logs and analytics
├── requirements.txt
├── .gitignore
└── README.md

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Flask App
bash
Copy code
flask run
Then open your browser at http://127.0.0.1:5000/.

🔍 API Endpoint
POST /predict
Parameter	Type	Description
file	image	Input image to analyze

Example Response:

json
Copy code
{
  "predicted_emotion": "happy",
  "confidence": 0.91
}
📊 Statistics
The dashboard provides:

Total Predictions

Top 3 Emotions Detected

Recent Activity (last 3 predictions)

Data is stored locally in a SQLite database.

🔒 Security Notes
Avoid committing model weights (*.bin, *.pt, *.h5, etc.) to public repositories.

Don’t commit uploaded files or logs — they may contain personal data.

Check .env or config files for API keys before pushing to GitHub.

Use Git LFS if you need to track large model files.

🧠 Possible Improvements
Add model selection or emotion intensity scale.

Integrate webcam capture for live emotion detection.

Visualize emotion confidence with charts.

Containerize with Docker for smoother deployment.

Add multilingual emotion labels.

🧾 License
This project is provided for educational and demonstration purposes.
You are free to modify and extend it for your own research or learning.

🙌 Credits
Developed as a demonstration of computer vision and emotion recognition using deep learning.
Inspired by open-source CNN-based emotion detection models.
Developer is NAOMI CHIAMAKA EGBE 23CG034058