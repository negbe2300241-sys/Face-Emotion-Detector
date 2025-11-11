## 😃 Emotion Detection Web App
## 🧠 Overview

Emotion Detection Web App is a Flask-based machine learning application that identifies human emotions from uploaded images.
It combines computer vision, deep learning, and an intuitive web interface to provide real-time emotion analysis and visualization of prediction confidence levels.

## 🚀 Features

- 🖼️ Upload images via drag-and-drop or file picker

- ⚡ Real-time emotion prediction with confidence scores

- 📊 Dashboard displaying:

    - Total number of predictions made

    - Top detected emotions

    - Recent prediction history

- 🌐 JSON API endpoint (/predict) for integration with other applications

- 💻 Responsive, minimal frontend design

## 🧩 Tech Stack
|Category |	Technologies Used|
|-----------|-------------------|
|Backend |	Python, Flask|
|Frontend |	HTML, CSS, JavaScript (Vanilla)|
|Machine Learning |	TensorFlow / PyTorch|
|Database |	SQLite|

## 🗂️ Project Structure
emotion_detection_web_app/\
│
├── app.py                  # Main Flask application\
├── static/\
│   ├── css/                # Custom CSS styles\
│   └── js/                 # Frontend logic\
├── templates/\
│   ├── index.html          # Main user interface\
│   └── layout.html         # Base template\
├── model_cache/            # Model weights and cache\
├── uploads/                # Uploaded images (excluded from git)\
├── logs/                   # Logs and analytics\
├── requirements.txt\
├── .gitignore\
└── README.md

## ⚙️ Installation & Setup
1️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
```

Activate it:
```bash
macOS/Linux:

source venv/bin/activate
```

```bash
Windows:

venv\Scripts\activate
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the Application
```bash
flask run
```

Open your browser and visit:
👉 http://127.0.0.1:5000/

## 🔍 API Endpoint

POST /predict

Parameter	Type	Description
file	image	Image file to analyze

Example Response:

{
  "predicted_emotion": "happy",
  "confidence": 0.91
}

## 📊 Dashboard Insights

The analytics section displays:

Total number of predictions

Top 3 most frequently detected emotions

Recent activity (last 3 predictions)

All data is stored locally using SQLite.

## 🔒 Security Notes

❌ Avoid committing large model files (*.h5, *.pt, *.bin, etc.) to public repositories

🚫 Do not commit uploaded files or logs (may contain personal data)

🧩 Check for API keys or sensitive data before pushing to GitHub

📦 Use Git LFS for tracking large model files if needed

## 🌱 Possible Improvements

🎛️ Add multiple model options or emotion intensity scales

📸 Integrate webcam support for live emotion detection

📈 Visualize confidence levels using charts (e.g., Chart.js)

🐳 Containerize with Docker for smoother deployment

🌍 Add multilingual emotion labels

## 🧾 License

This project is open-source and provided for educational and demonstration purposes.
You are free to modify and extend it for your own learning or research.

## 🙌 Credits

Developed by Naomi Chiamaka Egbe (23CG034058)
A demonstration of deep learning–powered emotion recognition inspired by open-source CNN emotion models.