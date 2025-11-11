from flask import Flask, render_template, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from PIL import Image
import sqlite3
from datetime import datetime
import logging
from model import get_emotion_detector, predict_emotion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "emotion_detection_secret_key_2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Helper functions ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_database_stats():
    try:
        conn = sqlite3.connect("emotion_predictions.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute(
            "SELECT predicted_emotion, COUNT(*) FROM predictions GROUP BY predicted_emotion ORDER BY COUNT(*) DESC"
        )
        emotion_counts = dict(cursor.fetchall())

        cursor.execute(
            "SELECT timestamp, predicted_emotion, confidence, source FROM predictions ORDER BY timestamp DESC LIMIT 5"
        )
        recent_predictions = cursor.fetchall()

        conn.close()

        return {
            "total_predictions": total_predictions,
            "emotion_counts": emotion_counts,
            "recent_predictions": recent_predictions,
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {"total_predictions": 0, "emotion_counts": {}, "recent_predictions": []}

# ---------- Routes ----------
@app.route("/")
def index():
    stats = get_database_stats()
    return render_template("index.html", stats=stats, title="Emotion Detection Web App")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": "File type not allowed. Please upload an image file."}
            ), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)

            try:
                img = Image.open(temp_file.name)
                img.verify()
                img = Image.open(temp_file.name)
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except Exception as img_error:
                os.unlink(temp_file.name)
                return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400

            logger.info(f"Making prediction for uploaded image: {file.filen
