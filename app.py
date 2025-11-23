from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from time import time_ns

from image_compare.clip_only_comparison import CLIPComparator

import os
from PIL import Image
import io

# ------------------------------------
# App Setup
# ------------------------------------
app = Flask(__name__)

# Enable base CORS (needed for simple GETs)
CORS(app)

# Force CORS on ALL responses (Cloud Run requires this)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3001"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

# Handle OPTIONS requests (important for uploads)
@app.route('/api/upload', methods=['OPTIONS'])
def upload_options():
    return '', 200

# ------------------------------------
# Score Storage
# ------------------------------------
scores = {
    "team1": 0,
    "team2": 0
}

# ------------------------------------
# CLIP Setup
# ------------------------------------
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ ERROR: CLIP not installed!")
    print("Install with:")
    print("  pip install ftfy regex")
    print("  pip install git+https://github.com/openai/CLIP.git")
    raise

comparator = CLIPComparator(
    "./image_compare/images/Riley1.jpg",
    "./image_compare/images/Rohan1.jpg"
)

runtimes = []

# ------------------------------------
# File Upload Config
# ------------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------
# Routes
# ------------------------------------

# Get scores
@app.route('/api/scores', methods=['GET'])
def get_scores():
    return jsonify(scores)


# Upload + process image
@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        start_time = time_ns()
        print("Upload started")

        # Must include "image"
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        image_bytes = file.read()

        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Run CLIP comparison
        result = comparator.compare(image_a_stream=image)

        team = 'team1' if result['best_match'] == 1 else 'team2'
        scores[team] += 1

        end_time = time_ns()
        duration = end_time - start_time
        runtimes.append(duration)

        print(f"Upload ended. Took {duration} ns")

        # Debug: average runtimes
        if len(runtimes) % 20 == 0:
            print("Average Runtime:", sum(runtimes) / len(runtimes))

        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename,
            'team': team,
            'newScore': scores[team],
            'allScores': scores
        })

    except Exception as e:
        print("Upload error:", e)
        return jsonify({'error': str(e)}), 500


# Increment score manually
@app.route('/api/scores/increment', methods=['POST'])
def increment_score():
    data = request.get_json()
    team = data.get('team')

    if team not in scores:
        return jsonify({"error": "Invalid team"}), 400

    scores[team] += 1
    return jsonify({
        "success": True,
        "team": team,
        "newScore": scores[team],
        "allScores": scores
    })


# Reset both scores
@app.route('/api/scores/reset', methods=['POST'])
def reset_scores():
    scores["team1"] = 0
    scores["team2"] = 0
    return jsonify({"success": True, "scores": scores})


# ------------------------------------
# Run App
# ------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
