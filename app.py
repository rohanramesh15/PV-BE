import os
from dotenv import load_dotenv

# Load environment variables FIRST before any other imports that use them
load_dotenv()

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from time import time_ns
import logging

# Import after load_dotenv so replicate sees the env vars
from image_compare.clip_only_comparison import CLIPComparator

from PIL import Image
import io
logger = logging.getLogger()

# Configure CORS origins from environment variable
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3001,http://localhost:3002')
allowed_origins = [origin.strip() for origin in cors_origins.split(',')]

logger.info(f"CORS enabled for origins: {allowed_origins}")

app = Flask(__name__)
CORS(app, origins=allowed_origins, supports_credentials=True)
  # Enable CORS for specified origins only

# In-memory storage for team scores
scores = {
    "team1": 0,
    "team2": 0
}

comparator = CLIPComparator("./image_compare/images/Riley1.jpg", "./image_compare/images/Rohan1.jpg")

runtimes = []


@app.route('/api/scores', methods=['GET'])
def get_scores():
    """Get current scores for both teams"""
    print("successfully sent the score")
    return jsonify(scores)



# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        start_time = time_ns()
        print("Started!")
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Secure the filename
        filename = secure_filename(file.filename)

        # Read image data
        image_data = file.read()

        # Optional: Process image with PIL
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        result = comparator.compare(image_a_stream=image)
        team = 'team2'
        if result['best_match'] == 1:
            team = 'team1'
        scores[team] += 1

        end_time = time_ns()
        print("Ended! Took " + str(end_time-start_time) + " nanoseconds")

        runtimes.append(end_time-start_time)
        if len(runtimes)%20 == 0:
            print("Average Runtime: " + str(sum(runtimes) / len(runtimes) ))

        # Get additional form data if any
        description = request.form.get('description', '')

        # Return success response
        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename,
            "team": team,
            "newScore": scores[team],
            "allScores": scores
        }), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500



@app.route('/api/scores/increment', methods=['POST'])
def increment_score():
    """Increment score for a specific team"""
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

@app.route('/api/scores/reset', methods=['POST'])
def reset_scores():
    """Reset all scores to zero"""

    scores["team1"] = 0
    scores["team2"] = 0
    return jsonify({
        "success": True,
        "scores": scores
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
