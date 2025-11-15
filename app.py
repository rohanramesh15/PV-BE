from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename


from image_compare.clip_only_comparison import compare_with_clip

import os

from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for team scores
scores = {
    "team1": 0,
    "team2": 0
}

@app.route('/api/scores', methods=['GET'])
def get_scores():
    """Get current scores for both teams"""
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


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
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
        result = compare_with_clip(image, "Riley1.jpg", "Rohan1.jpg")
        if result == "Riley1.jpg":
            scores['team1'] += 1
        else:
            scores['team2'] += 1
        
        # Save the file
        
        
        # Get additional form data if any
        description = request.form.get('description', '')
        
        # Return success response
        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename,
        }), 200
        
    except Exception as e:
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
    app.run(debug=True, port=5000)
