from flask import Flask, jsonify, request
from flask_cors import CORS

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
