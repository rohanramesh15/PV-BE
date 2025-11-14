# Backend (Flask)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

Server will run on http://localhost:5000

## API Endpoints

### GET /api/scores
Get current scores for both teams
- Response: `{"team1": 0, "team2": 0}`

### POST /api/scores/increment
Increment score for a specific team
- Request body: `{"team": "team1"}` or `{"team": "team2"}`
- Response: `{"success": true, "team": "team1", "newScore": 1, "allScores": {"team1": 1, "team2": 0}}`

### POST /api/scores/reset
Reset all scores to zero
- Response: `{"success": true, "scores": {"team1": 0, "team2": 0}}`
