from flask import Flask
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from backend.api.routes import api
from backend.dtc_v3.api import api_v3
from backend.dtc_v2.routes import api_v2

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

CORS(app, origins="*", allow_headers=["Content-Type", "ngrok-skip-browser-warning"], methods=["GET", "POST", "OPTIONS"])

app.register_blueprint(api)
app.register_blueprint(api_v2)
app.register_blueprint(api_v3)

@app.route("/", methods=["GET"])
def index():
    return {
        "name": "Assembly API",
        "status": "live",
        "version": "1.0.0",
        "endpoints": [
            "POST /api/simulation/start",
            "GET /api/simulation/<id>/debate",
            "GET /api/report/<id>",
            "GET /api/sentiment/history/<id>",
            "POST /api/inject",
            "POST /api/branch",
            "GET /api/agent/<id>/memory"
        ]
    }, 200

@app.route("/health", methods=["GET"])
def health():
    return {"status": "Assembly backend is live"}, 200

if __name__ == "__main__":
    print("Starting Assembly backend...")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=config.DEBUG)