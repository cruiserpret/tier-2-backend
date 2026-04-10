from flask import Flask
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from backend.api.routes import api

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Allow Hamza's frontend on port 3000 to talk to us
CORS(app, origins=["http://localhost:3000"])

# Register all routes
app.register_blueprint(api)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "Assembly backend is live"}, 200

if __name__ == "__main__":
    print("Starting Assembly backend...")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=config.DEBUG)