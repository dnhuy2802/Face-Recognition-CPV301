# Flask Root File

from flask import Flask
from flask_cors import CORS
from app import apis_bp

### Initialize Flask App ###
app = Flask(__name__)

### Cross-origin config ###
CORS(app, resources={r"/*": {"origins": "*"}})


### Frontend Routes ###
@app.route("/")
def main():
    return 'Back-end is running!'


### Import Blueprints ###
app.register_blueprint(apis_bp)
