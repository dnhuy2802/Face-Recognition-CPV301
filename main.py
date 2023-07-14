# Flask Root File

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from app import apis_bp

### Initialize Flask App ###
app = Flask(__name__)
io = SocketIO(app, cors_allowed_origins="*")


### Cross-origin config ###
CORS(app, resources={r"/*": {"origins": "*"}})


### Frontend Routes ###
@app.route("/")
def main():
    return 'Back-end is running!'


### Import Blueprints ###
app.register_blueprint(apis_bp)


### SocketIO ###
if __name__ == "__main__":
    io.run(app, debug=True)
