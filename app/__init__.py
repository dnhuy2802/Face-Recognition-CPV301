from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

### SocketIO ###
io = SocketIO()


### Create Flask App ###
def create_app():
    # Initialize Flask App
    app = Flask(__name__)
    # Cross-origin config
    CORS(app, resources={r"/*": {"origins": "*"}})
    # Initialize SocketIO
    io.init_app(app, cors_allowed_origins="*")
    # Return Flask App
    return app
