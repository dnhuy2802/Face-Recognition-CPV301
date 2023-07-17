# Flask Root File
from app import create_app, io
from app.app import apis_bp
from app.logic.model_table import ModelTable
from app.logic.face_detectors.hog_face import HogFaceDetector

### Initialize Flask App ###
flask_app = create_app()


### App Config ###
# Load model table
model_table = ModelTable()
model_table.load_table()
flask_app.config['MODEL_TABLE'] = model_table
# Load model
flask_app.config['MODEL'] = None
# Load detector
flask_app.config['DETECTOR'] = HogFaceDetector()


### Frontend Routes ###
@flask_app.route("/")
def main():
    return 'Back-end is running!'


### Register Blueprints ###
flask_app.register_blueprint(apis_bp)


### SocketIO ###
if __name__ == "__main__":
    io.run(flask_app, debug=True)
