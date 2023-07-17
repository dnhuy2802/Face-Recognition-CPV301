from flask import Blueprint, current_app, request
from flask_socketio import emit
from .. import io
from ..logic.utils.split_image import split_img
from ..models.response import ResponseObject, FaceResponseObject
from ..models.recognition_model import RecognizedModel
from ..logic.model_table import ModelTable
from ..utils.utils import convert_base64_to_image, convert_to_gray


### Create Blueprint ###
predict_bp = Blueprint('predict', __name__, url_prefix='/predict')


### Routes ###
@predict_bp.route('/models', methods=['GET'])
def models():
    table: ModelTable = current_app.config['MODEL_TABLE']
    return_object = []
    for _, row in table.dataframe.iterrows():
        return_object.append({
            'name': row['name'],
            'type': row['type'],
        })
    return ResponseObject.success(return_object)


@predict_bp.route('/models/set', methods=['POST'])
def set_model():
    # Get model name
    model_name = request.get_json()['name']
    # Get model
    table: ModelTable = current_app.config['MODEL_TABLE']
    model = RecognizedModel.from_df(table.get_model(model_name))
    print(model.type)
    # Set current model
    current_model = model.get_model_from_name()
    current_app.config['MODEL'] = current_model
    current_model.load(model.path)
    # Return to client
    return ResponseObject.success('Model set')


### SocketIO ###
@io.on('recognizing')
def on_recognizing(data):
    # Initialize Model
    detector = current_app.config['DETECTOR']
    model = current_app.config['MODEL']
    # Get data
    img = data['frame']
    # Convert frame from base64 to np.ndarray
    frame = convert_base64_to_image(img)
    # If frame is not None
    if frame is not None:
        # Detect faces
        cordinates = detector.detect_faces(frame)
        # If there is at least one face
        if len(cordinates) > 0:
            # Predict faces
            recognized_faces = []
            for cordinate in cordinates:
                cord = detector.get_cordinate(cordinate)
                # Crop face
                face = split_img(frame, cord)
                face_gray = convert_to_gray(face)
                # Predict
                name = model.predict(face_gray)
                # Add to recognized_faces
                recognized_faces.append(
                    FaceResponseObject(name, 1, [*cord]).to_dict())
            # Return to client
            emit('recognized', ResponseObject.success(
                recognized_faces, json=False))
        # If there is no face
        else:
            emit('recognized', ResponseObject.error(
                'No face detected', json=False))
