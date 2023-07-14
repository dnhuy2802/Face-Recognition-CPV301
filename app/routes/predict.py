from flask import Blueprint, request, jsonify, current_app


### Create Blueprint ###
predict_bp = Blueprint('predict', __name__, url_prefix='/predict')


### SocketIO ###
# io = current_app.extensions['socketio']


### Routes ###
@predict_bp.route('/', methods=['GET'])
def predict():
    # print(io)
    return "Predict Image"


### SocketIO ###
