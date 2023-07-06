from flask import Blueprint, request, jsonify


### Create Blueprint ###
predict_bp = Blueprint('predict', __name__, url_prefix='/predict')


### Routes ###
@predict_bp.route('/', methods=['GET'])
def predict():
    return "Predict Image"
