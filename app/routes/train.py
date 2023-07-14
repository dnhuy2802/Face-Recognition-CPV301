from flask import Blueprint, request, jsonify


### Create Blueprint ###
train_bp = Blueprint('train', __name__, url_prefix='/train')


### Routes ###
@train_bp.route('/', methods=['GET'])
def train():
    return "Train Image"
