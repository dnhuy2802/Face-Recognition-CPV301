from flask import Blueprint, request, jsonify


### Create Blueprint ###
train_bp = Blueprint('train', __name__, url_prefix='/train')


### Routes ###
@train_bp.route('/train', methods=['POST'])
def train():
    return "Train Image"
