from flask import Blueprint, request, jsonify
from ..models import ResponseObject


### Create Blueprint ###
train_bp = Blueprint('train', __name__, url_prefix='/train')


### Routes ###
@train_bp.route('/start', methods=['POST'])
def train():
    print(request.get_json())
    return ResponseObject.success('Training started')
