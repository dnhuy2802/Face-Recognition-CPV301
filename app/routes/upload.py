from flask import Blueprint, request, jsonify


### Create Blueprint ###
upload_bp = Blueprint('upload', __name__, url_prefix='/upload')


### Routes ###
@upload_bp.route('/', methods=['GET'])
def upload():
    return "Upload Image"
