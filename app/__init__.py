from flask import Blueprint
from .routes import register_faces_bp, train_bp, predict_bp


### Create Blueprint ###
apis_bp = Blueprint('apis', __name__, url_prefix='/apis')


### Import Routes ###
apis_bp.register_blueprint(register_faces_bp)
apis_bp.register_blueprint(train_bp)
apis_bp.register_blueprint(predict_bp)
