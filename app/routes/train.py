from flask import Blueprint, request
from flask_socketio import emit
from .. import io
from ..models import ResponseObject
from ..logic.training.train_trigger import TrainingTrigger
from ..models.recognition_model import RecognitionModel


### Create Blueprint ###
train_bp = Blueprint('train', __name__, url_prefix='/train')


### SocketIO ###
@io.on('training')
def on_train(data):
    model = RecognitionModel.from_json(data)
    trigger = TrainingTrigger(model)
    # Start training
    accuracy = trigger.training()
    # Return to client
    emit('trained', ResponseObject.success({
        'accuracy': accuracy * 100,
    }, json=False))
