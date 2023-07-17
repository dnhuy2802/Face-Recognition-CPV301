from app.logic.deeplearning.vgg_model import VGGModel, FaceRecognitionVGGModel
from app.logic.data_split import DataSplit

data_split = DataSplit()
data_split.data_split()

# vgg_model = VGGModel()
# vgg_model.model_train()

# vgg_face_model = FaceRecognitionVGGModel()
# vgg_face_model.face_predict()
