from app.logic.deeplearning.resnet_model import ResNetModel, FaceRecognitionResNetModel
from app.logic.data_split import DataSplit

# data_split = DataSplit()
# data_split.data_split()

# resnet_model = ResNetModel()
# resnet_model.model_train()

resnet_face_model = FaceRecognitionResNetModel()
resnet_face_model.face_predict()