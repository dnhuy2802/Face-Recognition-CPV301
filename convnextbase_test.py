from app.logic.deeplearning.convnextbase_model import ConvNextBaseModel, FaceRecognitionConvNextBaseModel
from app.logic.data_split import DataSplit

# data_split = DataSplit()
# data_split.data_split()

# convnextbase_model = ConvNextBaseModel()
# convnextbase_model.model_train()

convnextbase_face_model = FaceRecognitionConvNextBaseModel()
convnextbase_face_model.face_predict()
