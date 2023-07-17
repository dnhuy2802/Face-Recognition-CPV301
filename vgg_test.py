from app.logic.deeplearning.vgg_model import VGGModel
from app.logic.deeplearning import DeepLearningModelWrapper

# data_split = DataSplit()
# data_split.data_split()

# vgg_model = VGGModel()
# vgg_model.model_train()

# vgg_face_model = FaceRecognitionVGGModel()
# vgg_face_model.face_predict()

dl = DeepLearningModelWrapper(['MinhDoan', 'CongToan', 'KienPhuc'])
dl.test()
