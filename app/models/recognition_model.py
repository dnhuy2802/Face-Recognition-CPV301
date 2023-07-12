import enum


# Enum for Training Type
class TrainingType(enum.Enum):
    # Machine Learning
    EIGENFACE = 'Eigenface'
    # Deep Learning
    VGG16 = 'VGG16'


class RecognitionModel:
    def __init__(self, training_type: TrainingType, registered_faces: list[str], model_path: str):
        self.training_type = training_type
        self.registered_faces = registered_faces
        self.model_path = model_path

    def sign_model(self, model_path: str):
        self.model_path = model_path

    @staticmethod
    def from_json(json):
        return RecognitionModel.create(
            json['type'],
            json['faces'],
            None
        )

    @staticmethod
    def create(training_type: TrainingType, registered_faces: list[str], model_path: str):
        return RecognitionModel(
            training_type=training_type,
            registered_faces=registered_faces,
            model_path=model_path
        )
