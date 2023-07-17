import os
import enum
import pandas as pd
from ..logic.eigenfaces import PCASVM
from ..utils.utils import generate_uuid
from ..utils.constants import MODEL_DIR, ML_MODEL_EXT


class TrainingType(enum.Enum):
    # Machine Learning
    ML = 'ml'
    # Deep Learning
    DL = 'dl'


class ModelType(enum.Enum):
    # Machine Learning
    PCASVM = 'pcasvm'
    # Deep Learning
    VGG19 = 'vgg19'
    RESNET50 = 'resnet50'
    CONVNEXTBASE = 'convnextbase'


class MLOptions:
    def __init__(self, train: int, test: int):
        self.train = train / 100
        self.test = test / 100

    @staticmethod
    def from_json(json):
        return MLOptions(
            train=json['mlTrain'],
            test=json['mlTest'],
        )


class DLOptions:
    def __init__(self, train: int, valid: int, test: int, fine_tune: bool, batch_size: int):
        self.train = train / 100
        self.valid = valid / 100
        self.test = test / 100
        self.fine_tune = fine_tune
        self.batch_size = batch_size

    @staticmethod
    def from_json(json):
        return DLOptions(
            train=json['dlTrain'],
            valid=json['dlValid'],
            test=json['dlTest'],
            fine_tune=json['dlFineTune'],
            batch_size=json['dlBatchSize'],
        )


class RecognitionModel:
    def __init__(self, type: str, registered_faces: list[str], options: MLOptions | DLOptions, ext: str = ML_MODEL_EXT):
        self.type = type
        self.registered_faces = registered_faces
        self.options = options
        self.name = self.get_name()
        self.path = self.get_path()
        self.ext = ext

    def get_name(self):
        name = self.type + '_' + generate_uuid()
        return name

    def get_path(self):
        path = os.path.join(MODEL_DIR, self.name + self.ext)
        return path

    @staticmethod
    def from_json(json):
        if json['type'] == TrainingType.ML.value:
            json_type = json['options']['mlAlgorithm']
            options = MLOptions.from_json(json['options'])
        if json['type'] == TrainingType.DL.value:
            json_type = json['options']['dlNetwork']
            options = DLOptions.from_json(json['options'])
        return RecognitionModel(
            json_type,
            json['faces'],
            options,
        )


class RecognizedModel:
    def __init__(self, name: str, type: TrainingType, path: str, accuracy: float):
        self.name = name
        self.type = type
        self.path = path
        self.accuracy = accuracy

    def get_model_from_name(self):
        if self.type == ModelType.PCASVM.value:
            return PCASVM()
        if self.type == ModelType.VGG19.value:
            return None
        if self.type == ModelType.RESNET50.value:
            return None
        if self.type == ModelType.CONVNEXTBASE.value:
            return None
        return None

    @staticmethod
    def from_df(series: pd.DataFrame):
        return RecognizedModel(
            series['name'][0],
            series['type'][0],
            series['path'][0],
            series['accuracy'][0],
        )
