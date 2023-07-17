from flask import current_app
from ...models.recognition_model import RecognitionModel, ModelType
from ..eigenfaces import PCASVM
from ..model_table import ModelTable


class TrainingTrigger:
    def __init__(self, model: RecognitionModel):
        self.model = model

    def training(self):
        if self.model.type == ModelType.PCASVM.value:
            model = PCASVM(self.model.registered_faces, k=25)
        if self.model.type == ModelType.VGG19.value:
            ...
        if self.model.type == ModelType.RESNET50.value:
            ...
        if self.model.type == ModelType.CONVNEXTBASE.value:
            ...
        # Train model
        acc = model.fit()
        # Save model
        model.save(self.model.path)
        # Add model to table
        table: ModelTable = current_app.config['MODEL_TABLE']
        table.add_model(self.model.name, self.model.type, self.model.path, acc)
        # Return accuracy
        return acc
