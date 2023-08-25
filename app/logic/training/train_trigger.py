from flask import current_app
from ...models.recognition_model import RecognitionModel, ModelType
from ..eigenfaces import PCASVM
from ..deeplearning import DeepLearningModelWrapper
from ..model_table import ModelTable


class TrainingTrigger:
    def __init__(self, model: RecognitionModel):
        self.model = model

    def training(self):
        # Machine learning
        if self.model.type == ModelType.PCASVM.value:
            model = PCASVM(self.model.registered_faces,
                           split_ratio=self.model.options.test)
        # Deep learning
        else:
            model = DeepLearningModelWrapper(
                self.model.registered_faces,
                self.model.path,
                type=self.model.type,
                train_percent=self.model.options.train,
                valid_percent=self.model.options.valid,
                test_percent=self.model.options.test,
                batch_size=self.model.options.batch_size,
                fine_tune=self.model.options.fine_tune)
        # Train model
        acc = model.fit()
        # Save model
        model.save(self.model.path)
        # Add model to table
        table: ModelTable = current_app.config['MODEL_TABLE']
        table.add_model(self.model.name, self.model.type, self.model.path, acc)
        # Return accuracy
        return acc
