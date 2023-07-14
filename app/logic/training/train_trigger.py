import enum


# Enum for Training Type
class TrainingType(enum.Enum):
    # Machine Learning
    EIGENFACE = 'Eigenface'
    # Deep Learning
    VGG16 = 'VGG16'


class Training:
    def __init__(self, type: TrainingType):
        self.type = type
