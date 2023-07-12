### Face Model ###
# Used to store the data of the person

from unidecode import unidecode
from ..utils.utils import get_idetifier, generate_uuid


class FaceModel:
    def __init__(self, id, identifier: str, name: str, images: list[str]):
        self.id = id
        self.identifier = identifier
        self.name = name
        self.images = images

    @staticmethod
    def from_json(json):
        return FaceModel.create(
            json['name'],
            json['images']
        )

    @staticmethod
    def create(name: str, images: list[str]):
        return FaceModel(
            id=generate_uuid(),
            identifier=get_idetifier(name),
            name=name,
            images=images
        )
