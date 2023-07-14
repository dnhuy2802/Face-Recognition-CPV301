### IdentityModel ###
# Used to store the data of the person

from ..utils.utils import convert_to_snake_case, generate_uuid


class IdentityModel:
    def __init__(self, id, identifier: str, name: str, images: list[str]):
        self.id = id
        self.identifier = identifier
        self.name = name
        self.images = images

    @staticmethod
    def from_json(json):
        return IdentityModel.create(
            json['name'],
            json['images']
        )

    @staticmethod
    def create(name: str, images: list[str]):
        return IdentityModel(
            id=generate_uuid(),
            identifier=convert_to_snake_case(name),
            name=name,
            images=images
        )
