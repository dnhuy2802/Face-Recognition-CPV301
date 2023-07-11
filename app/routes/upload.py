from flask import Blueprint, request
from ..models import IdentityModel, ResponseObject
from ..utils.save_images import save_images, resize_images
from ..utils.get_identifiers import get_identifiers
from ..utils.delete_identifier import delete_identifier

### Create Blueprint ###
upload_bp = Blueprint('upload', __name__, url_prefix='/upload')


### Routes ###
@upload_bp.route('/images', methods=['POST'])
def upload():
    # Get the image from the request
    person = IdentityModel.from_json(request.get_json())
    # Save the image to resource folder
    path = save_images(person.name, person.images)
    # Resize the images
    resize_images(path)
    # Return the id of the person
    return ResponseObject.success(person.id)


@upload_bp.route('/identifiers', methods=['GET'])
def identifiers():
    identifiers = []
    for identifier in get_identifiers():
        identifiers.append({
            'id': identifier.id,
            'name': identifier.name,
            'thumbnail': identifier.images[0]
        })
    return ResponseObject.success(identifiers)


@upload_bp.route('/delete/<id>', methods=['DELETE'])
def delete(id):
    delete_identifier(id)
    return ResponseObject.success(True)
