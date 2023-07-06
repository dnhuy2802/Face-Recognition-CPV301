from flask import Blueprint, request
from ..models import FaceModel, ResponseObject
from ..logic.register_faces.save_images import save_images, resize_images
from ..logic.register_faces.get_images import get_images
from ..logic.register_faces.delete_images import delete_images


### Create Blueprint ###
register_faces_bp = Blueprint(
    'register_faces', __name__, url_prefix='/register_faces')


### Routes ###
@register_faces_bp.route('/new', methods=['POST'])
def upload_faces():
    # Get the image from the request
    person = FaceModel.from_json(request.get_json())
    # Save the image to resource folder
    path = save_images(person.name, person.images)
    # Resize the images
    resize_images(path)
    # Return the id of the person
    return ResponseObject.success(person.identifier)


@register_faces_bp.route('/', methods=['GET'])
def get_faces():
    faces = []
    for face in get_images():
        faces.append({
            'id': face.id,
            'identifier': face.identifier,
            'name': face.name,
            'thumbnail': face.images[0]
        })
    return ResponseObject.success(faces)


@register_faces_bp.route('/<name>', methods=['DELETE'])
def delete_faces(name):
    delete_images(name)
    return ResponseObject.success(True)
