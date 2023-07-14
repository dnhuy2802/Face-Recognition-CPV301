### Get all the identifiers ###

import os
import base64
from ...utils.constants import CACHE_DIR
from ...models import FaceModel


def get_images():
    # Return the names of the people in the cache directory
    ids = os.listdir(CACHE_DIR)
    for id in ids:
        # Get the path of the person
        path = os.path.join(CACHE_DIR, id)
        # Get first image of the person
        img_paths = os.listdir(path)
        img_path = os.path.join(path, img_paths[0])
        # Read the image
        with open(img_path, 'rb') as f:
            img = base64.b64encode(f.read()).decode('utf-8')
        # Return the name and image of the person
        yield FaceModel.create(id, [img])
