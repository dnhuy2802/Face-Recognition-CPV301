### Delete Images ###

import os
import shutil
from ...utils.constants import CACHE_DIR


def delete_images(name: str):
    # Delete the identifier from the cache directory
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        shutil.rmtree(path)
