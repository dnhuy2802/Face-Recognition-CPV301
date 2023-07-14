### Delete Identifier ###

import os
import shutil
from .constants import CACHE_DIR


def delete_identifier(name: str):
    # Delete the identifier from the cache directory
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        shutil.rmtree(path)
