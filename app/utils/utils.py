### Utils ###

import uuid
from unidecode import unidecode


def generate_uuid():
    return str(uuid.uuid4())


def get_idetifier(text: str):
    return unidecode(text.replace(" ", "_").lower())
