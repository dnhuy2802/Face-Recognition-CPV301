### Utils ###

import uuid


def generate_uuid():
    return str(uuid.uuid4())


def convert_to_snake_case(text: str):
    return text.replace(" ", "_").lower()
