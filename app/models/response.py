import typing
from flask import jsonify


class ResponseObject:
    @staticmethod
    def success(data: typing.Any | dict[str, typing.Any], json: bool = True):
        response = {
            'success': True,
            'data': data
        }
        if json:
            return jsonify(response)
        return response

    @staticmethod
    def error(message: str, json: bool = True):
        response = {
            'success': False,
            'message': message
        }
        if json:
            return jsonify(response)
        return response


class FaceResponseObject:
    def __init__(self, name, confidence, cordinates):
        self.name = name
        self.confidence = confidence
        self.cordinates = cordinates

    def to_dict(self):
        return {
            'name': self.name,
            'confidence': self.confidence,
            'cordinates': self.cordinates
        }
