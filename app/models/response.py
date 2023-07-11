import typing
from flask import jsonify


class ResponseObject:
    @staticmethod
    def success(data: typing.Any | dict[str, typing.Any]):
        return jsonify({
            'success': True,
            'data': data
        })

    @staticmethod
    def error(message: str):
        return jsonify({
            'success': False,
            'message': message
        })
