import typing as t

from marshmallow import Schema, fields
from marshmallow import ValidationError

from uci_model.api import config


class InvalidInputError(Exception):
    SYNTAX_ERROR_FIELD_MAP = {
        'age': 0,
        'sex': 1,
        'cp': 2,
        'trestbps': 3,
        'chol': 4,
        'fbs': 5,
        'restecg': 6,
        'thalach': 7,
        'exang': 8,
        'oldpeak': 9,
        'slope': 10,
        'ca': 11,
        'thal': 12
    }


class UCISchema(Schema):

    age = fields.Integer()
    sex = fields.Integer()
    cp = fields.Integer()
    trestbps = fields.Integer()
    chol = fields.Integer()
    fbs = fields.Integer()
    restecg = fields.Integer()
    thalach = fields.Integer()
    exang = fields.Integer()
    oldpeak = fields.Float()
    slope = fields.Integer()
    ca = fields.Integer()
    thal = fields.Integer()


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:

    indexes = errors.keys()
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):

    schema = UCISchema(many=True)
    error = InvalidInputError()

    for dict in input_data:
        for key, value in error.SYNTAX_ERROR_FIELD_MAP.items():
            dict[value] = dict[key]
            del dict[key]

    for dict in input_data:
        for key, value in error.SYNTAX_ERROR_FIELD_MAP.items():
            dict[key] = dict[value]
            del dict[value]


    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS
