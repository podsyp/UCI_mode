from flask import Blueprint, request, jsonify
from uci_model.predict import make_prediction
from uci_model import __version__ as _version
import os
from werkzeug.utils import secure_filename

import numpy as np

from uci_model.api.config import get_logger, UPLOAD_FOLDER
from uci_model.api.validation import validate_inputs, allowed_file
from uci_model.api import __version__ as api_version

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)

@prediction_app.route('/')
def home_endpoint():
    return 'Hello World!'


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Обработка данных из POST запроса как JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Валидирование инпута
        input_data, errors = validate_inputs(input_data=json_data)

        # Предикт модели
        result = make_prediction(input_data=input_data)
        _logger.debug(f'Outputs: {result}')

        # Конверт массива в лист
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Возвращение результата как JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})
