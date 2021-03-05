import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from uci_model.processing.data_management import load_pipeline, load_dataset

from uci_model.config import config
from uci_model.processing.validation import validate_inputs
from uci_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)
input_data = load_dataset(file_name=config.TRAINING_DATA_FILE)


def make_prediction(input_data: dict):
    """Предсказание с использованием сохраненного pipeline

    Аргументы:
        input_data: Данные модели

    Возвращает:
        Вектор предсказаний модели
    """
    data = pd.DataFrame(input_data)

    validated_data = validate_inputs(input_data=data)

    prediction = _pipe.predict(validated_data[config.FEATURES])

    results = {"predictions": prediction, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )

    return results

def make_validation():
    """Предсказание с использованием сохраненного pipeline

    Аргументы:
        input_data: Данные модели

    Возвращает:
        Вектор предсказаний модели
    """
    data = pd.DataFrame(input_data)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=config.SPLIT_SIZE, random_state=config.SEED
    )
    validated_data = validate_inputs(input_data=X_test)

    prediction = _pipe.predict(validated_data[config.FEATURES])

    results = {"predictions": prediction, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )
    print("test accuracy", accuracy_score(y_test, prediction))

    return None
