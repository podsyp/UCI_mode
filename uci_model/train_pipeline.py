import numpy as np
from sklearn.model_selection import train_test_split

from uci_model import pipeline
from uci_model.processing.data_management import load_dataset, save_pipeline
from uci_model.config import config
from uci_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Тренировка модели"""

    # Чтение файла
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # Разделение датасета на train-test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=config.SPLIT_SIZE, random_state=config.SEED
    )

    pipeline.pipe.fit(X_train[config.FEATURES], y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.pipe)


if __name__ == "__main__":
    run_training()
