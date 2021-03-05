import math

from uci_model.config import config as model_config
from uci_model.predict import make_prediction
from uci_model.processing.data_management import load_dataset
import pandas as pd
import pytest


from uci_model.api import config


@pytest.mark.differential
def test_model_prediction_differential(
        *,
        save_file: str = 'test_data_predictions.csv'):

    previous_model_df = pd.read_csv(f'{config.PACKAGE_ROOT}/{save_file}')
    previous_model_predictions = previous_model_df.predictions.values

    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    multiple_test_input = test_data[5:35]

    current_result = make_prediction(input_data=multiple_test_input)
    current_model_predictions = current_result.get('predictions')


    assert len(previous_model_predictions) == len(
        current_model_predictions)

    for previous_value, current_value in zip(
            previous_model_predictions, current_model_predictions):

        previous_value = previous_value.item()
        current_value = current_value.item()


        assert math.isclose(previous_value,
                            current_value,
                            rel_tol=model_config.ACCEPTABLE_MODEL_DIFFERENCE)
