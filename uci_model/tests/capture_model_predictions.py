import pandas as pd

from uci_model.predict import make_prediction
from uci_model.processing.data_management import load_dataset

from uci_model.api import config


def capture_predictions() -> None:

    save_file = 'test_data_predictions.csv'
    test_data = load_dataset(file_name='heart.csv')

    multiple_test_input = test_data[5:35]

    predictions = make_prediction(input_data=multiple_test_input)

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv(f'{config.PACKAGE_ROOT}/{save_file}')


if __name__ == '__main__':
    capture_predictions()
