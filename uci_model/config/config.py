import pathlib

import uci_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(uci_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

MEAN_ENCODING_C = 0.99

# Данные
TRAINING_DATA_FILE = "heart.csv"
TARGET = "target"

# Переменные
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal']

NUMERATOR_FEATURES = ['trestbps', 'chol', 'thalach']

DENOMINATOR_FEATURES = ['age']

CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'ca', 'thal']

PIPELINE_NAME = "logistic_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

ACCEPTABLE_MODEL_DIFFERENCE = 0.05

SEED = 777
SPLIT_SIZE = 0.15
