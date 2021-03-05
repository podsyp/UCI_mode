from uci_model.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:

    validated_data = input_data.copy()

    if input_data[config.FEATURES].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.FEATURES
        )

    if (input_data[config.DENOMINATOR_FEATURES] <= 0).any().any():
        denominator_vals = config.DENOMINATOR_FEATURES[
            (input_data[config.DENOMINATOR_FEATURES] <= 0).any()
        ]
        validated_data = validated_data[validated_data[denominator_vals] > 0]

    return validated_data
