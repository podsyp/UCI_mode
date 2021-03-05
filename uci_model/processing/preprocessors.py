import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#from uci_model.processing.errors import InvalidModelInputError


class NewFeats(BaseEstimator, TransformerMixin):

    def __init__(self, numerator=None, denominator=None) -> None:
        if not isinstance(numerator, list):
            self.numerator = [numerator]
        else:
            self.numerator = numerator

        if not isinstance(denominator, list):
            self.denominator = [denominator]
        else:
            self.denominator = denominator

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NewFeats":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()
        for num in self.numerator:
            for den in self.denominator:
                X[num + '_' + den] = X[num] / X[den]

        return X


class MeanEncoding(BaseEstimator, TransformerMixin):
    """   In Mean Encoding we take the number
    of labels into account along with the target variable
    to encode the labels into machine comprehensible values    """

    def __init__(self, feature, C=0.1):
        self.C = C
        if not isinstance(feature, list):
            self.feature = [feature]
        else:
            self.feature = feature

    def fit(self, X_train, y_train):

        self.encoding = dict()

        for f in self.feature:
            df = pd.DataFrame({'feature': X_train[f], 'target': y_train}).dropna()
            global_mean = df.target.mean()
            mean = df.groupby('feature').target.mean()
            size = df.groupby('feature').target.size()

            encoding = (global_mean * self.C + mean * size) / (self.C + size)

            self.encoding[f] = {'global_mean': global_mean, 'encoding': encoding}
        return self

    def transform(self, X_test):
        for f in self.feature:
            X_test[f] = X_test[f].map(self.encoding[f]['encoding']).fillna(self.encoding[f]['global_mean']).values

        return X_test
