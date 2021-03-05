from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from uci_model.processing import preprocessors as pp
from uci_model.config import config


pipe = Pipeline([
    ('new_feats', pp.NewFeats(numerator=config.NUMERATOR_FEATURES, denominator=config.DENOMINATOR_FEATURES)),
    ('mean_encod', pp.MeanEncoding(feature=config.CATEGORICAL_FEATURES, C=config.MEAN_ENCODING_C)),
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegressionCV(random_state=config.SEED)),
])