from flask import Flask

from uci_model.api.config import get_logger

app = Flask(__name__)

_logger = get_logger(logger_name=__name__)


def create_app(*, config_object) -> Flask:
    """Создать Flask приложение"""

    flask_app = Flask('ml_api')
    flask_app.config.from_object(config_object)

    # импорт blueprints
    from uci_model.api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    _logger.debug('Application instance created')

    return flask_app
