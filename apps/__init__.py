from flask import Blueprint, Flask
from flask_socketio import SocketIO

socketio = SocketIO()


def create_app():
    app = Flask(__name__, static_folder="static")
    from .home.home import home_bp
    from .predict.predict import predict_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(predict_bp)

    socketio.init_app(app,  cors_allowed_origins="*")
    return app
