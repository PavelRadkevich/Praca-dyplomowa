from flask import Blueprint

def init_app(app):
    from .home.home import home_bp
    from .predict.predict import predict_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(predict_bp)
