from flask import Blueprint

def init_app(app):
    from .home.home import home_bp

    app.register_blueprint(home_bp)
