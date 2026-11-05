import os
from flask import Flask
from app.config import Config
from app.extensions import db, migrate, cors

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    is_prod = (
        app.config.get("FLASK_ENV") == "production"
        or os.environ.get("FLASK_ENV") == "production"
    )
    if is_prod:
        if not (app.config.get("JWT_SECRET") or app.config.get("SECRET_KEY")):
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: Neither JWT_SECRET nor SECRET_KEY "
                "is configured. Production startup aborted to prevent unauthenticated access."
            )
        if not app.config.get("FIELD_ENCRYPTION_KEY"):
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: FIELD_ENCRYPTION_KEY "
                "is not configured. Production startup aborted to prevent unencrypted or corrupted data at rest."
            )

    db.init_app(app)

    test_schema = app.config.get("TEST_DATABASE_SCHEMA")
    if test_schema:
        import re
        from sqlalchemy import event

        # Validate schema identifier format to prevent injection or malformed values
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", test_schema):
            raise ValueError(
                f"CRITICAL SECURITY: Invalid test schema identifier: {test_schema!r}. "
                "Must be a valid alphanumeric/underscore PostgreSQL identifier."
            )
        quoted_test_schema = f'"{test_schema}"'

        with app.app_context():
            engine = db.engine

            def _enforce_test_schema(dbapi_conn):
                cursor = dbapi_conn.cursor()
                try:
                    # Fail-closed isolation: Strictly use test_schema, NEVER fall back to public!
                    cursor.execute(f"SET search_path TO {quoted_test_schema};")
                finally:
                    cursor.close()

            @event.listens_for(engine, "connect")
            def on_test_connect(dbapi_conn, record):
                _enforce_test_schema(dbapi_conn)

            @event.listens_for(engine, "checkout")
            def on_test_checkout(dbapi_conn, record, proxy):
                _enforce_test_schema(dbapi_conn)

            @event.listens_for(engine, "begin")
            def on_test_begin(conn):
                # Strict fail-closed: If SET search_path fails, raise immediately to abort transaction!
                # Silent exception suppression is strictly prohibited.
                conn.exec_driver_sql(f"SET search_path TO {quoted_test_schema};")
    migrate.init_app(app, db)
    cors.init_app(app, resources={r"/*": {"origins": app.config.get("CORS_ORIGINS", "*")}}, supports_credentials=True)

    from app.api.health import health_bp
    from app.api.auth import auth_bp
    from app.api.documents import documents_bp
    from app.api.analysis import analysis_bp
    from app.api.chat import chat_bp
    from app.api.audit import audit_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(documents_bp, url_prefix="/api")
    app.register_blueprint(analysis_bp, url_prefix="/api")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(audit_bp, url_prefix="/api")

    @app.route("/health", methods=["GET"])
    def root_health():
        from app.api.health import health_check
        return health_check()

    from werkzeug.exceptions import HTTPException

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        from flask import jsonify
        return jsonify({"error": e.description}), e.code

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        if isinstance(error, HTTPException):
            return error
        import logging
        from flask import jsonify, request
        logging.getLogger(__name__).error(f"Unhandled exception on {request.path}: {error}", exc_info=True)
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({"error": "Server error. Please retry in a few moments."}), 500

    return app
