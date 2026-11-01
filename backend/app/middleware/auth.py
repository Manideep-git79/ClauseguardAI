import jwt
from functools import wraps
from flask import request, jsonify, g, current_app
from app.extensions import db
from app.models.user import User, Session

import os
import logging

logger = logging.getLogger(__name__)

def get_jwt_secret():
    """
    Returns the canonical JWT signing and verification secret.
    Prioritizes JWT_SECRET from app config or environment, falling back to SECRET_KEY.
    In production, fails immediately if no valid secret is configured.
    Never logs secret values.
    """
    secret = None
    if current_app:
        secret = current_app.config.get("JWT_SECRET") or current_app.config.get("SECRET_KEY")
    if not secret:
        secret = os.environ.get("JWT_SECRET") or os.environ.get("SECRET_KEY")

    if not secret:
        is_prod = (
            (current_app and current_app.config.get("FLASK_ENV") == "production")
            or os.environ.get("FLASK_ENV") == "production"
        )
        if is_prod:
            logger.error("Authentication halted: No JWT_SECRET or SECRET_KEY configured in production.")
            raise RuntimeError("CRITICAL SECURITY ERROR: No valid JWT signing secret configured in production environment.")
        logger.warning("No JWT signing secret configured in environment.")
        return None

    return secret

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif request.cookies.get("token"):
            token = request.cookies.get("token")
        elif request.args.get("token"):
            token = request.args.get("token")
        elif request.args.get("auth_token"):
            token = request.args.get("auth_token")

        if not token:
            return jsonify({"error": "Authentication required. No token provided."}), 401

        jwt_secret = get_jwt_secret()
        if not jwt_secret:
            return jsonify({"error": "Authentication service unavailable: JWT secret is not configured."}), 500

        try:
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        except Exception:
            resp = jsonify({"error": "Invalid or expired session token."})
            resp.set_cookie("token", "", expires=0, path="/", httponly=True)
            return resp, 401

        session_id = payload.get("sessionId")
        user_id = payload.get("userId")

        if not user_id:
            resp = jsonify({"error": "Invalid token payload."})
            resp.set_cookie("token", "", expires=0, path="/", httponly=True)
            return resp, 401

        if session_id:
            session_obj = db.session.get(Session, session_id)
            if session_obj and session_obj.revoked:
                resp = jsonify({"error": "Session has been revoked or expired."})
                resp.set_cookie("token", "", expires=0, path="/", httponly=True)
                return resp, 401
            g.session = session_obj

        user_obj = db.session.get(User, user_id)
        if not user_obj:
            return jsonify({"error": "User no longer exists."}), 401

        g.user = user_obj
        return f(*args, **kwargs)
    return decorated_function

def optional_auth(f):
    """Optionally attaches g.user and g.session if a valid token is present."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.user = None
        g.session = None
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif request.cookies.get("token"):
            token = request.cookies.get("token")
        elif request.args.get("token"):
            token = request.args.get("token")
        elif request.args.get("auth_token"):
            token = request.args.get("auth_token")

        if token:
            jwt_secret = get_jwt_secret()
            if jwt_secret:
                try:
                    payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                    session_id = payload.get("sessionId")
                    user_id = payload.get("userId")
                    if user_id:
                        user_obj = db.session.get(User, user_id)
                        if user_obj:
                            if session_id:
                                session_obj = db.session.get(Session, session_id)
                                if session_obj and not session_obj.revoked:
                                    g.session = session_obj
                                    g.user = user_obj
                                elif not session_obj:
                                    g.user = user_obj
                            else:
                                g.user = user_obj
                except Exception:
                    pass
        return f(*args, **kwargs)
    return decorated_function
