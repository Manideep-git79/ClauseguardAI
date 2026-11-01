import os
from pathlib import Path
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(backend_dir / ".env")

class Config:
    FLASK_ENV = os.environ.get("FLASK_ENV", "development")

    raw_secret_key = os.environ.get("SECRET_KEY")
    raw_jwt_secret = os.environ.get("JWT_SECRET")

    # Canonicalize JWT secret: prefer dedicated JWT_SECRET, fallback to SECRET_KEY
    JWT_SECRET = raw_jwt_secret or raw_secret_key
    # Flask application secret key
    SECRET_KEY = raw_secret_key or raw_jwt_secret

    raw_db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://clauseguard:clauseguard@localhost:5432/clauseguard"
    )
    if raw_db_url.startswith("postgres://"):
        raw_db_url = raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif raw_db_url.startswith("postgresql://"):
        raw_db_url = raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    SQLALCHEMY_DATABASE_URI = raw_db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    if raw_db_url.startswith("sqlite"):
        SQLALCHEMY_ENGINE_OPTIONS = {}
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": 60,
            "pool_timeout": 30,
            "max_overflow": 10,
            "pool_size": 5,
            "connect_args": {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        }
    
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
    GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    
    FIELD_ENCRYPTION_KEY = os.environ.get("FIELD_ENCRYPTION_KEY", "")
    FIELD_ENCRYPTION_FALLBACK_KEYS = os.environ.get("FIELD_ENCRYPTION_FALLBACK_KEYS", "")
    TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")
    
    import re
    frontend_env = os.environ.get("FRONTEND_URL", "")
    cors_env = os.environ.get("CORS_ORIGINS", "")
    allowed_origins = set()
    for item in f"{frontend_env},{cors_env}".split(","):
        s = item.strip().rstrip("/")
        if s:
            allowed_origins.add(s)
    # Always include common dev origins and regex patterns for Vercel/Railway
    allowed_origins.update({
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    })
    allowed_origins.add(re.compile(r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$"))
    allowed_origins.add(re.compile(r"^https://.*\.vercel\.app$"))
    allowed_origins.add(re.compile(r"^https://.*\.railway\.app$"))
    allowed_origins.add(re.compile(r"^https://.*\.up\.railway\.app$"))
    CORS_ORIGINS = list(allowed_origins)
    
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 20))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    BASELINE_MODEL_PATH = os.environ.get("BASELINE_MODEL_PATH", str(MODELS_DIR / "baseline_tfidf_lr.joblib"))
    BERT_CLAUSE_TYPE_MODEL_PATH = os.environ.get("BERT_CLAUSE_TYPE_MODEL_PATH", str(MODELS_DIR / "bert_clause_type" / "final"))
    BERT_FAVORABILITY_MODEL_PATH = os.environ.get("BERT_FAVORABILITY_MODEL_PATH", str(MODELS_DIR / "bert_favorability" / "final"))
