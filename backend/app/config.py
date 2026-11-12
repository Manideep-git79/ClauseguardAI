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

    from urllib.parse import urlparse

    raw_prod_db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://clauseguard:clauseguard@localhost:5432/clauseguard"
    )
    if raw_prod_db_url.startswith("postgres://"):
        raw_prod_db_url = raw_prod_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif raw_prod_db_url.startswith("postgresql://"):
        raw_prod_db_url = raw_prod_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    parsed_prod = urlparse(raw_prod_db_url)
    prod_host = (parsed_prod.hostname or "").lower()

    TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
    TEST_DATABASE_SCHEMA = os.environ.get("TEST_DATABASE_SCHEMA")
    IS_TESTING = (
        os.environ.get("TESTING", "").lower() in ("true", "1", "yes")
        or bool(TEST_DATABASE_URL)
        or bool(TEST_DATABASE_SCHEMA)
        or bool(os.environ.get("FLASK_ENV") == "testing")
    )

    if IS_TESTING:
        # 1. Require TEST_DATABASE_URL whenever TESTING is enabled
        # Never allow TEST_DATABASE_SCHEMA alone to select DATABASE_URL
        if not TEST_DATABASE_URL:
            raise RuntimeError(
                "CRITICAL CONFIGURATION ERROR: TEST_DATABASE_URL is strictly required whenever TESTING is enabled. "
                "Silent fallback to production DATABASE_URL or selecting DATABASE_URL via TEST_DATABASE_SCHEMA "
                "alone is strictly prohibited."
            )

        raw_test_db_url = TEST_DATABASE_URL
        if raw_test_db_url.startswith("postgres://"):
            raw_test_db_url = raw_test_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif raw_test_db_url.startswith("postgresql://"):
            raw_test_db_url = raw_test_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

        parsed_test = urlparse(raw_test_db_url)
        test_host = (parsed_test.hostname or "").lower()

        # 2. Reject production database connection unconditionally, even when TEST_DATABASE_SCHEMA is supplied!
        if raw_test_db_url.strip() == raw_prod_db_url.strip():
            raise RuntimeError(
                "CRITICAL SAFETY VIOLATION: TEST_DATABASE_URL is identical to the production DATABASE_URL! "
                "Tests are strictly forbidden from connecting to the production database string, "
                "even when TEST_DATABASE_SCHEMA is supplied. A dedicated test branch or database is required."
            )

        # 3. Reject connections sharing the production Neon endpoint/host
        if prod_host and ("neon.tech" in prod_host or "aws.neon.tech" in prod_host):
            prod_endpoint_prefix = prod_host.split(".")[0].replace("-pooler", "")
            if prod_endpoint_prefix and prod_endpoint_prefix in test_host:
                raise RuntimeError(
                    f"CRITICAL SAFETY VIOLATION: TEST_DATABASE_URL targets production compute endpoint '{prod_host}'! "
                    "Tests must run on a dedicated test branch endpoint, not the production branch. "
                    "Execution halted to protect production data."
                )

        effective_db_url = raw_test_db_url
    else:
        # Normal production application traffic remains 100% unchanged
        effective_db_url = raw_prod_db_url

    SQLALCHEMY_DATABASE_URI = effective_db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    if effective_db_url.startswith("sqlite"):
        SQLALCHEMY_ENGINE_OPTIONS = {}
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
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
