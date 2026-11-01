import os
import unittest
import io
import sys
import uuid
import datetime
import jwt
from pathlib import Path
from cryptography.fernet import Fernet

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.config import Config
from app.models.user import User
from app.models.document import Document
from app.middleware.auth import get_jwt_secret
from app.utils.encryption import (
    encrypt_text,
    decrypt_text,
    get_fernet,
)

class TestSecurityTenantIsolation(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

        # Clean up / set up test users
        self.user_a = self._get_or_create_user("tenant_a@clauseguard.ai", "Tenant A", "user")
        self.user_b = self._get_or_create_user("tenant_b@clauseguard.ai", "Tenant B", "user")
        self.admin = self._get_or_create_user("tenant_admin@clauseguard.ai", "Tenant Admin", "admin")

        self.token_a = self._create_token(self.user_a.id)
        self.token_b = self._create_token(self.user_b.id)
        self.token_admin = self._create_token(self.admin.id)

        self.headers_a = {"Authorization": f"Bearer {self.token_a}"}
        self.headers_b = {"Authorization": f"Bearer {self.token_b}"}
        self.headers_admin = {"Authorization": f"Bearer {self.token_admin}"}

        # Create a document owned by User A
        self.doc_a = Document(
            id=uuid.uuid4(),
            filename="tenant_a_agreement.pdf",
            original_format="pdf",
            document_type="rental_agreement",
            status="analyzed",
            user_id=self.user_a.id,
            raw_text_encrypted=encrypt_text("Confidential agreement for Tenant A."),
            redacted_text="Confidential agreement for [PERSON_1].",
            overall_risk_score=25.0,
            risk_band="Low",
        )
        db.session.add(self.doc_a)

        # Create a document owned by User B
        self.doc_b = Document(
            id=uuid.uuid4(),
            filename="tenant_b_employment.pdf",
            original_format="pdf",
            document_type="employment_contract",
            status="analyzed",
            user_id=self.user_b.id,
            raw_text_encrypted=encrypt_text("Confidential contract for Tenant B."),
            redacted_text="Confidential contract for [PERSON_2].",
            overall_risk_score=45.0,
            risk_band="Medium",
        )
        db.session.add(self.doc_b)
        db.session.commit()

    def tearDown(self):
        # Clean up test documents
        try:
            if hasattr(self, "doc_a") and self.doc_a:
                d = db.session.get(Document, self.doc_a.id)
                if d:
                    db.session.delete(d)
            if hasattr(self, "doc_b") and self.doc_b:
                d = db.session.get(Document, self.doc_b.id)
                if d:
                    db.session.delete(d)
            db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            db.session.remove()
            self.app_context.pop()

    def _get_or_create_user(self, email, name, role):
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(
                name=name,
                email=email,
                password_hash="test_security_hash_123",
                role=role,
            )
            db.session.add(user)
            db.session.commit()
        return user

    def _create_token(self, user_id):
        payload = {
            "userId": str(user_id),
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1),
        }
        return jwt.encode(payload, get_jwt_secret(), algorithm="HS256")

    # 1. Unauthenticated requests are rejected with 401
    def test_unauthenticated_document_access_rejected(self):
        endpoints = [
            ("GET", "/api/documents"),
            ("POST", "/api/documents"),
            ("GET", f"/api/documents/{self.doc_a.id}"),
            ("DELETE", f"/api/documents/{self.doc_a.id}"),
            ("GET", f"/api/documents/{self.doc_a.id}/status"),
            ("POST", f"/api/documents/{self.doc_a.id}/reanalyze"),
            ("GET", f"/api/documents/{self.doc_a.id}/clauses"),
            ("GET", f"/api/documents/{self.doc_a.id}/deadlines"),
            ("GET", f"/api/documents/{self.doc_a.id}/pii-summary"),
            ("GET", f"/api/documents/{self.doc_a.id}/exclusions-summary"),
            ("GET", f"/api/documents/{self.doc_a.id}/missing-clauses"),
            ("GET", f"/api/documents/{self.doc_a.id}/report.pdf"),
            ("POST", f"/api/documents/{self.doc_a.id}/chat"),
            ("GET", f"/api/documents/{self.doc_a.id}/chat/history"),
            ("POST", "/api/chat"),
            ("GET", "/api/chat/history"),
        ]

        for method, url in endpoints:
            if method == "GET":
                res = self.client.get(url)
            elif method == "POST":
                res = self.client.post(url, json={})
            elif method == "DELETE":
                res = self.client.delete(url)
            self.assertEqual(
                res.status_code, 401,
                f"Expected 401 for unauthenticated {method} {url}, got {res.status_code}"
            )

    # 2. Cross-tenant access is forbidden with 403
    def test_cross_tenant_document_access_forbidden(self):
        # User B trying to access User A's document endpoints
        endpoints = [
            ("GET", f"/api/documents/{self.doc_a.id}"),
            ("DELETE", f"/api/documents/{self.doc_a.id}"),
            ("GET", f"/api/documents/{self.doc_a.id}/status"),
            ("POST", f"/api/documents/{self.doc_a.id}/reanalyze"),
            ("GET", f"/api/documents/{self.doc_a.id}/clauses"),
            ("GET", f"/api/documents/{self.doc_a.id}/deadlines"),
            ("GET", f"/api/documents/{self.doc_a.id}/pii-summary"),
            ("GET", f"/api/documents/{self.doc_a.id}/exclusions-summary"),
            ("GET", f"/api/documents/{self.doc_a.id}/missing-clauses"),
            ("GET", f"/api/documents/{self.doc_a.id}/report.pdf"),
            ("POST", f"/api/documents/{self.doc_a.id}/chat"),
            ("GET", f"/api/documents/{self.doc_a.id}/chat/history"),
        ]

        for method, url in endpoints:
            if method == "GET":
                res = self.client.get(url, headers=self.headers_b)
            elif method == "POST":
                res = self.client.post(url, json={"message": "test"}, headers=self.headers_b)
            elif method == "DELETE":
                res = self.client.delete(url, headers=self.headers_b)
            self.assertEqual(
                res.status_code, 403,
                f"Expected 403 for cross-tenant {method} {url}, got {res.status_code}"
            )

    # 3. Tenant listing isolation: User A only sees A's docs, User B only sees B's docs
    def test_tenant_listing_isolation(self):
        res_a = self.client.get("/api/documents", headers=self.headers_a)
        self.assertEqual(res_a.status_code, 200)
        docs_a = res_a.get_json()
        ids_a = [d["id"] for d in docs_a]
        self.assertIn(str(self.doc_a.id), ids_a)
        self.assertNotIn(str(self.doc_b.id), ids_a)

        res_b = self.client.get("/api/documents", headers=self.headers_b)
        self.assertEqual(res_b.status_code, 200)
        docs_b = res_b.get_json()
        ids_b = [d["id"] for d in docs_b]
        self.assertIn(str(self.doc_b.id), ids_b)
        self.assertNotIn(str(self.doc_a.id), ids_b)

    # 4. Admin override allows access to documents across tenants
    def test_admin_override_access(self):
        res = self.client.get(f"/api/documents/{self.doc_a.id}", headers=self.headers_admin)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["id"], str(self.doc_a.id))

        res_list = self.client.get("/api/documents", headers=self.headers_admin)
        self.assertEqual(res_list.status_code, 200)
        all_ids = [d["id"] for d in res_list.get_json()]
        self.assertIn(str(self.doc_a.id), all_ids)
        self.assertIn(str(self.doc_b.id), all_ids)

    # 5. Empty / 0-byte upload rejection
    def test_zero_byte_upload_rejected(self):
        data = {
            "file": (io.BytesIO(b""), "empty.pdf"),
            "document_type": "rental_agreement",
        }
        res = self.client.post("/api/documents", data=data, content_type="multipart/form-data", headers=self.headers_a)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Uploaded file is empty (0 bytes).", res.get_json()["error"])

    # 6. Encryption key persistence and deterministic behavior
    def test_encryption_roundtrip_with_configured_key(self):
        # Standard round-trip with configured key
        encrypted_new = encrypt_text("New standard encrypted clause.")
        self.assertEqual(decrypt_text(encrypted_new), "New standard encrypted clause.")

        # Cipher does not change between calls
        c1 = get_fernet()
        c2 = get_fernet()
        self.assertEqual(c1._signing_key, c2._signing_key)

    # 7. Multi-key legacy recovery using configured fallback keys
    def test_encryption_legacy_key_multi_key_recovery(self):
        # Generate an arbitrary legacy key
        legacy_key = Fernet.generate_key().decode()
        legacy_cipher = Fernet(legacy_key.encode())
        sample_text = "Historical confidential document encrypted under legacy key."
        legacy_ciphertext = legacy_cipher.encrypt(sample_text.encode("utf-8"))

        # Configure the legacy key in FIELD_ENCRYPTION_FALLBACK_KEYS
        orig_fallbacks = self.app.config.get("FIELD_ENCRYPTION_FALLBACK_KEYS")
        try:
            self.app.config["FIELD_ENCRYPTION_FALLBACK_KEYS"] = legacy_key
            decrypted = decrypt_text(legacy_ciphertext)
            self.assertEqual(decrypted, sample_text)

            # Without the configured fallback key, decryption must fail safely
            self.app.config["FIELD_ENCRYPTION_FALLBACK_KEYS"] = ""
            with self.assertRaises(Exception):
                decrypt_text(legacy_ciphertext)
        finally:
            self.app.config["FIELD_ENCRYPTION_FALLBACK_KEYS"] = orig_fallbacks

    # 8. Unconfigured encryption key fails safely without hardcoded fallbacks
    def test_missing_encryption_key_fails_safely(self):
        orig_key = self.app.config.get("FIELD_ENCRYPTION_KEY")
        orig_env_key = os.environ.get("FIELD_ENCRYPTION_KEY")
        try:
            self.app.config["FIELD_ENCRYPTION_KEY"] = None
            if "FIELD_ENCRYPTION_KEY" in os.environ:
                del os.environ["FIELD_ENCRYPTION_KEY"]

            with self.assertRaises(RuntimeError) as ctx:
                encrypt_text("Text that cannot be encrypted without key")
            self.assertIn("FIELD_ENCRYPTION_KEY is required", str(ctx.exception))
        finally:
            self.app.config["FIELD_ENCRYPTION_KEY"] = orig_key
            if orig_env_key is not None:
                os.environ["FIELD_ENCRYPTION_KEY"] = orig_env_key

    # 9. Production startup fails if JWT_SECRET or FIELD_ENCRYPTION_KEY is missing
    def test_missing_production_secrets_fails_startup(self):
        class MissingJwtConfig(Config):
            FLASK_ENV = "production"
            JWT_SECRET = None
            SECRET_KEY = None
            FIELD_ENCRYPTION_KEY = Fernet.generate_key().decode()

        class MissingEncryptionKeyConfig(Config):
            FLASK_ENV = "production"
            JWT_SECRET = "production_jwt_secret_valid_123"
            SECRET_KEY = "production_jwt_secret_valid_123"
            FIELD_ENCRYPTION_KEY = None

        orig_flask_env = os.environ.get("FLASK_ENV")
        orig_jwt_secret = os.environ.get("JWT_SECRET")
        orig_secret_key = os.environ.get("SECRET_KEY")
        orig_enc_key = os.environ.get("FIELD_ENCRYPTION_KEY")
        try:
            os.environ["FLASK_ENV"] = "production"
            if "JWT_SECRET" in os.environ:
                del os.environ["JWT_SECRET"]
            if "SECRET_KEY" in os.environ:
                del os.environ["SECRET_KEY"]

            with self.assertRaises(RuntimeError) as ctx1:
                create_app(MissingJwtConfig)
            self.assertIn("CRITICAL SECURITY CONFIGURATION ERROR", str(ctx1.exception))
            self.assertIn("JWT_SECRET", str(ctx1.exception))

            os.environ["JWT_SECRET"] = "production_jwt_secret_valid_123"
            os.environ["SECRET_KEY"] = "production_jwt_secret_valid_123"
            if "FIELD_ENCRYPTION_KEY" in os.environ:
                del os.environ["FIELD_ENCRYPTION_KEY"]

            with self.assertRaises(RuntimeError) as ctx2:
                create_app(MissingEncryptionKeyConfig)
            self.assertIn("CRITICAL SECURITY CONFIGURATION ERROR", str(ctx2.exception))
            self.assertIn("FIELD_ENCRYPTION_KEY", str(ctx2.exception))
        finally:
            if orig_flask_env is not None:
                os.environ["FLASK_ENV"] = orig_flask_env
            else:
                os.environ.pop("FLASK_ENV", None)
            if orig_jwt_secret is not None:
                os.environ["JWT_SECRET"] = orig_jwt_secret
            else:
                os.environ.pop("JWT_SECRET", None)
            if orig_secret_key is not None:
                os.environ["SECRET_KEY"] = orig_secret_key
            else:
                os.environ.pop("SECRET_KEY", None)
            if orig_enc_key is not None:
                os.environ["FIELD_ENCRYPTION_KEY"] = orig_enc_key
            else:
                os.environ.pop("FIELD_ENCRYPTION_KEY", None)

    # 10. Missing JWT secret at runtime safely rejects requests
    def test_missing_jwt_secret_rejects_protected_operation(self):
        orig_jwt = self.app.config.get("JWT_SECRET")
        orig_sec = self.app.config.get("SECRET_KEY")
        orig_env_jwt = os.environ.get("JWT_SECRET")
        orig_env_sec = os.environ.get("SECRET_KEY")
        try:
            self.app.config["JWT_SECRET"] = None
            self.app.config["SECRET_KEY"] = None
            if "JWT_SECRET" in os.environ:
                del os.environ["JWT_SECRET"]
            if "SECRET_KEY" in os.environ:
                del os.environ["SECRET_KEY"]

            res = self.client.get("/api/documents", headers={"Authorization": "Bearer some_dummy_token"})
            self.assertEqual(res.status_code, 500)
            self.assertIn("JWT secret is not configured", res.get_json().get("error", ""))
        finally:
            self.app.config["JWT_SECRET"] = orig_jwt
            self.app.config["SECRET_KEY"] = orig_sec
            if orig_env_jwt is not None:
                os.environ["JWT_SECRET"] = orig_env_jwt
            if orig_env_sec is not None:
                os.environ["SECRET_KEY"] = orig_env_sec

    # 11. Canonical JWT secret consistency
    def test_secret_consistency_between_jwt_secret_and_secret_key(self):
        canonical_secret = get_jwt_secret()
        self.assertIsNotNone(canonical_secret)
        self.assertEqual(canonical_secret, self.app.config.get("JWT_SECRET"))

if __name__ == "__main__":
    unittest.main()
