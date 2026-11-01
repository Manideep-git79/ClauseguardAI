import unittest
import io
import sys
import datetime
import jwt
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.middleware.auth import get_jwt_secret

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

        # Ensure a test user exists for authenticated test endpoints
        test_email = "test_api_runner@clauseguard.ai"
        user = User.query.filter_by(email=test_email).first()
        if not user:
            user = User(
                name="Test API Runner",
                email=test_email,
                password_hash="test_password_hash",
                role="user"
            )
            db.session.add(user)
            db.session.commit()
        self.user = user

        payload = {
            "userId": str(user.id),
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1),
        }
        token = jwt.encode(payload, get_jwt_secret(), algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def tearDown(self):
        self.app_context.pop()

    def test_health_check(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertEqual(data["service"], "ClauseGuard AI")

    def test_unauthenticated_list_documents_rejected(self):
        res = self.client.get("/api/documents")
        self.assertEqual(res.status_code, 401)
        self.assertIn("Authentication required", res.get_json().get("error", ""))

    def test_list_documents(self):
        res = self.client.get("/api/documents", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.get_json(), list)

    def test_upload_missing_file(self):
        res = self.client.post("/api/documents", data={}, headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn("No file uploaded", res.get_json()["error"])

    def test_upload_invalid_doc_type(self):
        data = {
            "file": (io.BytesIO(b"Sample legal text"), "test.txt"),
            "document_type": "arbitrary_unsupported_contract",
        }
        res = self.client.post("/api/documents", data=data, content_type="multipart/form-data", headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid document_type", res.get_json()["error"])

    def test_upload_unsupported_extension(self):
        data = {
            "file": (io.BytesIO(b"Executable script"), "malicious.exe"),
            "document_type": "rental_agreement",
        }
        res = self.client.post("/api/documents", data=data, content_type="multipart/form-data", headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.get_json()["error"])

if __name__ == "__main__":
    unittest.main()

