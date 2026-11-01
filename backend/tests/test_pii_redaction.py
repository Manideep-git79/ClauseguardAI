import unittest
import sys
import uuid
import datetime
import jwt
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.document import Document
from app.models.clause import Clause
from app.models.pii_finding import PIIFinding
from app.middleware.auth import get_jwt_secret
from app.services.privacy.pii_redactor import redact, DocumentEntityResolver

class TestPIIRedaction(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        db.session.remove()
        self.app_context.pop()

    # 1. Baseline entity redaction
    def test_redact_entities(self):
        sample = "Tenant Alice Walker will pay Landlord Bob Vance at bob@realty.com or (312) 555-0199."
        redacted_text, findings = redact(sample)

        self.assertNotIn("Alice Walker", redacted_text)
        self.assertNotIn("Bob Vance", redacted_text)
        self.assertNotIn("bob@realty.com", redacted_text)
        self.assertNotIn("312) 555-0199", redacted_text)

        entity_types = [f["entity_type"] for f in findings]
        self.assertIn("EMAIL_ADDRESS", entity_types)
        self.assertTrue(any(t in ["PERSON", "NAME"] for t in entity_types))

    # 2. Date and time preserved for legal deadline engine
    def test_date_time_preserved(self):
        sample = "This lease begins on September 1, 2026 and requires 30 days written notice prior to August 31, 2027."
        redacted_text, findings = redact(sample)

        self.assertIn("September 1, 2026", redacted_text)
        self.assertIn("30 days", redacted_text)
        self.assertIn("August 31, 2027", redacted_text)

        entity_types = [f["entity_type"] for f in findings]
        self.assertNotIn("DATE_TIME", entity_types)

    # 3. Neutral sequential labels generated without contextual roles
    def test_neutral_sequential_labels_generated(self):
        sample = (
            "Tenant Alice Walker agrees to transfer funds to Landlord Bob Vance. "
            "Contact: alice@example.com or phone (555) 019-9283."
        )
        redacted_text, findings = redact(sample)

        # Must contain neutral labels like [Person 1], [Person 2], [Email 1], [Phone 1]
        self.assertIn("[Person 1]", redacted_text)
        self.assertIn("[Person 2]", redacted_text)
        self.assertIn("[Email 1]", redacted_text)
        self.assertIn("[Phone 1]", redacted_text)

        # Must NEVER contain contextual role labels in pseudonyms
        forbidden_roles = ["Tenant", "Landlord", "Employer", "Employee", "Party A", "Party B", "Guarantor", "Witness"]
        for role in forbidden_roles:
            self.assertNotIn(f"[{role}", redacted_text)
            for f in findings:
                self.assertNotIn(role, f.get("replacement_label", ""))

    # 4. Repeated entities reuse the same label within a document
    def test_repeated_entities_reuse_label(self):
        sample = (
            "Alice Walker entered the agreement. Notice must be delivered to Alice Walker. "
            "Send confirmation to alice@example.com, and cc alice@example.com."
        )
        redacted_text, findings = redact(sample)

        # Alice Walker appears twice; both must be [Person 1]
        self.assertEqual(redacted_text.count("[Person 1]"), 2)
        # No [Person 2] should be generated for the same person
        self.assertNotIn("[Person 2]", redacted_text)

        # alice@example.com appears twice; both must be [Email 1]
        self.assertEqual(redacted_text.count("[Email 1]"), 2)
        self.assertNotIn("[Email 2]", redacted_text)

    # 5. Distinct entities sharing a first name are NOT merged
    def test_distinct_entities_not_merged(self):
        resolver = DocumentEntityResolver()
        label1 = resolver.get_or_assign_label("PERSON", "John Smith")
        label2 = resolver.get_or_assign_label("PERSON", "John Doe")
        label3 = resolver.get_or_assign_label("PERSON", "John")

        self.assertEqual(label1, "[Person 1]")
        self.assertEqual(label2, "[Person 2]")
        self.assertEqual(label3, "[Person 3]")
        self.assertNotEqual(label1, label2)
        self.assertNotEqual(label1, label3)
        self.assertNotEqual(label2, label3)

    # 6. Redacted-text offsets align with exact character positions
    def test_redacted_text_offset_alignment(self):
        sample = (
            "Executive Alice Walker (SSN: 000-12-3456) can be reached at "
            "alice.walker@corp.net or (415) 555-2671 located in Chicago."
        )
        redacted_text, findings = redact(sample)

        self.assertGreater(len(findings), 0)
        for f in findings:
            start = f["start"]
            end = f["end"]
            expected_label = f["replacement_label"]
            actual_slice = redacted_text[start:end]
            self.assertEqual(
                actual_slice,
                expected_label,
                f"Offset mismatch: expected '{expected_label}' at [{start}:{end}], found '{actual_slice}'"
            )

    # 7. Adjacent findings and Unicode text handle offset shifts correctly
    def test_adjacent_findings_and_unicode_offsets(self):
        sample = (
            "Contact: Alice Walker, bob@example.com in Zürich. "
            "Associate: René Müller (555) 234-5678."
        )
        redacted_text, findings = redact(sample)

        for f in findings:
            start = f["start"]
            end = f["end"]
            label = f["replacement_label"]
            self.assertEqual(redacted_text[start:end], label)

    # 8. Client-facing PII summary endpoint returns privacy-first mapping table
    def test_pii_summary_endpoint_privacy_first(self):
        # Create test user and token
        user = User.query.filter_by(email="pii_test_user@clauseguard.ai").first()
        if not user:
            user = User(
                name="PII Test User",
                email="pii_test_user@clauseguard.ai",
                password_hash="test_password_hash",
                role="user"
            )
            db.session.add(user)
            db.session.commit()

        token = jwt.encode(
            {"userId": str(user.id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)},
            get_jwt_secret(),
            algorithm="HS256"
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Create a document with redacted text and findings
        raw_synthetic_text = "Alice Walker agreed to terms. Contact alice@example.com."
        redacted_text, pii_findings = redact(raw_synthetic_text)

        doc = Document(
            id=uuid.uuid4(),
            filename="pii_privacy_test.pdf",
            original_format="pdf",
            document_type="rental_agreement",
            status="analyzed",
            user_id=user.id,
            redacted_text=redacted_text,
            overall_risk_score=20.0,
            risk_band="Low",
        )
        db.session.add(doc)
        db.session.flush()

        # Add a clause to test section references
        clause = Clause(
            id=uuid.uuid4(),
            document_id=doc.id,
            clause_index=0,
            clause_type="parties",
            redacted_text=redacted_text,
            start_offset=0,
            end_offset=len(redacted_text),
            risk_score=20.0,
            is_substantive=True,
        )
        db.session.add(clause)

        for f in pii_findings:
            pf = PIIFinding(
                document_id=doc.id,
                entity_type=f["entity_type"],
                start_offset=f["start"],
                end_offset=f["end"],
                confidence=f["confidence"],
            )
            db.session.add(pf)
        db.session.commit()

        try:
            res = self.client.get(f"/api/documents/{doc.id}/pii-summary", headers=headers)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()

            # Check mapping_table presence and structure
            self.assertIn("mapping_table", data)
            mapping_table = data["mapping_table"]
            self.assertGreater(len(mapping_table), 0)

            for item in mapping_table:
                # Required privacy columns
                self.assertIn("replacement_label", item)
                self.assertIn("entity_type", item)
                self.assertIn("occurrence_count", item)
                self.assertIn("clause_references", item)
                self.assertIn("review_status", item)

                # Ensure replacement label is neutral format [Type N]
                self.assertTrue(item["replacement_label"].startswith("["))
                self.assertTrue(item["replacement_label"].endswith("]"))

                # Strict requirement: NEVER display or return original PII values
                self.assertNotIn("Alice Walker", str(item))
                self.assertNotIn("alice@example.com", str(item))

            # Strict requirement: Entire response must contain zero raw PII values
            self.assertNotIn("Alice Walker", res.get_data(as_text=True))
            self.assertNotIn("alice@example.com", res.get_data(as_text=True))
        finally:
            d = db.session.get(Document, doc.id)
            if d:
                db.session.delete(d)
            db.session.commit()

if __name__ == "__main__":
    unittest.main()
