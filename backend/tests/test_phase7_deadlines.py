import os
import sys
import uuid
import datetime
from datetime import date
from pathlib import Path
import unittest

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.document import Document
from app.models.clause import Clause
from app.models.deadline import Deadline
from app.middleware.auth import get_jwt_secret
from app.utils.encryption import encrypt_text
from app.services.deadlines.deadline_extractor import (
    extract_deadlines_from_clause,
    extract_all_deadlines,
    extract_document_anchor_dates,
    calculate_deterministic_due_date,
    is_historical_or_example,
)
from app.reports.pdf_report import generate_pdf_report

class TestPhase7Deadlines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def setUp(self):
        self.client = self.app.test_client()

    # 1. Explicit calendar deadline
    def test_explicit_calendar_deadline(self):
        text = "Tenant must pay the initial security deposit no later than May 15, 2026."
        dl_list = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dl_list), 1)
        target = next(d for d in dl_list if d["parsed_date"] is not None)
        self.assertEqual(target["parsed_date"], date(2026, 5, 15))
        self.assertEqual(target["status"], "RESOLVED_EXPLICIT")
        self.assertEqual(target["responsible_party"], "Tenant")

    # 2. Relative deadline with a known trigger date
    def test_relative_deadline_with_known_trigger_date(self):
        clauses = [
            {"id": uuid.uuid4(), "redacted_text": "This Agreement is effective as of September 1, 2026."},
            {"id": uuid.uuid4(), "redacted_text": "Buyer shall deliver initial payment within 30 calendar days following the Effective Date."},
        ]
        doc_anchors = extract_document_anchor_dates(clauses)
        self.assertEqual(doc_anchors.get("effective_date"), date(2026, 9, 1))

        deadlines = extract_deadlines_from_clause(
            clauses[1]["redacted_text"],
            clause_id=clauses[1]["id"],
            doc_anchors=doc_anchors
        )
        self.assertGreaterEqual(len(deadlines), 1)
        rel_dl = deadlines[0]
        self.assertEqual(rel_dl["status"], "RESOLVED_CALCULATED")
        self.assertEqual(rel_dl["parsed_date"], date(2026, 10, 1))
        self.assertIn("2026-09-01", rel_dl["calculation_note"])
        self.assertEqual(rel_dl["responsible_party"], "Buyer")

    # 3. Relative deadline with an unknown trigger date
    def test_relative_deadline_with_unknown_trigger_date(self):
        text = "Tenant must provide written notice within 30 days after receipt of the renewal offer."
        deadlines = extract_deadlines_from_clause(text, clause_id=uuid.uuid4(), doc_anchors={})
        dl = deadlines[0]
        self.assertIsNone(dl["parsed_date"], "Must NOT calculate calendar date when trigger date is unknown")
        self.assertEqual(dl["status"], "UNRESOLVED_EVENT_BASED")
        self.assertIn("renewal offer", dl["trigger_event"].lower())
        self.assertEqual(dl["duration_count"], 30)

    # 4. Calendar days versus business days
    def test_calendar_days_versus_business_days(self):
        text_cal = "Company shall respond within 10 calendar days following the Effective Date."
        text_bus = "Company shall respond within 10 business days following the Effective Date."
        anchors = {"effective_date": date(2026, 10, 1)}

        dl_cal = extract_deadlines_from_clause(text_cal, clause_id=uuid.uuid4(), doc_anchors=anchors)[0]
        self.assertEqual(dl_cal["duration_unit"], "calendar_days")
        self.assertEqual(dl_cal["day_count_convention"], "calendar_days")
        self.assertEqual(dl_cal["status"], "RESOLVED_CALCULATED")
        self.assertEqual(dl_cal["parsed_date"], date(2026, 10, 11))

        dl_bus = extract_deadlines_from_clause(text_bus, clause_id=uuid.uuid4(), doc_anchors=anchors)[0]
        self.assertEqual(dl_bus["duration_unit"], "business_days")
        self.assertEqual(dl_bus["day_count_convention"], "business_days")
        # Must not silently treat business days as calendar days
        self.assertEqual(dl_bus["status"], "NEEDS_REVIEW")
        self.assertIsNone(dl_bus["parsed_date"])
        self.assertIn("business day", dl_bus["review_reason"].lower())

    # 5. Month-end and year-end calculations
    def test_month_end_and_year_end_calculations(self):
        # Jan 31 + 1 month in non-leap year 2026 -> Feb 28, 2026
        d1, note1 = calculate_deterministic_due_date(date(2026, 1, 31), 1, "months")
        self.assertEqual(d1, date(2026, 2, 28))

        # Nov 30, 2026 + 2 months -> Jan 30, 2027 (crosses year end)
        d2, note2 = calculate_deterministic_due_date(date(2026, 11, 30), 2, "months")
        self.assertEqual(d2, date(2027, 1, 30))

        # Jan 15, 2027 - 30 days -> Dec 16, 2026 (crosses year end backwards)
        d3, note3 = calculate_deterministic_due_date(date(2027, 1, 15), 30, "calendar_days", is_prior=True)
        self.assertEqual(d3, date(2026, 12, 16))

    # 6. Leap-year dates
    def test_leap_year_dates(self):
        # Leap year 2024: Jan 31 + 1 month -> Feb 29, 2024
        d_leap, _ = calculate_deterministic_due_date(date(2024, 1, 31), 1, "months")
        self.assertEqual(d_leap, date(2024, 2, 29))

        # Explicit leap date parsing
        text = "Option must be exercised by February 29, 2028."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertEqual(dls[0]["parsed_date"], date(2028, 2, 29))

    # 7. Renewal and opt-out windows
    def test_renewal_and_opt_out_windows(self):
        anchors = {"expiration_date": date(2027, 8, 31)}
        text = "Tenant must provide written notice to opt-out at least 60 calendar days prior to the Expiration Date."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4(), doc_anchors=anchors)
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertEqual(dl["status"], "RESOLVED_CALCULATED")
        # 2027-08-31 - 60 days = 2027-07-02
        self.assertEqual(dl["parsed_date"], date(2027, 7, 2))
        self.assertEqual(dl["responsible_party"], "Tenant")

    # 8. Automatic renewal conditions
    def test_automatic_renewal_conditions(self):
        text = "This lease shall automatically renew for successive terms of one (1) year unless Tenant provides written opt-out notice at least 60 days prior to the expiration date."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertEqual(dl["deadline_type"], "renewal_opt_out")
        self.assertIn("opt", dl["required_action"].lower())

    # 9. Termination notice and cure periods
    def test_termination_notice_and_cure_periods(self):
        text_term = "Either party may terminate this agreement upon 30 days written notice."
        dls_term = extract_deadlines_from_clause(text_term, clause_id=uuid.uuid4())
        self.assertEqual(dls_term[0]["deadline_type"], "termination_notice")
        self.assertEqual(dls_term[0]["duration_count"], 30)

        text_cure = "The defaulting party shall have 10 days from receipt of written notice to cure such breach."
        dls_cure = extract_deadlines_from_clause(text_cure, clause_id=uuid.uuid4())
        self.assertEqual(dls_cure[0]["deadline_type"], "cure_period")
        self.assertEqual(dls_cure[0]["duration_count"], 10)
        self.assertIn("receipt of written notice", dls_cure[0]["trigger_event"].lower())

    # 10. Recurring payment obligations
    def test_recurring_payment_obligations(self):
        text = "Tenant shall pay monthly rent in advance on or before the 1st day of each month."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertEqual(dl["deadline_type"], "payment_due")
        self.assertEqual(dl["recurrence"], "monthly on the 1st")
        self.assertEqual(dl["responsible_party"], "Tenant")

    # 11. Historical dates that are not deadlines
    def test_historical_dates_not_deadlines(self):
        text = "WHEREAS, on January 1, 2018, the parties entered into a prior consulting agreement."
        self.assertTrue(is_historical_or_example(text))
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertEqual(len(dls), 0, "Historical recital date must not become an actionable deadline")

    # 12. Example dates that must not become actionable deadlines
    def test_example_dates_not_deadlines(self):
        text = "For example, if notice is given on June 1, 2026, performance would be required accordingly."
        self.assertTrue(is_historical_or_example(text))
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertEqual(len(dls), 0, "Illustrative example date must not become an actionable deadline")

    # 13. Conditional obligations and exceptions
    def test_conditional_obligations_and_exceptions(self):
        text = "Employee shall return all company property within 5 days of termination, unless excused by force majeure."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertIsNotNone(dl["conditions"])
        self.assertIn("unless excused by force majeure", dl["conditions"])

    # 14. Multiple deadlines in one clause
    def test_multiple_deadlines_in_one_clause(self):
        text = (
            "Rent is payable on or before the 1st day of each month. "
            "Tenant shall have a 5-day grace period before a late fee is assessed. "
            "If default continues, Landlord may terminate this lease upon giving a 3-day notice to quit."
        )
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        types_found = {d["deadline_type"] for d in dls}
        self.assertIn("payment_due", types_found)
        self.assertIn("grace_period_end", types_found)
        self.assertTrue("termination_notice" in types_found or "notice_period" in types_found)
        self.assertGreaterEqual(len(dls), 3)

    # 15. Duplicate extraction without duplicate persisted records (Idempotence)
    def test_idempotent_deadline_storage(self):
        from app.models.user import User
        from app.models.document import Document
        from app.services.pipeline import process_document_pipeline

        # Create user & document
        test_user = User.query.filter_by(email="phase7_test@clauseguard.ai").first()
        if not test_user:
            test_user = User(name="Phase 7 Test User", email="phase7_test@clauseguard.ai", password_hash="dummy")
            db.session.add(test_user)
            db.session.commit()

        doc = Document(
            id=uuid.uuid4(),
            filename="idempotent_test.txt",
            original_format="txt",
            document_type="commercial_contract",
            status="pending",
            user_id=test_user.id,
            raw_text_encrypted=encrypt_text("Tenant shall pay rent on the 1st of each month and give 30 days notice to vacate."),
            redacted_text="Tenant shall pay rent on the 1st of each month and give 30 days notice to vacate.",
        )
        db.session.add(doc)
        db.session.commit()

        temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_idempotent.txt"))
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("Tenant shall pay rent on the 1st of each month and give 30 days notice to vacate.")

        try:
            # Run pipeline once
            process_document_pipeline(str(doc.id), temp_path, "txt")
            count_1 = Deadline.query.filter_by(document_id=doc.id).count()
            self.assertGreater(count_1, 0)

            # Re-create temp file for second run as pipeline safely deletes temp upload file in finally block
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write("Tenant shall pay rent on the 1st of each month and give 30 days notice to vacate.")

            # Re-run pipeline on the same document
            process_document_pipeline(str(doc.id), temp_path, "txt")
            count_2 = Deadline.query.filter_by(document_id=doc.id).count()
            self.assertEqual(count_1, count_2, "Re-analysis must be strictly idempotent and not create duplicate deadlines")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Clean up
            db.session.delete(doc)
            db.session.commit()

    # 16. Dates repeated in different substantive clauses
    def test_dates_repeated_in_different_substantive_clauses(self):
        c1_id = uuid.uuid4()
        c2_id = uuid.uuid4()
        clauses = [
            {"id": c1_id, "redacted_text": "Tenant shall give 30 days written notice to terminate this agreement."},
            {"id": c2_id, "redacted_text": "Landlord shall give 30 days written notice to inspect the premises."},
        ]
        all_dls = extract_all_deadlines(clauses)
        self.assertEqual(len(all_dls), 2, "Both clauses must retain their deadlines since they represent distinct legal obligations")

    # 17. Missing or ambiguous triggering events
    def test_missing_or_ambiguous_triggering_events(self):
        text = "Written confirmation must be submitted within 14 days."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertEqual(dl["status"], "UNRESOLVED_EVENT_BASED")
        self.assertIsNone(dl["parsed_date"])

    # 18. Conflicting duration numbers flags needs_review
    def test_conflicting_duration_numbers_flags_needs_review(self):
        text = "The borrower shall have thirty (60) days to cure the delinquency."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertEqual(dl["status"], "NEEDS_REVIEW")
        self.assertIn("Conflicting duration numbers", dl["review_reason"])

    # 19. Provenance: Clause IDs, page numbers, and offsets
    def test_provenance_clause_ids_pages_and_offsets(self):
        c_id = uuid.uuid4()
        meta = {"page_number": 3, "start_offset": 120, "end_offset": 240}
        text = "Tenant shall pay monthly rent in advance on or before the 1st day of each month."
        dls = extract_deadlines_from_clause(text, clause_id=c_id, clause_meta=meta)
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertEqual(dl["clause_id"], c_id)
        self.assertEqual(dl["page_number"], 3)
        self.assertEqual(dl["start_offset"], 120)

    # 20. PII pseudonymization preserved in output
    def test_pii_pseudonymization_preserved(self):
        text = "[Person 1] shall provide written notice to [Person 2] within 30 days of receipt."
        dls = extract_deadlines_from_clause(text, clause_id=uuid.uuid4())
        self.assertGreaterEqual(len(dls), 1)
        dl = dls[0]
        self.assertIn("[Person 1]", dl["raw_text"])
        self.assertEqual(dl["responsible_party"], "[Person 1]")

    # 21. Tenant isolation on deadlines endpoint
    def test_tenant_isolation_on_deadlines_endpoint(self):
        import jwt
        jwt_secret = get_jwt_secret()

        user_a = User.query.filter_by(email="tenant_a_p7@clauseguard.ai").first()
        if not user_a:
            user_a = User(name="User A", email="tenant_a_p7@clauseguard.ai", password_hash="dummy")
            db.session.add(user_a)
        user_b = User.query.filter_by(email="tenant_b_p7@clauseguard.ai").first()
        if not user_b:
            user_b = User(name="User B", email="tenant_b_p7@clauseguard.ai", password_hash="dummy")
            db.session.add(user_b)
        db.session.commit()

        doc_a = Document(
            id=uuid.uuid4(),
            filename="user_a_doc.txt",
            original_format="txt",
            document_type="lease",
            status="analyzed",
            user_id=user_a.id,
            raw_text_encrypted=encrypt_text("Sample"),
            redacted_text="Sample",
        )
        db.session.add(doc_a)
        db.session.commit()

        # Token for User B
        token_b = jwt.encode(
            {"userId": str(user_b.id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
            jwt_secret,
            algorithm="HS256",
        )

        resp = self.client.get(
            f"/api/documents/{doc_a.id}/deadlines",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        self.assertEqual(resp.status_code, 403, "User B must not access User A deadlines")

        # Clean up
        db.session.delete(doc_a)
        db.session.commit()

    # 22. Empty, malformed, and unsupported inputs
    def test_empty_and_malformed_inputs(self):
        self.assertEqual(extract_deadlines_from_clause(""), [])
        self.assertEqual(extract_deadlines_from_clause(None), [])
        self.assertEqual(extract_deadlines_from_clause("   "), [])
        self.assertEqual(extract_all_deadlines([]), [])

    # 23. PDF report generation with enriched deadlines
    def test_pdf_report_with_enriched_deadlines(self):
        mock_doc = Document(
            id=uuid.uuid4(),
            filename="sample_contract.pdf",
            document_type="residential_lease",
            overall_risk_score=25.0,
            risk_band="Low",
            uploaded_at=datetime.datetime.now(datetime.timezone.utc),
        )
        mock_dl = Deadline(
            document_id=mock_doc.id,
            deadline_type="payment_due",
            raw_text="Tenant shall pay rent on the 1st of each month.",
            obligation_title="Rent Payment",
            responsible_party="Tenant",
            required_action="pay rent",
            trigger_event="first day of each month",
            parsed_date=date(2026, 6, 1),
            duration_count=1,
            duration_unit="months",
            status="RESOLVED_EXPLICIT",
            confidence="confident",
            page_number=1,
        )
        mock_clause = Clause(
            document_id=mock_doc.id,
            clause_index=0,
            clause_type="rent_payment",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=15.0,
            favorability_label="fair",
            redacted_text="Tenant shall pay rent on the 1st of each month.",
            page_number=1,
        )

        pdf_bytes = generate_pdf_report(
            document=mock_doc,
            clauses=[mock_clause],
            missing_clauses=[],
            deadlines=[mock_dl],
            pii_findings=[],
            integrity_findings=[]
        )
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Must produce a valid PDF byte stream")

if __name__ == "__main__":
    unittest.main()
