import os
import sys
import uuid
import datetime
from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import jwt

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.document import Document
from app.models.clause import Clause
from app.models.deadline import Deadline
from app.models.chat import ChatSession, ChatMessage
from app.middleware.auth import get_jwt_secret
from app.utils.encryption import encrypt_text
from app.services.rag.chat_engine import (
    answer_document_question,
    deterministic_document_qa,
    SYSTEM_INSTRUCTION,
)
from app.services.rag.intent import classify_query_intent

class TestRAGPhase3GroundedGeneration(unittest.TestCase):
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
        self.jwt_secret = get_jwt_secret()

        # Controlled test double: deterministic legal QA by default
        self.genai_patcher = patch("app.services.rag.chat_engine.get_genai_client", return_value=None)
        self.genai_patcher.start()
        self.embed_client_patcher = patch("app.services.rag.embedder.get_genai_client", return_value=None)
        self.embed_client_patcher.start()
        self.embed_patcher = patch("app.services.rag.chat_engine.embed_query", return_value=[0.0] * 768)
        self.embed_patcher.start()

        # Test user
        self.user = User.query.filter_by(email="phase3_tester@clauseguard.ai").first()
        if not self.user:
            self.user = User(name="Phase 3 Tester", email="phase3_tester@clauseguard.ai", password_hash="dummy")
            db.session.add(self.user)
            db.session.commit()

        self.token = jwt.encode(
            {"userId": str(self.user.id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
            self.jwt_secret,
            algorithm="HS256",
        )

        # Create test document
        self.doc = Document(
            id=uuid.uuid4(),
            filename="phase3_test_contract.txt",
            original_format="txt",
            document_type="commercial_contract",
            status="analyzed",
            user_id=self.user.id,
            raw_text_encrypted=encrypt_text("Sample contract text"),
            redacted_text="Commercial Agreement between Company and Provider.",
            overall_risk_score=25.0,
            risk_band="Low",
        )
        db.session.add(self.doc)
        db.session.commit()

        # Clause 0: Payment Terms
        self.clause_rent = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=0,
            clause_type="rent_payment",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=10.0,
            favorability_label="fair",
            redacted_text="Tenant shall pay monthly rent in advance on or before the 1st day of each month.",
            page_number=1,
        )

        # Clause 1: Governing Law
        self.clause_gov = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=1,
            clause_type="governing_law",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=5.0,
            favorability_label="fair",
            redacted_text="This agreement shall be governed by and construed in accordance with the laws of California.",
            page_number=2,
        )

        # Clause 2: Termination Notice with condition and exception
        self.clause_term = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=2,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=20.0,
            favorability_label="fair",
            redacted_text="Either party may terminate this agreement upon giving 30 calendar days written notice, unless excused by force majeure.",
            page_number=3,
        )

        # Clause 3: Employer Dismissal / Involuntary Termination
        self.clause_dismissal = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=3,
            clause_type="termination",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=35.0,
            favorability_label="unfavorable",
            redacted_text="Company may discharge or dismiss Employee for gross misconduct immediately without prior notice.",
            page_number=3,
        )

        # Clause 4: Severance
        self.clause_severance = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=4,
            clause_type="severance",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=15.0,
            favorability_label="fair",
            redacted_text="In the event of redundancy, Employee shall receive severance compensation of $25,000.",
            page_number=4,
        )

        db.session.add_all([
            self.clause_rent,
            self.clause_gov,
            self.clause_term,
            self.clause_dismissal,
            self.clause_severance,
        ])

        # Relative deadline
        self.dl_relative = Deadline(
            document_id=self.doc.id,
            clause_id=self.clause_term.id,
            deadline_type="notice_period",
            raw_text="Tenant must provide written notice within 30 days after receipt of renewal notice.",
            obligation_title="Renewal Opt-Out Notice",
            responsible_party="Tenant",
            required_action="provide written notice",
            trigger_event="receipt of renewal notice",
            parsed_date=None,
            duration_count=30,
            duration_unit="calendar_days",
            status="UNRESOLVED_EVENT_BASED",
            page_number=3,
        )
        db.session.add(self.dl_relative)
        db.session.commit()

    def tearDown(self):
        self.genai_patcher.stop()
        self.embed_client_patcher.stop()
        self.embed_patcher.stop()
        try:
            db.session.rollback()
            db.session.query(ChatMessage).delete()
            db.session.query(ChatSession).delete()
            db.session.query(Deadline).delete()
            db.session.query(Clause).delete()
            db.session.query(Document).delete()
            db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            db.session.remove()

    # 1. Direct First-Sentence Answer & Plain-English Explanation (Governing Law)
    def test_direct_first_sentence_governing_law(self):
        res = answer_document_question(self.doc.id, "What is the governing law of this agreement?")
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        answer = res["answer"]

        # First sentence must answer directly with the jurisdiction
        first_sentence = answer.split("\n\n")[1] if "###" in answer else answer.split(".")[0]
        self.assertIn("California", first_sentence)
        self.assertIn("The governing law of this agreement is", answer)
        self.assertIn("In plain English", answer)
        self.assertIn("[Clause 1, Page 2]", answer)
        self.assertIn(str(self.clause_gov.id), res["cited_clause_ids"])

    # 2. Direct First-Sentence Answer & Timing Explanation (Payment)
    def test_direct_first_sentence_payment(self):
        res = answer_document_question(self.doc.id, "When does rent need to be paid?")
        self.assertTrue(res["grounded"])
        answer = res["answer"]

        self.assertIn("on or before the 1st day of each month", answer)
        self.assertIn("In plain English", answer)
        self.assertIn("[Clause 0, Page 1]", answer)
        self.assertIn(str(self.clause_rent.id), res["cited_clause_ids"])

    # 3. Preservation of Negation and Conditions (Termination & Force Majeure)
    def test_preservation_of_conditions_and_exceptions(self):
        res = answer_document_question(self.doc.id, "What are the termination rules?")
        self.assertTrue(res["grounded"])
        answer = res["answer"]

        # Notice period must be present
        self.assertIn("30 calendar days", answer)
        # Condition/exception must be preserved
        self.assertIn("force majeure", answer.lower())
        self.assertIn("Conditions & Exceptions", answer)
        self.assertIn("[Clause 2, Page 3]", answer)

    # 4. Legal Action Distinction: Involuntary Dismissal vs General Termination
    def test_distinct_legal_action_dismissal(self):
        res = answer_document_question(self.doc.id, "Can they fire or dismiss me without notice?")
        self.assertTrue(res["grounded"])
        answer = res["answer"]

        # Must address dismissal/discharge specifically
        self.assertTrue("dismissal" in answer.lower() or "discharge" in answer.lower() or "fire" in answer.lower())
        self.assertIn("gross misconduct", answer.lower())
        self.assertIn("without prior notice", answer.lower())
        self.assertIn(str(self.clause_dismissal.id), res["cited_clause_ids"])

    # 5. Supporting Quote is Not the Sole Answer (No raw blockquote dump)
    def test_quote_is_supporting_evidence_not_sole_answer(self):
        res = answer_document_question(self.doc.id, "What severance do I get upon redundancy?")
        self.assertTrue(res["grounded"])
        answer = res["answer"]

        # Must have plain English explanation before the blockquote
        self.assertIn("In plain English", answer)
        self.assertIn("25,000", answer)
        self.assertIn("#### Supporting Contractual Language:", answer)
        self.assertIn("> \"In the event of redundancy", answer)

    # 6. Fallback Transparency & Notice
    def test_fallback_transparency_and_notice(self):
        res = answer_document_question(self.doc.id, "What is the governing law?")
        self.assertTrue(res["fallback_used"])
        self.assertIn("Generated via deterministic contract analysis due to automated service timeout", res["answer"])

    # 7. Unsupported Question Honest Abstention
    def test_honest_abstention_unsupported_question(self):
        res = answer_document_question(self.doc.id, "What is the policy regarding pets like cats and dogs?")
        self.assertFalse(res["grounded"])
        self.assertIsNone(res["confidence"])
        self.assertEqual(len(res["cited_clause_ids"]), 0)
        self.assertEqual(len(res["sources"]), 0)
        self.assertIn("The provided document does not contain information addressing this question.", res["answer"])

    # 8. Relative Event-Based Deadline Remains Grounded
    def test_relative_deadline_remains_grounded(self):
        res = answer_document_question(self.doc.id, "When must the renewal opt-out notice be given?")
        # Must be grounded because evidence exists in the contract
        self.assertTrue(res["grounded"], "Event-based relative deadline with evidence must be grounded")
        self.assertIn("receipt of renewal notice", res["answer"].lower())
        self.assertIn("cannot be determined from the contract alone", res["answer"].lower())
        self.assertTrue(len(res["cited_clause_ids"]) > 0)

    # 9. Fabricated Citation Prevention
    def test_fabricated_citation_prevention(self):
        with patch("app.services.rag.chat_engine.get_genai_client") as mock_client:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            # Model hallucinated Clause 999 which does not exist
            mock_resp.text = "The contract governing law is California pursuant to [Clause 1, Page 2] and [Clause 999, Page 99]."
            mock_inst.models.generate_content.return_value = mock_resp
            mock_client.return_value = mock_inst

            res = answer_document_question(self.doc.id, "What is the governing law?")
            self.assertTrue(res["grounded"])
            self.assertFalse(res["fallback_used"])
            # Clause 1 should be cited, but fabricated Clause 999 must NOT be in cited_clause_ids
            self.assertIn(str(self.clause_gov.id), res["cited_clause_ids"])
            self.assertEqual(len(res["cited_clause_ids"]), 1)

    # 10. Live Gemini Generation (when API key is present) or Mocked Instruction Validation
    def test_gemini_prompt_contains_query_constraints_and_deduplication(self):
        with patch("app.services.rag.chat_engine.get_genai_client") as mock_client:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = "You may terminate the agreement upon 30 calendar days written notice pursuant to [Clause 2, Page 3]."
            mock_inst.models.generate_content.return_value = mock_resp
            mock_client.return_value = mock_inst

            # Ask question with negation: "Can I terminate without cause?"
            res = answer_document_question(self.doc.id, "Can I terminate without cause?")
            self.assertTrue(res["grounded"])
            self.assertFalse(res["fallback_used"])

            # Verify prompt sent to Gemini had QUERY_CONSTRAINTS
            call_args = mock_inst.models.generate_content.call_args
            prompt_sent = call_args.kwargs.get("contents") or call_args[1].get("contents")
            self.assertIn("<QUERY_CONSTRAINTS>", prompt_sent)
            self.assertIn("NEGATION / QUALIFICATIONS", prompt_sent)
            self.assertIn("DIRECT PLAIN-ENGLISH ANSWER FIRST", prompt_sent)

    # 11. Regression: Severance query must not be hijacked by generic monthly payment
    def test_severance_not_hijacked_by_monthly_payment(self):
        res = answer_document_question(self.doc.id, "What severance compensation am I entitled to upon redundancy?")
        self.assertTrue(res["grounded"])
        answer = res["answer"]
        self.assertIn("severance benefits of $25,000", answer)
        self.assertIn("[Clause 4, Page 4]", answer)
        self.assertIn(str(self.clause_severance.id), res["cited_clause_ids"])
        # Must NOT cite rent/payment clause
        self.assertNotIn(str(self.clause_rent.id), res["cited_clause_ids"])

    # 12. Regression: Penalty query must cite penalty/liquidated damages rather than ongoing rent
    def test_penalty_not_hijacked_by_generic_payment(self):
        penalty_clause = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=5,
            clause_type="liquidated_damages",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=70.0,
            favorability_label="unfavorable",
            redacted_text="In the event of unexcused early departure, Employee shall pay an early termination penalty fee of $15,000.",
            page_number=5,
        )
        db.session.add(penalty_clause)
        db.session.commit()

        res = answer_document_question(self.doc.id, "Do I need to pay a penalty if I leave early?")
        self.assertTrue(res["grounded"])
        answer = res["answer"]
        self.assertIn("penalty/liquidated damages obligation of $15,000", answer)
        self.assertIn("[Clause 5, Page 5]", answer)
        self.assertIn(str(penalty_clause.id), res["cited_clause_ids"])
        self.assertNotIn(str(self.clause_rent.id), res["cited_clause_ids"])

if __name__ == "__main__":
    unittest.main()

