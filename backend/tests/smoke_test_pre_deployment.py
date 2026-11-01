"""
DocuGuard AI: Pre-Deployment Smoke-Test & Verification Suite
Tests answer quality, terminology, conditions, payment deadlines, and source support across:
Group 1: Deterministic legal QA fallback (when Gemini is unavailable / timeout).
Group 2: Mocked Gemini generation (verifying citation validation, grounding, injection boundaries).
Group 3: Live Gemini integration (verified against gemini-3.5-flash-lite when API key is active).
"""

import os
import sys
import uuid
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.document import Document
from app.models.clause import Clause
from app.models.user import User
from app.models.chat import ChatSession, ChatMessage
from app.services.rag.chat_engine import answer_document_question
from app.services.rag.embedder import get_genai_client

class PreDeploymentSmokeTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def setUp(self):
        # Setup non-sensitive authorized test document (Commercial Lease)
        self.test_user = User.query.filter_by(email="smoke_tester_qa@clauseguard.ai").first()
        if not self.test_user:
            self.test_user = User(
                name="Smoke Tester QA",
                email="smoke_tester_qa@clauseguard.ai",
                password_hash="dummy_hash"
            )
            db.session.add(self.test_user)
            db.session.commit()

        self.doc_id = uuid.uuid4()
        self.doc = Document(
            id=self.doc_id,
            filename="smoke_test_commercial_lease.txt",
            original_format="txt",
            document_type="commercial_lease",
            status="analyzed",
            user_id=self.test_user.id,
            redacted_text="Commercial Lease between Commercial Landlord and Commercial Tenant.",
            overall_risk_score=35.0,
            risk_band="Medium",
        )
        db.session.add(self.doc)

        # Clause 0: Payment terms with explicit due date and amount
        self.c0 = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_id,
            clause_index=0,
            clause_type="rent_payment",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=10.0,
            favorability_label="fair",
            redacted_text="Tenant shall pay monthly base rent of $3,500 in advance on the 1st day of each calendar month.",
            page_number=1,
        )
        # Clause 1: Governing law
        self.c1 = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_id,
            clause_index=1,
            clause_type="governing_law",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=5.0,
            favorability_label="fair",
            redacted_text="This agreement shall be governed by and construed in accordance with the laws of New York.",
            page_number=1,
        )
        # Clause 2: Termination notice with 30 days and exception
        self.c2 = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_id,
            clause_index=2,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=20.0,
            favorability_label="fair",
            redacted_text="Either party may terminate this lease upon giving 30 calendar days written notice, unless excused by an act of God.",
            page_number=2,
        )
        # Clause 3: Conflicting / distinct termination notice (60 days)
        self.c3 = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_id,
            clause_index=3,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=30.0,
            favorability_label="needs_review",
            redacted_text="Landlord requires 60 calendar days written notice prior to terminating any commercial tenancy.",
            page_number=2,
        )
        # Clause 4: Landlord immediate termination for hazardous waste
        self.c4 = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_id,
            clause_index=4,
            clause_type="termination",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=60.0,
            favorability_label="unfavorable",
            redacted_text="Landlord may immediately terminate the tenancy without prior notice in the event of unauthorized hazardous waste disposal.",
            page_number=3,
        )

        db.session.add_all([self.c0, self.c1, self.c2, self.c3, self.c4])
        db.session.commit()

    def tearDown(self):
        try:
            db.session.rollback()
            ChatMessage.query.filter_by(document_id=self.doc_id).delete()
            ChatSession.query.filter_by(document_id=self.doc_id).delete()
            Clause.query.filter_by(document_id=self.doc_id).delete()
            Document.query.filter_by(id=self.doc_id).delete()
            db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            db.session.remove()


class TestGroup1DeterministicFallback(PreDeploymentSmokeTestBase):
    """
    Tests deterministic legal QA fallback when Gemini is unavailable.
    Asserts exact answer completeness, terminology, condition preservation,
    and source support across all 8 mandatory checklist scenarios.
    """

    def setUp(self):
        super().setUp()
        self.genai_patcher = patch("app.services.rag.chat_engine.get_genai_client", return_value=None)
        self.genai_patcher.start()
        self.embed_patcher = patch("app.services.rag.chat_engine.embed_query", return_value=[0.0] * 768)
        self.embed_patcher.start()

    def tearDown(self):
        self.genai_patcher.stop()
        self.embed_patcher.stop()
        super().tearDown()

    def test_item_1_direct_factual_governing_law(self):
        """Item 1: Direct factual question with valid citation"""
        res = answer_document_question(self.doc_id, "What is the governing law of this lease?", user_id=self.test_user.id)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        self.assertIn(str(self.c1.id), res["cited_clause_ids"])

        ans = res["answer"]
        self.assertIn("New York", ans)
        self.assertIn("[Clause 1, Page 1]", ans)
        self.assertIn("In plain English", ans)
        # Verify source support
        self.assertTrue("laws of New York" in self.c1.redacted_text)

    def test_item_2_paraphrased_landlord_eviction_tenancy_terms(self):
        """Item 2: Paraphrased question with semantically relevant evidence & tenancy terminology"""
        res = answer_document_question(self.doc_id, "Can the landlord evict or terminate me without advance notice?", user_id=self.test_user.id)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        self.assertIn(str(self.c4.id), res["cited_clause_ids"])

        ans = res["answer"]
        # Terminology assertion: Must use tenancy terms (eviction / tenancy termination), NOT employment "dismissal"
        self.assertNotIn("dismissal", ans.lower())
        self.assertTrue("evict" in ans.lower() or "tenancy" in ans.lower() or "landlord" in ans.lower())
        # Grammatical check: No duplicated prepositions
        self.assertNotIn("in cases of in the event of", ans.lower())
        # Exception preservation: Immediate termination allowed ONLY for unauthorized hazardous waste disposal
        self.assertIn("hazardous waste disposal", ans.lower())
        self.assertIn("[Clause 4, Page 3]", ans)

    def test_item_3_negation_and_contractual_exception(self):
        """Item 3: Question involving negation or a contractual exception"""
        res = answer_document_question(self.doc_id, "Can I terminate the lease without notice?", user_id=self.test_user.id)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        self.assertIn(str(self.c2.id), res["cited_clause_ids"])

        ans = res["answer"]
        # Negation: Must affirm termination without notice is NOT permitted
        self.assertTrue("not permitted" in ans.lower() or "requires advance" in ans.lower() or "mandates advance" in ans.lower())
        # Notice period preservation
        self.assertIn("30 calendar days", ans)
        # Contractual exception preservation
        self.assertIn("act of god", ans.lower())
        self.assertIn("[Clause 2, Page 2]", ans)

    def test_item_4_severance_honest_abstention(self):
        """Item 4: Severance question when no severance provision exists"""
        res = answer_document_question(self.doc_id, "What is the severance compensation payout?", user_id=self.test_user.id)
        self.assertFalse(res["grounded"])
        self.assertIsNone(res["confidence"])
        self.assertEqual(len(res["cited_clause_ids"]), 0)
        self.assertIn("The provided document does not contain information addressing this question.", res["answer"])

    def test_item_5_penalty_leaving_early_limitation(self):
        """Item 5: Penalty question when retrieved clause contains no penalty obligation"""
        res = answer_document_question(self.doc_id, "Do I have to pay a penalty fee if I leave early?", user_id=self.test_user.id)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        # Should cite the termination notice clause governing departure
        self.assertIn(str(self.c2.id), res["cited_clause_ids"])

        ans = res["answer"]
        # Quality assertion: Must state no penalty was established by retrieved provisions and explain limitation
        self.assertTrue("does not establish a financial penalty" in ans.lower() or "does not stipulate a financial penalty" in ans.lower())
        self.assertIn("no penalty fee is established", ans.lower())
        self.assertIn("limited to the retrieved sections", ans.lower())
        # Never fabricate a penalty fee
        self.assertNotIn("$", ans)

    def test_item_6_conflicting_notice_periods(self):
        """Item 6: Conflicting notice-period provisions"""
        res = answer_document_question(self.doc_id, "How many days notice are required to terminate the agreement?", user_id=self.test_user.id)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        # Must surface and cite BOTH clauses
        self.assertIn(str(self.c2.id), res["cited_clause_ids"])
        self.assertIn(str(self.c3.id), res["cited_clause_ids"])

        ans = res["answer"]
        self.assertIn("Conflicting Contract Provisions Identified", ans)
        self.assertIn("30 calendar days", ans)
        self.assertIn("60 calendar days", ans)
        self.assertIn("[Clause 2, Page 2]", ans)
        self.assertIn("[Clause 3, Page 2]", ans)

    def test_item_7_unsupported_question_abstention(self):
        """Item 7: Unsupported question that should abstain"""
        res = answer_document_question(self.doc_id, "What is the pet policy for dogs and cats?", user_id=self.test_user.id)
        self.assertFalse(res["grounded"])
        self.assertIsNone(res["confidence"])
        self.assertEqual(len(res["cited_clause_ids"]), 0)
        self.assertIn("The provided document does not contain information addressing this question.", res["answer"])

    def test_item_8_timeout_and_exact_payment_deadline_preservation(self):
        """Item 8: Gemini timeout triggering transparent fallback + exact payment deadline preservation"""
        res = answer_document_question(self.doc_id, "When does monthly base rent need to be paid?", user_id=self.test_user.id)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        self.assertIn(str(self.c0.id), res["cited_clause_ids"])

        ans = res["answer"]
        # Exact deadline preservation: Not just "monthly", must preserve full deadline
        self.assertIn("1st day of each calendar month", ans)
        self.assertIn("in advance", ans.lower())
        # Exact amount preservation
        self.assertIn("$3,500", ans)
        self.assertIn("[Clause 0, Page 1]", ans)
        # Transparent fallback notice
        self.assertIn("deterministic contract analysis due to automated service timeout", ans)


class TestGroup2MockedGeminiGeneration(PreDeploymentSmokeTestBase):
    """
    Tests Gemini generation with mocked client to verify prompt constraints,
    citation mapping, hallucinated citation removal, and ungrounded handling.
    """

    def setUp(self):
        super().setUp()
        self.embed_patcher = patch("app.services.rag.chat_engine.embed_query", return_value=[0.0] * 768)
        self.embed_patcher.start()

    def tearDown(self):
        self.embed_patcher.stop()
        super().tearDown()

    def test_mocked_gemini_valid_generation_and_citation_mapping(self):
        """Verify that Gemini answer text is parsed and citation IDs are properly mapped"""
        with patch("app.services.rag.chat_engine.get_genai_client") as mock_client:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = (
                "The governing law of this commercial lease is New York pursuant to [Clause 1, Page 1]. "
                "Any legal dispute must be resolved under New York jurisdiction."
            )
            mock_inst.models.generate_content.return_value = mock_resp
            mock_client.return_value = mock_inst

            res = answer_document_question(self.doc_id, "What is the governing law of this lease?", user_id=self.test_user.id)
            self.assertFalse(res["fallback_used"])
            self.assertTrue(res["grounded"])
            self.assertIn(str(self.c1.id), res["cited_clause_ids"])
            self.assertIn("New York", res["answer"])

    def test_mocked_gemini_hallucinated_citation_dropped(self):
        """Verify that hallucinated clause citations not in retrieved evidence are eliminated"""
        with patch("app.services.rag.chat_engine.get_genai_client") as mock_client:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            # Model hallucinated Clause 99 and Clause 42 which do not exist in retrieved clauses
            mock_resp.text = (
                "Under [Clause 1, Page 1], governing law is New York. "
                "Also see [Clause 99, Page 10] and [Clause 42] for additional details."
            )
            mock_inst.models.generate_content.return_value = mock_resp
            mock_client.return_value = mock_inst

            res = answer_document_question(self.doc_id, "What is the governing law?", user_id=self.test_user.id)
            self.assertFalse(res["fallback_used"])
            self.assertTrue(res["grounded"])
            # Only valid Clause 1 should be retained
            self.assertEqual(res["cited_clause_ids"], [str(self.c1.id)])

    def test_mocked_gemini_ungrounded_abstention_handling(self):
        """Verify that model responses stating unaddressed content become ungrounded with None confidence"""
        with patch("app.services.rag.chat_engine.get_genai_client") as mock_client:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = "The provided document does not contain information addressing this question."
            mock_inst.models.generate_content.return_value = mock_resp
            mock_client.return_value = mock_inst

            res = answer_document_question(self.doc_id, "What is the pet policy for dogs and cats?", user_id=self.test_user.id)
            self.assertFalse(res["fallback_used"])
            self.assertFalse(res["grounded"])
            self.assertIsNone(res["confidence"])
            self.assertEqual(len(res["cited_clause_ids"]), 0)

    def test_mocked_gemini_timeout_triggers_transparent_fallback(self):
        """Verify that Gemini API failure or timeout smoothly triggers deterministic fallback"""
        with patch("app.services.rag.chat_engine.get_genai_client") as mock_client:
            mock_inst = MagicMock()
            mock_inst.models.generate_content.side_effect = Exception("504 DEADLINE_EXCEEDED (Gateway Timeout)")
            mock_client.return_value = mock_inst

            res = answer_document_question(self.doc_id, "What is the governing law of this lease?", user_id=self.test_user.id)
            self.assertTrue(res["fallback_used"])
            self.assertTrue(res["grounded"])
            self.assertIn("New York", res["answer"])
            self.assertIn(str(self.c1.id), res["cited_clause_ids"])


class TestGroup3LiveGeminiIntegration(PreDeploymentSmokeTestBase):
    """
    Live integration test against Google GenAI API using verified gemini-3.5-flash-lite.
    """

    def setUp(self):
        super().setUp()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured in environment; skipping live integration tests.")

    def test_live_gemini_direct_factual_generation(self):
        """Test live Gemini grounded answer generation for direct factual query"""
        # Explicitly configure verified working model
        with patch.dict(os.environ, {"GEMINI_CHAT_MODEL": "gemini-3.5-flash-lite"}):
            res = answer_document_question(self.doc_id, "What is the governing law of this commercial lease?", user_id=self.test_user.id)
            print(f"\n[Live Gemini Direct Factual] Model: gemini-3.5-flash-lite")
            print(f"Fallback Used: {res.get('fallback_used')}")
            print(f"Grounded:      {res.get('grounded')}")
            print(f"Cited IDs:     {res.get('cited_clause_ids')}")
            print(f"Answer:        {res.get('answer')[:160]}...")

            self.assertFalse(res["fallback_used"], "Live Gemini call should succeed without fallback")
            self.assertTrue(res["grounded"], "Live Gemini response must be grounded")
            self.assertIn(str(self.c1.id), res["cited_clause_ids"], "Live Gemini must cite Clause 1")
            self.assertIn("new york", res["answer"].lower())

    def test_live_gemini_negation_and_exception(self):
        """Test live Gemini grounded generation preserving negation and exceptions"""
        with patch.dict(os.environ, {"GEMINI_CHAT_MODEL": "gemini-3.5-flash-lite"}):
            res = answer_document_question(self.doc_id, "Can the landlord evict or terminate me without advance notice?", user_id=self.test_user.id)
            print(f"\n[Live Gemini Negation & Exception] Model: gemini-3.5-flash-lite")
            print(f"Fallback Used: {res.get('fallback_used')}")
            print(f"Grounded:      {res.get('grounded')}")
            print(f"Cited IDs:     {res.get('cited_clause_ids')}")
            print(f"Answer:        {res.get('answer')[:160]}...")

            self.assertFalse(res["fallback_used"])
            self.assertTrue(res["grounded"])
            self.assertIn(str(self.c4.id), res["cited_clause_ids"])
            self.assertIn("hazardous waste", res["answer"].lower())

    def test_live_gemini_payment_deadline_preservation(self):
        """Test live Gemini exact payment deadline and amount preservation"""
        with patch.dict(os.environ, {"GEMINI_CHAT_MODEL": "gemini-3.5-flash-lite"}):
            res = answer_document_question(self.doc_id, "When does monthly base rent need to be paid and what is the amount?", user_id=self.test_user.id)
            print(f"\n[Live Gemini Payment Deadline] Model: gemini-3.5-flash-lite")
            print(f"Fallback Used: {res.get('fallback_used')}")
            print(f"Grounded:      {res.get('grounded')}")
            print(f"Cited IDs:     {res.get('cited_clause_ids')}")
            print(f"Answer:        {res.get('answer')[:160]}...")

            self.assertFalse(res["fallback_used"])
            self.assertTrue(res["grounded"])
            self.assertIn(str(self.c0.id), res["cited_clause_ids"])
            self.assertIn("1st", res["answer"].lower())
            self.assertIn("3,500", res["answer"])


def run_all_pre_deployment_smoke_tests():
    """Runs all 3 test groups and prints a unified summary report"""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestGroup1DeterministicFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestGroup2MockedGeminiGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestGroup3LiveGeminiIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

if __name__ == "__main__":
    result = run_all_pre_deployment_smoke_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
