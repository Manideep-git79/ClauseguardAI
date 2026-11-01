import os
import sys
import uuid
import datetime
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.document import Document
from app.models.clause import Clause
from app.models.deadline import Deadline
from app.models.missing_clause import MissingClause
from app.models.chat import ChatSession, ChatMessage
from app.middleware.auth import get_jwt_secret
from app.utils.encryption import encrypt_text
from app.services.rag.intent import classify_query_intent, normalize_query_text
from app.services.rag.retriever import retrieve_relevant_clauses
from app.services.rag.chat_engine import answer_document_question, deterministic_document_qa
import jwt

class TestRAGSemanticIntent(unittest.TestCase):
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

        # Patch live LLM and embedding calls for deterministic offline tests
        self.genai_patcher = patch("app.services.rag.chat_engine.get_genai_client", return_value=None)
        self.genai_patcher.start()
        self.embed_client_patcher = patch("app.services.rag.embedder.get_genai_client", return_value=None)
        self.embed_client_patcher.start()
        self.embed_patcher = patch("app.services.rag.chat_engine.embed_query", return_value=[0.0] * 768)
        self.embed_patcher.start()

        # Test user
        self.user_a = User.query.filter_by(email="tenant_rag_a@clauseguard.ai").first()
        if not self.user_a:
            self.user_a = User(name="User RAG A", email="tenant_rag_a@clauseguard.ai", password_hash="dummy")
            db.session.add(self.user_a)

        self.user_b = User.query.filter_by(email="tenant_rag_b@clauseguard.ai").first()
        if not self.user_b:
            self.user_b = User(name="User RAG B", email="tenant_rag_b@clauseguard.ai", password_hash="dummy")
            db.session.add(self.user_b)
        db.session.commit()

        self.token_a = jwt.encode(
            {"userId": str(self.user_a.id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
            self.jwt_secret,
            algorithm="HS256",
        )
        self.token_b = jwt.encode(
            {"userId": str(self.user_b.id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
            self.jwt_secret,
            algorithm="HS256",
        )

        # 1. Document with multiple high-risk and unfavorable clauses
        self.doc_risky = Document(
            id=uuid.uuid4(),
            filename="commercial_lease_risky.txt",
            original_format="txt",
            document_type="commercial_contract",
            status="analyzed",
            user_id=self.user_a.id,
            overall_risk_score=68.5,
            risk_band="High",
            redacted_text="Commercial Lease between [Landlord] and [Tenant].",
        )
        db.session.add(self.doc_risky)
        db.session.commit()

        # Clause 0: Preamble
        self.c_preamble = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_risky.id,
            clause_index=0,
            clause_type="preamble",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=5.0,
            favorability_label="neutral",
            redacted_text="This Commercial Lease Agreement is entered into between Acme Properties and Beta Corp.",
            page_number=1,
        )
        # Clause 1: Rent Payment
        self.c_rent = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_risky.id,
            clause_index=1,
            clause_type="rent_payment",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=15.0,
            favorability_label="fair",
            redacted_text="Tenant shall pay monthly rent of $5,000 in advance on the first day of each calendar month.",
            page_number=1,
        )
        # Clause 2: Unfavorable Indemnity
        self.c_indemnity = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_risky.id,
            clause_index=2,
            clause_type="indemnity",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=80.0,
            favorability_label="unfavorable",
            redacted_text="Tenant shall unconditionally indemnify, defend, and hold harmless Landlord against any and all claims, liabilities, damages, and attorney fees arising from the premises, even if caused by Landlord's sole negligence.",
            page_number=2,
        )
        # Clause 3: Unfavorable Termination (using different terminology: "discontinue appointment / lease")
        self.c_termination = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_risky.id,
            clause_index=3,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=65.0,
            favorability_label="unfavorable",
            redacted_text="The company reserves the absolute right to discontinue the appointment and terminate this agreement immediately without prior notice or cure period if performance is unsatisfactory.",
            page_number=3,
        )
        # Clause 4: Early departure penalty / Liquidated damages
        self.c_penalty = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_risky.id,
            clause_index=4,
            clause_type="liquidated_damages",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=70.0,
            favorability_label="unfavorable",
            redacted_text="If Tenant vacates before the lease expiry, Tenant must pay an early termination penalty fee of $15,000 and forfeit all prepaid rent.",
            page_number=4,
        )
        # Clause 5: Governing Law
        self.c_gov = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_risky.id,
            clause_index=5,
            clause_type="governing_law",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=5.0,
            favorability_label="fair",
            redacted_text="This agreement is governed by the laws of New York State.",
            page_number=5,
        )
        db.session.add_all([self.c_preamble, self.c_rent, self.c_indemnity, self.c_termination, self.c_penalty, self.c_gov])

        # Missing clause for risky doc
        self.missing_protection = MissingClause(
            document_id=self.doc_risky.id,
            clause_type="dispute_resolution",
            severity="high",
            checklist_note="Standard arbitration or mediation clause is missing.",
        )
        db.session.add(self.missing_protection)

        # 2. Document with NO unfavorable clauses (Clean/Fair contract)
        self.doc_clean = Document(
            id=uuid.uuid4(),
            filename="consulting_agreement_clean.txt",
            original_format="txt",
            document_type="commercial_contract",
            status="analyzed",
            user_id=self.user_a.id,
            overall_risk_score=12.0,
            risk_band="Low",
            redacted_text="Consulting Agreement between [Client] and [Consultant].",
        )
        db.session.add(self.doc_clean)
        db.session.commit()

        self.c_clean_services = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_clean.id,
            clause_index=0,
            clause_type="services",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=10.0,
            favorability_label="fair",
            redacted_text="Consultant shall perform advisory software architecture services with professional care.",
            page_number=1,
        )
        self.c_clean_term = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_clean.id,
            clause_index=1,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=15.0,
            favorability_label="fair",
            redacted_text="Either party may terminate this agreement upon giving 30 calendar days mutual written notice.",
            page_number=1,
        )
        self.c_clean_gov = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_clean.id,
            clause_index=2,
            clause_type="governing_law",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=5.0,
            favorability_label="neutral",
            redacted_text="This agreement is governed by the laws of Delaware.",
            page_number=2,
        )
        db.session.add_all([self.c_clean_services, self.c_clean_term, self.c_clean_gov])

        # 3. Document for User B (for tenant isolation tests)
        self.doc_user_b = Document(
            id=uuid.uuid4(),
            filename="user_b_private.txt",
            original_format="txt",
            document_type="employment_contract",
            status="analyzed",
            user_id=self.user_b.id,
            overall_risk_score=20.0,
            risk_band="Low",
        )
        db.session.add(self.doc_user_b)
        db.session.commit()

        self.c_user_b = Clause(
            id=uuid.uuid4(),
            document_id=self.doc_user_b.id,
            clause_index=0,
            clause_type="confidential_severance",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=20.0,
            favorability_label="fair",
            redacted_text="Private severance payout of $100,000 for User B only.",
            page_number=1,
        )
        db.session.add(self.c_user_b)
        db.session.commit()

    def tearDown(self):
        self.genai_patcher.stop()
        self.embed_client_patcher.stop()
        self.embed_patcher.stop()
        try:
            db.session.rollback()
            ChatMessage.query.delete()
            ChatSession.query.delete()
            Deadline.query.delete()
            MissingClause.query.delete()
            Clause.query.delete()
            Document.query.delete()
            db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            db.session.remove()

    # 1. Short risk queries trigger document risk summary, NOT "information not found"
    def test_short_risk_queries_trigger_risk_summary(self):
        queries = [
            "any risks",
            "what are the risks",
            "are there risks",
            "is there anything dangerous?",
            "what are the risky terms?",
            "any problematic conditions?",
            "what should I worry about?",
        ]
        for q in queries:
            res = answer_document_question(self.doc_risky.id, q)
            self.assertTrue(res["grounded"], f"Query '{q}' should be grounded, got: {res['answer']}")
            self.assertNotIn("does not contain information", res["answer"].lower())
            self.assertIn("68.5", res["answer"])
            self.assertIn("High", res["answer"])
            self.assertTrue(len(res["cited_clause_ids"]) > 0)
            self.assertTrue(len(res["sources"]) > 0)

    # 2. British vs American spelling variations
    def test_british_and_american_spelling_variations(self):
        # British: "unfavourable", American: "unfavorable"
        res_uk = answer_document_question(self.doc_risky.id, "any unfavourable clauses detected")
        res_us = answer_document_question(self.doc_risky.id, "any unfavorable clauses detected")

        self.assertTrue(res_uk["grounded"])
        self.assertTrue(res_us["grounded"])
        self.assertNotIn("does not contain information", res_uk["answer"].lower())
        self.assertNotIn("does not contain information", res_us["answer"].lower())

        # Both should cite the unfavorable clauses (Clause 2 and/or Clause 3, 4)
        self.assertIn(str(self.c_indemnity.id), res_uk["cited_clause_ids"])
        self.assertIn(str(self.c_indemnity.id), res_us["cited_clause_ids"])

    # 3. Semantically equivalent paraphrases (Group A, Group B, Group C)
    def test_semantic_paraphrases_groups(self):
        # Group A: Risk inquiries
        group_a = ["any risks", "what are the risks", "any dangerous clauses", "what should I worry about"]
        for q in group_a:
            res = answer_document_question(self.doc_risky.id, q)
            self.assertTrue(res["grounded"])
            self.assertTrue(any(str(c.id) in res["cited_clause_ids"] for c in [self.c_indemnity, self.c_penalty, self.c_termination]))

        # Group B: Unfavorable inquiries
        group_b = ["any unfavourable clauses", "unfavorable terms", "unfair conditions", "clauses against me"]
        for q in group_b:
            res = answer_document_question(self.doc_risky.id, q)
            self.assertTrue(res["grounded"])
            self.assertIn("unfavorable", res["answer"].lower())
            self.assertIn(str(self.c_indemnity.id), res["cited_clause_ids"])

        # Group C: Resignation, early departure, penalties
        res_depart = answer_document_question(self.doc_risky.id, "do I need to pay if I leave")
        self.assertTrue(res_depart["grounded"])
        self.assertIn(str(self.c_penalty.id), res_depart["cited_clause_ids"])
        self.assertIn("15,000", res_depart["answer"])

    # 4. Questions with differing terminology ("can they fire me" matching "discontinue appointment")
    def test_differing_terminology_termination(self):
        res = answer_document_question(self.doc_risky.id, "can they fire me")
        self.assertTrue(res["grounded"])
        self.assertIn(str(self.c_termination.id), res["cited_clause_ids"])
        self.assertIn("discontinue the appointment", res["answer"].lower())

    # 5. Document with NO unfavorable clauses reports cleanly without "information not found"
    def test_document_with_no_unfavorable_clauses(self):
        res = answer_document_question(self.doc_clean.id, "any unfavourable clauses?")
        self.assertTrue(res["grounded"])
        self.assertNotIn("does not contain information", res["answer"].lower())
        self.assertIn("did not identify any clauses classified as unfavorable", res["answer"].lower())
        self.assertIn("12.0", res["answer"])
        self.assertIn("Low", res["answer"])
        # Must not claim 100% legal immunity
        self.assertIn("does not", res["answer"].lower())

    # 6. Document with multiple unfavorable clauses lists and cites them
    def test_document_with_multiple_unfavorable_clauses(self):
        res = answer_document_question(self.doc_risky.id, "unfavourable clauses")
        self.assertTrue(res["grounded"])
        self.assertIn("[Clause 2, Page 2]", res["answer"])
        self.assertIn("[Clause 3, Page 3]", res["answer"])
        self.assertIn("[Clause 4, Page 4]", res["answer"])
        self.assertIn("Indemnity", res["answer"])
        self.assertIn("Liquidated Damages", res["answer"])

    # 7. Unsupported question receives grounded=False and NO arbitrary 80% confidence
    def test_unsupported_question_no_false_80_percent_confidence(self):
        res = answer_document_question(self.doc_risky.id, "What is the pet policy for dogs and cats?")
        self.assertFalse(res["grounded"])
        self.assertIn("does not contain information", res["answer"].lower())
        self.assertEqual(len(res["cited_clause_ids"]), 0)
        self.assertEqual(len(res["sources"]), 0)
        # CRITICAL: confidence must NOT be hardcoded 0.80
        self.assertNotEqual(res["confidence"], 0.80)
        self.assertIsNone(res["confidence"])

    # 8. Cross-document and cross-tenant isolation
    def test_cross_tenant_and_cross_document_isolation(self):
        # Searching doc_risky for User B private severance
        res = answer_document_question(self.doc_risky.id, "What is the private severance payout?")
        self.assertFalse(res["grounded"])
        self.assertNotIn("100,000", res["answer"])
        self.assertNotIn(str(self.c_user_b.id), res["cited_clause_ids"])

        # User B trying to query User A's document via API
        resp = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {self.token_b}"},
            json={"question": "any risks", "documentId": str(self.doc_risky.id)}
        )
        self.assertEqual(resp.status_code, 403)

    # 9. Specific clause lookup ("clause 2")
    def test_specific_clause_lookup(self):
        res = answer_document_question(self.doc_risky.id, "show me clause 2")
        self.assertTrue(res["grounded"])
        self.assertIn(str(self.c_indemnity.id), res["cited_clause_ids"])
        self.assertIn("[Clause 2, Page 2]", res["answer"])

    # 10. Missing or zero vector embeddings fallback gracefully to hybrid/BM25
    def test_zero_vector_fallback_to_bm25_and_structured(self):
        # Even with zero vector embeddings, BM25 and structured retrieval must succeed
        clauses = retrieve_relevant_clauses(
            self.doc_risky.id,
            query_vector=[0.0] * 768,
            query_text="rent payment advance",
            top_k=2
        )
        self.assertTrue(len(clauses) > 0)
        self.assertEqual(clauses[0]["clause_index"], 1)
        self.assertEqual(clauses[0]["clause_type"], "rent_payment")

    # 11. End-to-end API compatibility with frontend contract
    def test_api_chat_endpoint_contract(self):
        resp = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={"question": "any risks", "documentId": str(self.doc_risky.id)}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("id", data)
        self.assertIn("answer", data)
        self.assertTrue(data["grounded"])
        self.assertTrue(isinstance(data["confidence"], float))
        self.assertTrue(data["confidence"] > 0.60)
        self.assertTrue(len(data["sources"]) > 0)
        self.assertIn("section", data["sources"][0])
        self.assertIn("page_number", data["sources"][0])

if __name__ == "__main__":
    unittest.main()
