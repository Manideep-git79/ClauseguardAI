import os
import sys
import math
import uuid
import datetime
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.document import Document
from app.models.clause import Clause
from app.services.rag.embedder import (
    embed_text,
    embed_query,
    batch_embed_texts,
    validate_embedding_vector,
    is_transient_error,
)
from app.services.rag.intent import (
    classify_query_intent,
    normalize_query_text,
    extract_negated_conditions,
)
from app.services.rag.retriever import (
    retrieve_relevant_clauses,
    compute_bm25_scores,
    extract_query_keywords,
)


class TestRAGPhase2ParaphraseEmbeddings(unittest.TestCase):
    """
    Comprehensive, deterministic test suite for Phase 2 RAG improvements:
    1. Embedding failure handling (timeout, rate-limit 429, invalid dimensions, NaN/Inf).
    2. Explicit retrieval fallback when vector search is unavailable.
    3. Paraphrase retrieval and query expansion for termination, payment, and price-escalation.
    4. Legally distinct actions preservation (resignation != dismissal != lease surrender != general termination).
    5. Negation preservation ("without cause", "without notice", "without penalty").
    6. Non-hallucination when no relevant clause exists (empty results rather than first five).
    7. Preservation of Phase 1 BM25 Robertson-Spärck Jones boilerplate ranking.
    8. Real database pgvector integration test (verifying Vector(768) and cosine distance).
    """

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def setUp(self):
        db.session.rollback()
        # Create test user for database tests
        self.user = User.query.filter_by(email="phase2_rag_test@clauseguard.ai").first()
        if not self.user:
            self.user = User(
                name="Phase 2 Test User",
                email="phase2_rag_test@clauseguard.ai",
                password_hash="dummy_hash",
            )
            db.session.add(self.user)
            db.session.commit()
        self.test_docs = []

    def tearDown(self):
        try:
            for doc_id in self.test_docs:
                Clause.query.filter_by(document_id=doc_id).delete(synchronize_session=False)
                Document.query.filter_by(id=doc_id).delete(synchronize_session=False)
            db.session.commit()
        except Exception:
            db.session.rollback()

    def _create_test_document(self, filename: str) -> Document:
        doc = Document(
            id=uuid.uuid4(),
            user_id=self.user.id,
            filename=filename,
            original_format="txt",
            document_type="commercial_contract",
            status="analyzed",
            processing_stage="completed",
            overall_risk_score=25.0,
            risk_band="Low",
        )
        db.session.add(doc)
        db.session.commit()
        self.test_docs.append(doc.id)
        return doc

    # =========================================================================
    # SECTION A: EMBEDDING RELIABILITY (MOCKED UNIT TESTS)
    # =========================================================================

    def test_embedding_timeout_bounded_retry_returns_none(self):
        """Mocked Test: Verify embed_text retries bounded times on TimeoutError and returns None, never zero-vector."""
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = TimeoutError("Deadline exceeded connecting to Gemini embedding API")

        with patch("app.services.rag.embedder.get_genai_client", return_value=mock_client), \
             patch("time.sleep", return_value=None):
            result = embed_text("Test query text for timeout handling", max_retries=3, initial_delay=0.01)

        self.assertIsNone(result, "embed_text must return None on timeout, not a fake zero vector")
        self.assertEqual(mock_client.models.embed_content.call_count, 3, "embed_text should retry transient timeout 3 times")

    def test_embedding_rate_limit_429_returns_none(self):
        """Mocked Test: Verify embed_text retries bounded times on HTTP 429 rate limit and returns None."""
        class Mock429Error(Exception):
            code = 429
            def __str__(self):
                return "429 ResourceExhausted: Quota exceeded for gemini-embedding-001"

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Mock429Error()

        with patch("app.services.rag.embedder.get_genai_client", return_value=mock_client), \
             patch("time.sleep", return_value=None):
            result = embed_query("How can I terminate this agreement?", max_retries=2)

        self.assertIsNone(result, "embed_query must return None on 429 rate limit")
        self.assertEqual(mock_client.models.embed_content.call_count, 2)

    def test_embedding_invalid_dimensions_rejected(self):
        """Mocked Test: Verify embed_text rejects vectors with incorrect dimensionality (e.g. 512 instead of 768)."""
        mock_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.1] * 512  # Invalid 512-dim vector
        mock_response.embeddings = [mock_emb]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response

        with patch("app.services.rag.embedder.get_genai_client", return_value=mock_client):
            result = embed_text("Check invalid dimensions")

        self.assertIsNone(result, "embed_text must reject 512-dim vector when 768 is expected")
        self.assertFalse(validate_embedding_vector([0.1] * 512, expected_dim=768))

    def test_embedding_nan_or_infinite_values_rejected(self):
        """Mocked Test: Verify embed_text rejects vectors containing NaN, +Inf, or -Inf."""
        nan_vec = [0.05] * 768
        nan_vec[42] = float("nan")

        inf_vec = [0.05] * 768
        inf_vec[100] = float("inf")

        neg_inf_vec = [0.05] * 768
        neg_inf_vec[200] = float("-inf")

        self.assertFalse(validate_embedding_vector(nan_vec, 768), "Vector with NaN must be rejected")
        self.assertFalse(validate_embedding_vector(inf_vec, 768), "Vector with +Inf must be rejected")
        self.assertFalse(validate_embedding_vector(neg_inf_vec, 768), "Vector with -Inf must be rejected")

        # Verify embed_text rejects response containing NaN
        mock_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = nan_vec
        mock_response.embeddings = [mock_emb]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response

        with patch("app.services.rag.embedder.get_genai_client", return_value=mock_client):
            result = embed_text("Text with nan vector")

        self.assertIsNone(result, "embed_text must return None when vector contains NaN")

    def test_batch_embed_texts_no_silent_zero_padding(self):
        """Mocked Test: Verify batch_embed_texts returns None for failed chunks instead of silent fake zeros."""
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = RuntimeError("Batch embedding failed")

        with patch("app.services.rag.embedder.get_genai_client", return_value=mock_client), \
             patch("time.sleep", return_value=None):
            results = batch_embed_texts(["Clause 1 text", "Clause 2 text"], batch_size=2, max_retries=1)

        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0], "Batch embedder must return None for failed items, not [0.0]*768")
        self.assertIsNone(results[1], "Batch embedder must return None for failed items, not [0.0]*768")

    # =========================================================================
    # SECTION B: PARAPHRASE RETRIEVAL & INTENT CLASSIFICATION (UNIT TESTS)
    # =========================================================================

    def test_paraphrase_termination_distinct_actions(self):
        """Deterministic Test: Verify resignation, dismissal, lease surrender, and general termination retain distinct meanings."""
        # 1. Resignation (Employee action)
        resignation_intent = classify_query_intent("How do I give my notice and resign?")
        self.assertEqual(resignation_intent.intent, "TERMINATION_QUERY")
        self.assertEqual(resignation_intent.metadata.get("action_type"), "resignation")
        self.assertIn("resignation", resignation_intent.target_clause_types)
        # Must not cross-contaminate with dismissal or surrender terms
        for term in resignation_intent.expansion_terms:
            self.assertNotIn("dismissal", term.lower())
            self.assertNotIn("fire", term.lower())
            self.assertNotIn("surrender", term.lower())

        # 2. Dismissal (Employer action)
        dismissal_intent = classify_query_intent("Can they fire me if I make a mistake?")
        self.assertEqual(dismissal_intent.intent, "TERMINATION_QUERY")
        self.assertEqual(dismissal_intent.metadata.get("action_type"), "dismissal")
        self.assertIn("dismissal", dismissal_intent.target_clause_types)
        # Must not cross-contaminate with resignation
        for term in dismissal_intent.expansion_terms:
            self.assertNotIn("resignation", term.lower())
            self.assertNotIn("voluntary", term.lower())

        # 3. Lease Surrender (Tenant action)
        surrender_intent = classify_query_intent("Can I vacate early and hand back the keys?")
        self.assertEqual(surrender_intent.intent, "TERMINATION_QUERY")
        self.assertEqual(surrender_intent.metadata.get("action_type"), "lease_surrender")
        self.assertIn("surrender", surrender_intent.target_clause_types)
        self.assertIn("vacating", surrender_intent.target_clause_types)
        for term in surrender_intent.expansion_terms:
            self.assertNotIn("dismissal", term.lower())
            self.assertNotIn("fire", term.lower())

        # 4. General Termination
        general_intent = classify_query_intent("How can this agreement be terminated?")
        self.assertEqual(general_intent.intent, "TERMINATION_QUERY")
        self.assertEqual(general_intent.metadata.get("action_type"), "general_termination")

    def test_paraphrase_price_escalation_intent(self):
        """Deterministic Test: Verify questions about rent/price increases map to PRICE_ESCALATION_QUERY."""
        queries = [
            "Can the landlord raise rent during the term?",
            "Can prices go up?",
            "Is there an annual rent increase or CPI escalation?",
            "Can the provider hike the monthly fee?",
        ]
        for q in queries:
            intent_obj = classify_query_intent(q)
            self.assertEqual(
                intent_obj.intent,
                "PRICE_ESCALATION_QUERY",
                f"Query '{q}' should be classified as PRICE_ESCALATION_QUERY, got {intent_obj.intent}"
            )
            self.assertTrue(
                any("escalation" in t or "increase" in t for t in intent_obj.expansion_terms),
                f"Expansion terms for '{q}' should contain escalation or increase"
            )

    def test_negation_and_contractual_conditions_preserved(self):
        """Deterministic Test: Verify negation conditions ('without cause', 'without notice', 'without penalty') are preserved."""
        queries_with_conditions = [
            ("Can they fire me without cause?", "without cause"),
            ("Can I terminate the agreement without notice?", "without notice"),
            ("Can I vacate the premises without penalty?", "without penalty"),
        ]
        for query, expected_negation in queries_with_conditions:
            intent_obj = classify_query_intent(query)
            self.assertTrue(intent_obj.metadata.get("negation_detected"), f"Negation should be detected in: '{query}'")
            negated_conds = intent_obj.metadata.get("negated_conditions", [])
            self.assertTrue(
                any(expected_negation in c for c in negated_conds),
                f"Expected '{expected_negation}' in negated conditions: {negated_conds}"
            )
            # Verify the negated condition is prioritized in expansion terms
            self.assertIn(
                expected_negation,
                intent_obj.expansion_terms,
                f"Expansion terms should include verbatim negated condition '{expected_negation}'"
            )

    # =========================================================================
    # SECTION C: RETRIEVAL FALLBACK & FUSION (DATABASE TESTS)
    # =========================================================================

    def test_semantic_search_unavailable_with_useful_lexical_results(self):
        """Database Test: When query_vector is None (failed/unavailable), lexical & structured retrieval succeed."""
        test_doc = self._create_test_document("phase2_fallback_test.txt")

        c1 = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=1,
            clause_type="lease_term",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="The term of this lease shall commence on October 1 and continue for twelve calendar months.",
            embedding=None,  # No embedding
        )
        c2 = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=2,
            clause_type="rent_payment",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="Tenant shall pay monthly rent of $2,500 on or before the first day of each month.",
            embedding=None,
        )
        c3 = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=3,
            clause_type="surrender",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="Upon early surrender or vacating of the premises, tenant must hand over all keys and restore possession.",
            embedding=None,
        )
        db.session.add_all([c1, c2, c3])
        db.session.commit()

        # Query vector is None (embedding failed or timed out)
        results = retrieve_relevant_clauses(
            document_id=test_doc.id,
            query_vector=None,
            query_text="Can I vacate early and hand over the keys?",
            top_k=2,
        )

        self.assertGreater(len(results), 0, "Retriever must not return empty when lexical results match, even without vector")
        # Top retrieved clause must be the surrender clause (Clause index 3)
        self.assertEqual(results[0]["clause_index"], 3)
        self.assertIn("keys", results[0]["redacted_text"].lower())
        self.assertIn("surrender", results[0]["redacted_text"].lower())

    def test_zero_lexical_overlap_with_valid_semantic_candidate(self):
        """Database Test: When query keywords have zero lexical overlap with clause body, valid vector cosine retrieval ranks the candidate."""
        test_doc = self._create_test_document("phase2_semantic_only.txt")

        # Dummy normalized 768-dim vector pointing along dim 0
        target_vec = [0.0] * 768
        target_vec[0] = 1.0

        other_vec = [0.0] * 768
        other_vec[1] = 1.0

        c_target = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=1,
            clause_type="custom_clause",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            # Zero lexical overlap with query "orbital satellite communications"
            redacted_text="The company reserves the right to terminate employment immediately for gross misconduct.",
            embedding=target_vec,
        )
        c_other = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=2,
            clause_type="custom_clause",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="Governing law shall be the State of California.",
            embedding=other_vec,
        )
        db.session.add_all([c_target, c_other])
        db.session.commit()

        # Query with zero lexical words matching c_target, but with query_vector aligned with c_target
        query_vector = [0.0] * 768
        query_vector[0] = 0.99
        query_vector[2] = 0.01

        results = retrieve_relevant_clauses(
            document_id=test_doc.id,
            query_vector=query_vector,
            query_text="orbital satellite communications",
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["clause_index"], 1, "Semantic search must retrieve candidate even with 0 lexical overlap")

    def test_no_relevant_clause_genuinely_present_returns_empty(self):
        """Database Test: Out-of-scope question with no lexical or vector match returns [], NEVER first five clauses."""
        test_doc = self._create_test_document("phase2_no_match.txt")

        c1 = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=1,
            clause_type="rent_payment",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="Rent is two thousand dollars per month.",
            embedding=None,
        )
        c2 = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=2,
            clause_type="governing_law",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="This agreement is governed by the laws of Texas.",
            embedding=None,
        )
        db.session.add_all([c1, c2])
        db.session.commit()

        results = retrieve_relevant_clauses(
            document_id=test_doc.id,
            query_vector=None,
            query_text="What are the rules regarding space shuttle launch permits and deep sea dredging?",
            top_k=5,
        )
        self.assertEqual(
            results,
            [],
            "Retriever must return empty list when no relevant clauses exist, never arbitrary first five"
        )

    def test_preservation_of_phase1_boilerplate_ranking(self):
        """Database Test: Verify Phase 1 Robertson-Spärck Jones IDF stops 'Entire Agreement' repeating 'agreement' 7x from outranking Early Termination."""
        test_doc = self._create_test_document("phase2_boilerplate_test.txt")

        # Boilerplate repeating "agreement" 7 times
        c_entire = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=1,
            clause_type="entire_agreement",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text=(
                "This agreement constitutes the entire agreement between the parties. "
                "No other agreement, oral agreement, or written agreement shall supersede this agreement. "
                "Any prior agreement is terminated by this agreement."
            ),
            embedding=None,
        )
        # Operative early termination clause mentioning agreement once
        c_term = Clause(
            id=uuid.uuid4(),
            document_id=test_doc.id,
            clause_index=2,
            clause_type="termination_notice",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text=(
                "Either party may terminate this agreement early by providing thirty days written notice "
                "to the other party in case of material breach or default."
            ),
            embedding=None,
        )
        db.session.add_all([c_entire, c_term])
        db.session.commit()

        results = retrieve_relevant_clauses(
            document_id=test_doc.id,
            query_vector=None,
            query_text="How can I terminate the agreement early?",
            top_k=2,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(
            results[0]["clause_index"],
            2,
            "Early termination clause must outrank repetitive 'Entire Agreement' boilerplate"
        )

    # =========================================================================
    # SECTION D: LIVE DATABASE & PGVECTOR COMPATIBILITY (INTEGRATION TEST)
    # =========================================================================

    def test_live_db_pgvector_compatibility(self):
        """REAL INTEGRATION TEST: Verify pgvector 768-dim column compatibility and cosine distance query in Neon PostgreSQL."""
        live_test_doc = self._create_test_document("live_pgvector_compatibility_test.txt")

        test_768_vec = [1.0 / math.sqrt(768)] * 768

        test_clause = Clause(
            id=uuid.uuid4(),
            document_id=live_test_doc.id,
            clause_index=1,
            clause_type="payment_terms",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE",
            redacted_text="Live pgvector integration test clause text.",
            embedding=test_768_vec,
        )
        db.session.add(test_clause)
        db.session.commit()

        # Execute actual pgvector cosine distance query against Neon PostgreSQL
        distance_col = Clause.embedding.cosine_distance(test_768_vec)
        live_result = (
            db.session.query(Clause, distance_col.label("distance"))
            .filter(Clause.document_id == live_test_doc.id)
            .filter(Clause.embedding.isnot(None))
            .first()
        )

        self.assertIsNotNone(live_result, "Live Neon PostgreSQL pgvector query returned None")
        retrieved_c, distance = live_result
        self.assertEqual(str(retrieved_c.id), str(test_clause.id))
        # Identical normalized vectors have cosine distance ~ 0.0
        self.assertAlmostEqual(float(distance), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
