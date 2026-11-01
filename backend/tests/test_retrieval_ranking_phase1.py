import os
import sys
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
from app.services.rag.retriever import (
    retrieve_relevant_clauses,
    compute_bm25_scores,
    extract_query_keywords,
)
from app.services.rag.intent import classify_query_intent


class TestRetrievalRankingPhase1(unittest.TestCase):
    """
    Regression and verification test suite for Phase 1 RAG Retrieval Ranking Repairs.

    Verifies:
    1. Repeated 'agreement' boilerplate does not overpower specific operative clauses (e.g., early termination).
    2. Incidental metadata substring matches (e.g., 'term' in 'determination') receive zero token boost.
    3. Exact legal terminology matches produce discriminative BM25 relevance scores.
    4. Valid vector candidates are preserved when lexical overlap is zero.
    5. Empty candidate lists return [] without arbitrary fallback to the first five clauses.
    6. Legal risk metadata (risk_score, favorability_label) is isolated from lexical relevance.
    7. Candidate pool size is separated from final bounded top_k.
    8. Substantive-clause eligibility gate and document scoping are strictly preserved.
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
        # Create test user
        self.user = User.query.filter_by(email="phase1_retrieval_test@clauseguard.ai").first()
        if not self.user:
            self.user = User(
                name="Phase 1 Test User",
                email="phase1_retrieval_test@clauseguard.ai",
                password_hash="dummy_hash"
            )
            db.session.add(self.user)
            db.session.commit()

        # Dedicated test document
        self.doc = Document(
            id=uuid.uuid4(),
            filename="phase1_retrieval_test_contract.txt",
            original_format="txt",
            document_type="commercial_contract",
            status="analyzed",
            user_id=self.user.id,
            overall_risk_score=35.0,
            risk_band="Medium",
            redacted_text="Commercial Agreement with multiple clauses for ranking evaluation.",
        )
        db.session.add(self.doc)
        db.session.commit()

        self.created_clause_ids = []

    def tearDown(self):
        # Clean up clauses and document created for this test run
        try:
            if self.created_clause_ids:
                db.session.query(Clause).filter(Clause.id.in_(self.created_clause_ids)).delete(synchronize_session=False)
            if self.doc:
                db.session.query(Clause).filter(Clause.document_id == self.doc.id).delete(synchronize_session=False)
                db.session.query(Document).filter(Document.id == self.doc.id).delete(synchronize_session=False)
            db.session.commit()
        except Exception:
            db.session.rollback()

    def _create_clause(self, index, clause_type, text, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", risk_score=10.0, favorability="fair", embedding=None):
        c = Clause(
            id=uuid.uuid4(),
            document_id=self.doc.id,
            clause_index=index,
            clause_type=clause_type,
            content_type=content_type,
            is_substantive=is_substantive,
            risk_score=risk_score,
            favorability_label=favorability,
            redacted_text=text,
            page_number=1,
            start_offset=index * 100,
            end_offset=(index + 1) * 100,
            embedding=embedding,
        )
        db.session.add(c)
        db.session.commit()
        self.created_clause_ids.append(c.id)
        return c

    # -------------------------------------------------------------------------
    # 1. Boilerplate 'agreement' vs specific early termination clause
    # -------------------------------------------------------------------------
    def test_boilerplate_agreement_vs_early_termination(self):
        """
        Verify that an Entire Agreement boilerplate clause with 7x repetitions of 'agreement'
        does NOT outrank an Early Termination clause for query: 'Does this agreement allow early termination?'
        """
        c_boilerplate = self._create_clause(
            index=0,
            clause_type="entire_agreement",
            text=(
                "This Agreement constitutes the entire agreement between the parties with respect to the "
                "subject matter of this Agreement. No amendment or modification of this Agreement shall be "
                "valid or binding unless in an agreement in writing and signed by both parties to this Agreement. "
                "Any prior agreement or understanding is superseded by this Agreement."
            ),
            risk_score=5.0,
            favorability="fair"
        )
        c_termination = self._create_clause(
            index=1,
            clause_type="termination_clause",
            text=(
                "Either party may terminate this agreement prior to the expiration date upon thirty days "
                "written notice of early termination without cause."
            ),
            risk_score=15.0,
            favorability="fair"
        )

        query = "Does this agreement allow early termination?"
        keywords = extract_query_keywords(query)
        self.assertIn("agreement", keywords)
        self.assertIn("early", keywords)
        self.assertIn("termination", keywords)

        # Direct unit BM25 test
        bm25_results = compute_bm25_scores([c_boilerplate, c_termination], query_keywords=keywords)
        score_boilerplate = dict(bm25_results)[c_boilerplate]
        score_termination = dict(bm25_results)[c_termination]

        # The specific termination clause must score higher than the boilerplate clause
        self.assertGreater(
            score_termination,
            score_boilerplate,
            f"Expected termination score ({score_termination}) > boilerplate score ({score_boilerplate})"
        )

        # End-to-end retrieval ranking test
        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=[0.0] * 768,
            query_text=query,
            top_k=2
        )
        self.assertTrue(len(retrieved) > 0)
        self.assertEqual(retrieved[0]["clause_id"], str(c_termination.id))
        self.assertEqual(retrieved[0]["clause_type"], "termination_clause")

    # -------------------------------------------------------------------------
    # 2. Incidental metadata substring matches
    # -------------------------------------------------------------------------
    def test_incidental_metadata_substring_matches(self):
        """
        Verify that query term 'term' does not trigger an accidental metadata boost
        in clause_type='determination_of_disputes' where 'term' is merely an incidental substring.
        """
        c_dispute = self._create_clause(
            index=0,
            clause_type="determination_of_disputes",
            text="All disputes and claims shall be submitted to final and binding arbitration in New York.",
            risk_score=10.0,
            favorability="fair"
        )
        c_duration = self._create_clause(
            index=1,
            clause_type="term_and_duration",
            text="The initial term of this contract shall be three years from the effective date.",
            risk_score=10.0,
            favorability="fair"
        )

        query = "What is the term of this contract?"
        keywords = extract_query_keywords(query)
        self.assertIn("term", keywords)
        self.assertIn("contract", keywords)

        bm25_results = compute_bm25_scores([c_dispute, c_duration], query_keywords=keywords)
        scores = dict(bm25_results)

        # Dispute clause has 0 occurrences of 'term' in body and 'term' is only a substring of 'determination'
        self.assertEqual(scores[c_dispute], 0.0, "Incidental substring in clause_type must not grant BM25 score")
        self.assertGreater(scores[c_duration], 0.0)

    # -------------------------------------------------------------------------
    # 3. Exact legal terminology
    # -------------------------------------------------------------------------
    def test_exact_legal_terminology(self):
        """
        Verify that specific legal terminology ('indemnify and hold harmless')
        produces discriminative scores and properly ranks the indemnification clause.
        """
        c_indemnity = self._create_clause(
            index=0,
            clause_type="indemnification",
            text="Provider shall defend, indemnify and hold harmless Customer against all third-party claims.",
            risk_score=75.0,
            favorability="unfavorable"
        )
        c_notice = self._create_clause(
            index=1,
            clause_type="notices",
            text="All notices shall be in writing and delivered to the registered office of each party.",
            risk_score=5.0,
            favorability="neutral"
        )

        query = "indemnify and hold harmless"
        keywords = extract_query_keywords(query)
        self.assertIn("indemnify", keywords)
        self.assertIn("harmless", keywords)

        bm25_results = compute_bm25_scores([c_indemnity, c_notice], query_keywords=keywords)
        scores = dict(bm25_results)

        self.assertGreater(scores[c_indemnity], 0.0)
        self.assertEqual(scores[c_notice], 0.0)

        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=[0.0] * 768,
            query_text=query,
            top_k=2
        )
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["clause_id"], str(c_indemnity.id))

    # -------------------------------------------------------------------------
    # 4. Zero lexical overlap with valid vector candidates
    # -------------------------------------------------------------------------
    def test_zero_lexical_overlap_with_valid_vector(self):
        """
        Verify that when lexical BM25 overlap is zero, a valid vector candidate
        is preserved, retrieved, and assigned a matched status.
        """
        # Clause text uses synonyms/paraphrase: 'discontinue operations'
        dummy_vec = [0.05] * 768
        c_vector = self._create_clause(
            index=0,
            clause_type="service_level",
            text="The vendor may cease operational support upon ninety calendar days notification.",
            risk_score=20.0,
            favorability="fair",
            embedding=dummy_vec
        )

        query = "How can the software solution be abandoned?"
        # Query keywords have zero lexical match with the clause text
        keywords = extract_query_keywords(query)
        bm25_scores = compute_bm25_scores([c_vector], query_keywords=keywords)
        self.assertEqual(dict(bm25_scores)[c_vector], 0.0)

        # Call retriever with matching query_vector
        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=dummy_vec,
            query_text=query,
            top_k=2
        )

        self.assertTrue(len(retrieved) > 0, "Valid vector candidate must be retrieved despite zero lexical overlap")
        self.assertEqual(retrieved[0]["clause_id"], str(c_vector.id))
        self.assertEqual(retrieved[0]["retrieval_status"], "matched")
        self.assertGreater(retrieved[0]["similarity"], 0.90)

    # -------------------------------------------------------------------------
    # 5. Empty candidate lists without arbitrary first-five selection
    # -------------------------------------------------------------------------
    def test_empty_candidate_list_no_first_five(self):
        """
        Verify that when a query has no vector match, zero lexical overlap,
        and no structured intent match, the retriever returns an empty list []
        and never arbitrarily returns the first five clauses of the document.
        """
        self._create_clause(index=0, clause_type="preamble", text="This is the contract preamble.")
        self._create_clause(index=1, clause_type="services", text="Consulting services will be provided.")
        self._create_clause(index=2, clause_type="fees", text="Fees are payable on net 30 terms.")
        self._create_clause(index=3, clause_type="governing_law", text="Governed by the laws of New York.")
        self._create_clause(index=4, clause_type="confidentiality", text="Information shall be kept strictly secret.")

        # Completely unrelated out-of-domain query
        query = "Are astronauts authorized to pilot spacecraft on the moon?"
        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=[0.0] * 768,
            query_text=query,
            top_k=5
        )

        # Must return empty list, NOT the first 5 clauses!
        self.assertEqual(retrieved, [], "Retriever must return [] when no evidence is found, not arbitrary first 5")

    # -------------------------------------------------------------------------
    # 6. Metadata boost isolation: Risk and favorability must not bias BM25
    # -------------------------------------------------------------------------
    def test_metadata_boost_isolation(self):
        """
        Verify that favorability_label='unfavorable' and high risk_score=95.0
        do not artificially inflate lexical BM25 relevance for unrelated queries.
        """
        c_unrelated_risky = self._create_clause(
            index=0,
            clause_type="intellectual_property",
            text="All inventions, patents and trade secrets remain exclusive property of Licensor.",
            risk_score=95.0,
            favorability="unfavorable"
        )
        c_relevant_safe = self._create_clause(
            index=1,
            clause_type="payment_schedule",
            text="Client shall submit payment for invoices within twenty business days of receipt.",
            risk_score=5.0,
            favorability="fair"
        )

        query = "When is payment due?"
        keywords = extract_query_keywords(query)
        self.assertIn("payment", keywords)
        self.assertIn("due", keywords)

        bm25_results = compute_bm25_scores([c_unrelated_risky, c_relevant_safe], query_keywords=keywords)
        scores = dict(bm25_results)

        # Risky clause has zero match for payment keywords
        self.assertEqual(scores[c_unrelated_risky], 0.0, "High risk and unfavorable label must not grant BM25 score")
        self.assertGreater(scores[c_relevant_safe], 0.0)

        # End-to-end retrieval ranking
        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=[0.0] * 768,
            query_text=query,
            top_k=2
        )
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["clause_id"], str(c_relevant_safe.id))

    # -------------------------------------------------------------------------
    # 7. Candidate pool size separation from bounded top_k
    # -------------------------------------------------------------------------
    def test_candidate_pool_size_separation(self):
        """
        Verify that candidate_pool_size allows pooling broader candidates
        while final context is strictly bounded by top_k.
        """
        for i in range(10):
            self._create_clause(
                index=i,
                clause_type=f"clause_type_{i}",
                text=f"This section {i} details payment obligation number {i} for financial accounting.",
                risk_score=10.0 + i,
                favorability="fair"
            )

        query = "payment obligation accounting"
        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=[0.0] * 768,
            query_text=query,
            top_k=3,
            candidate_pool_size=8
        )

        # Final result must be strictly bounded to top_k = 3
        self.assertEqual(len(retrieved), 3)

    # -------------------------------------------------------------------------
    # 8. Substantive-clause eligibility gate preservation
    # -------------------------------------------------------------------------
    def test_substantive_clause_eligibility_gate(self):
        """
        Verify that non-substantive clauses (e.g. headers, page numbers) are excluded
        from substantive legal queries, while party queries can access title/signatory blocks.
        """
        c_header = self._create_clause(
            index=0,
            clause_type="page_header",
            text="Confidential Commercial Agreement Page 1 of 10",
            is_substantive=False,
            content_type="PAGE_HEADER"
        )
        c_substantive = self._create_clause(
            index=1,
            clause_type="confidentiality",
            text="Each party agrees to maintain confidentiality of all proprietary information.",
            is_substantive=True,
            content_type="SUBSTANTIVE_CLAUSE"
        )

        # General legal query: header should be filtered out by eligibility gate
        retrieved = retrieve_relevant_clauses(
            document_id=self.doc.id,
            query_vector=[0.0] * 768,
            query_text="What are the confidentiality obligations?",
            top_k=5
        )
        retrieved_ids = [r["clause_id"] for r in retrieved]
        self.assertIn(str(c_substantive.id), retrieved_ids)
        self.assertNotIn(str(c_header.id), retrieved_ids)


if __name__ == "__main__":
    unittest.main()
