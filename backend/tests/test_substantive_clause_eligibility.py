import unittest
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import uuid
from datetime import datetime, timezone

from app.services.nlp.clause_eligibility_gate import (
    classify_content_type,
    has_operative_legal_language,
    is_heading_or_subheading,
    is_salutation,
    is_party_metadata,
    is_date_metadata,
    is_address_metadata,
    is_definition_clause,
    CONTENT_TYPE_HEADING,
    CONTENT_TYPE_SALUTATION,
    CONTENT_TYPE_PARTY_METADATA,
    CONTENT_TYPE_DATE_METADATA,
    CONTENT_TYPE_PAGE_HEADER,
    CONTENT_TYPE_PAGE_FOOTER,
    CONTENT_TYPE_SIGNATURE_BLOCK,
    CONTENT_TYPE_OCR_NOISE,
    CONTENT_TYPE_EMPTY_OR_WHITESPACE,
    CONTENT_TYPE_SUBSTANTIVE_CLAUSE,
    CONTENT_TYPE_DEFINITION,
    CONTENT_TYPE_UNCERTAIN,
)
from app.services.nlp.block_segmenter import (
    segment_blocks,
    get_sequential_spans,
)
from app.services.risk.risk_engine import (
    compute_clause_risk,
    compute_document_risk,
    compute_risk_band,
    evaluate_clause_risk_and_review,
)
from app.models.clause import Clause


class TestSubstantiveClauseEligibility(unittest.TestCase):
    """
    Automated regression test suite verifying the elimination of false risk scores
    from headings, salutations, party metadata, addresses, dates, signatures,
    and non-substantive structural text (Tests A through J and edge cases).
    """

    def test_a_heading_only_document(self):
        """
        Test A: Heading-only document
        Input: "RESIDENTIAL LEASE AGREEMENT"
        Expected: No substantive clause detected, zero risk contribution,
        and an appropriate insufficient-content status.
        """
        text = "RESIDENTIAL LEASE AGREEMENT"
        blocks = segment_blocks(text, document_type="rental_agreement")
        
        self.assertGreaterEqual(len(blocks), 1)
        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 0, "Heading must not be classified as substantive clause")
        
        for b in blocks:
            self.assertEqual(b["risk_score"], 0.0)
            self.assertEqual(b["classification"], "NOT_APPLICABLE")
            self.assertIn(b["content_type"], ("DOCUMENT_TITLE", "SECTION_HEADING", "HEADING"))
            self.assertIsNotNone(b["filter_reason"])

        # Risk aggregation
        clause_scores = [b["risk_score"] for b in substantive]
        has_substantive = bool(substantive)
        doc_risk = compute_document_risk(clause_scores, []) if has_substantive else 0.0
        band = compute_risk_band(doc_risk, has_substantive_content=has_substantive)

        self.assertEqual(doc_risk, 0.0)
        self.assertEqual(band, "insufficient_content")

    def test_b_salutation_only_document(self):
        """
        Test B: Salutation-only document
        Input: "Dear Tenant,"
        Expected: No substantive risk contribution and insufficient_content band.
        """
        text = "Dear Tenant,"
        blocks = segment_blocks(text, document_type="rental_agreement")
        
        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 0, "Salutation must not be classified as substantive clause")
        
        for b in blocks:
            self.assertEqual(b["risk_score"], 0.0)
            self.assertEqual(b["classification"], "NOT_APPLICABLE")
            self.assertIn(b["content_type"], ("GREETING", "SALUTATION"))

        has_substantive = bool(substantive)
        doc_risk = compute_document_risk([b["risk_score"] for b in substantive], []) if has_substantive else 0.0
        band = compute_risk_band(doc_risk, has_substantive_content=has_substantive)

        self.assertEqual(doc_risk, 0.0)
        self.assertEqual(band, "insufficient_content")

    def test_c_party_metadata(self):
        """
        Test C: Party metadata
        Input: "LANDLORD: John Smith\nTENANT: Jane Smith\nAddress: 123 Example Street"
        Expected: Metadata is excluded from risk scoring.
        """
        text = "LANDLORD: John Smith\nTENANT: Jane Smith\nAddress: 123 Example Street"
        blocks = segment_blocks(text, document_type="rental_agreement")

        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 0, "Party and address metadata must not be classified as substantive")

        for b in blocks:
            self.assertEqual(b["risk_score"], 0.0)
            self.assertEqual(b["classification"], "NOT_APPLICABLE")
            self.assertIn(b["content_type"], ("PARTY_METADATA", "ADDRESS_BLOCK", "ADDRESS"))

        has_substantive = bool(substantive)
        doc_risk = compute_document_risk([b["risk_score"] for b in substantive], []) if has_substantive else 0.0
        band = compute_risk_band(doc_risk, has_substantive_content=has_substantive)

        self.assertEqual(doc_risk, 0.0)
        self.assertEqual(band, "insufficient_content")

    def test_d_heading_followed_by_risky_clause(self):
        """
        Test D: Heading followed by a risky clause
        Input:
        "TERMINATION\nThe tenant must pay the outstanding contractual charges specified in this agreement upon valid termination."
        Expected: The heading contributes zero risk, and the substantive provision is evaluated normally.
        """
        text = (
            "TERMINATION\n"
            "The tenant must pay the outstanding contractual charges specified in this agreement upon valid termination."
        )
        blocks = segment_blocks(text, document_type="rental_agreement")

        # Heading block
        heading_blocks = [b for b in blocks if "TERMINATION" in b["text"] and not b["is_substantive"]]
        self.assertGreaterEqual(len(heading_blocks), 1)
        self.assertEqual(heading_blocks[0]["risk_score"], 0.0)
        self.assertEqual(heading_blocks[0]["classification"], "NOT_APPLICABLE")

        # Substantive clause block
        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 1)
        self.assertIn("outstanding contractual charges", substantive[0]["text"])

        # Invariant: text[start:end] == block text
        for b in blocks:
            self.assertEqual(text[b["start_offset"]:b["end_offset"]], b["text"])

        # Substantive clause evaluates normally in risk engine
        eval_res = evaluate_clause_risk_and_review(
            clause_type="termination",
            favorability_label="unfavorable",
            favorability_confidence=0.85,
            content_type=substantive[0]["content_type"],
            is_substantive=True,
        )
        self.assertGreater(eval_res["risk_score"], 0.0)

    def test_e_mixed_content(self):
        """
        Test E: Mixed content
        Input: "PAYMENT: The tenant shall pay rent on the first day of each month."
        Expected: The heading is separated from the operative clause where reliable splitting
        is possible. The payment obligation remains analyzable.
        """
        text = "PAYMENT: The tenant shall pay rent on the first day of each month."
        blocks = segment_blocks(text, document_type="rental_agreement")

        self.assertEqual(len(blocks), 2, "Mixed content must be split into heading and substantive clause")
        
        # Heading span
        h_block = blocks[0]
        self.assertFalse(h_block["is_substantive"])
        self.assertEqual(h_block["risk_score"], 0.0)
        self.assertEqual(h_block["classification"], "NOT_APPLICABLE")
        self.assertEqual(h_block["text"], "PAYMENT")

        # Substantive span
        s_block = blocks[1]
        self.assertTrue(s_block["is_substantive"])
        self.assertEqual(s_block["text"], "The tenant shall pay rent on the first day of each month.")

        # Exact offsets check
        for b in blocks:
            self.assertEqual(text[b["start_offset"]:b["end_offset"]], b["text"])

    def test_f_definitions(self):
        """
        Test F: Definitions
        Input: "Affiliate means an entity controlling, controlled by, or under common control with a party."
        Expected: Preserve the definition and its relationships for downstream interpretation.
        Do not blindly remove it as metadata.
        """
        text = "Affiliate means an entity controlling, controlled by, or under common control with a party."
        meta = classify_content_type(text, document_type="rental_agreement")

        self.assertEqual(meta["content_type"], "DEFINITION")
        self.assertTrue(meta["is_substantive"], "Definitions must remain substantive for downstream legal interpretation")
        self.assertEqual(meta["risk_score"], 0.0, "Definitions do not carry independent unilateral risk")
        self.assertEqual(meta["classification"], "NOT_APPLICABLE")

        # Verify through segment_blocks
        blocks = segment_blocks(text, document_type="rental_agreement")
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0]["is_substantive"])
        self.assertEqual(blocks[0]["content_type"], "DEFINITION")
        self.assertEqual(blocks[0]["risk_score"], 0.0)

    def test_g_repeated_headers(self):
        """
        Test G: Repeated headers across multipage contract
        Input: Multipage contract with running document title on every page and genuine substantive clauses in body.
        Expected: The repeated title does not contribute risk, and body clauses remain analyzable.
        """
        multipage_doc = (
            "[PAGE 1]\n"
            "STANDARD RESIDENTIAL LEASE AGREEMENT\n"
            "Tenant shall pay monthly rent of $2,500 on the first day of each calendar month.\n"
            "Page 1 of 3\n\n"
            "[PAGE 2]\n"
            "STANDARD RESIDENTIAL LEASE AGREEMENT\n"
            "Tenant shall keep the leased premises in good order and sanitary condition.\n"
            "Page 2 of 3\n\n"
            "[PAGE 3]\n"
            "STANDARD RESIDENTIAL LEASE AGREEMENT\n"
            "Either party may terminate this agreement upon 30 days prior written notice.\n"
            "Page 3 of 3"
        )
        blocks = segment_blocks(multipage_doc, document_type="rental_agreement")

        running_titles = [b for b in blocks if b["text"] == "STANDARD RESIDENTIAL LEASE AGREEMENT"]
        self.assertEqual(len(running_titles), 3)
        for rt in running_titles:
            self.assertFalse(rt["is_substantive"])
            self.assertEqual(rt["risk_score"], 0.0)

        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 3)
        for s in substantive:
            self.assertIn(s["content_type"], ("SUBSTANTIVE_CLAUSE", "DEFINITION"))

    def test_h_low_confidence_substantive_block(self):
        """
        Test H: Low-confidence substantive block
        Input: A genuine contractual provision that the role classifier cannot confidently categorize.
        Expected: It is not silently excluded as safe. It follows the established uncertainty or human-review pathway.
        """
        ambiguous_provision = "In furtherance of the general understanding and operational arrangements of the sides"
        meta = classify_content_type(ambiguous_provision, document_type="rental_agreement")

        self.assertTrue(meta["is_substantive"])
        self.assertEqual(meta["classification"], "UNCERTAIN")
        self.assertTrue(meta["human_review"])
        self.assertEqual(meta["risk_score"], 0.0)

        # Risk engine routing
        eval_res = evaluate_clause_risk_and_review(
            clause_type="UNCERTAIN",
            favorability_label="neutral",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
        )
        self.assertTrue(eval_res["human_review"])
        self.assertEqual(eval_res["classification"], "UNCERTAIN")
        self.assertEqual(eval_res["favorability_label"], "needs_review")

    def test_i_aggregation_integrity(self):
        """
        Test I: Aggregation integrity
        Input: A document containing multiple eligible clauses vs the same document with extra
        headings, salutations, addresses, dates, and signature blocks.
        Expected: Adding extra structural metadata must NOT change the aggregate risk score.
        """
        base_clauses_text = (
            "Tenant shall pay monthly rent of $2,000 on the first day of each month.\n\n"
            "Landlord may enter the premises without notice in emergency situations."
        )
        blocks_base = segment_blocks(base_clauses_text, document_type="rental_agreement")
        sub_base = [b for b in blocks_base if b["is_substantive"]]
        self.assertEqual(len(sub_base), 2)

        # Compute clause risk scores
        scores_base = []
        for b in sub_base:
            c_type = "rent" if "rent" in b["text"] else "access"
            fav = "fair" if "rent" in b["text"] else "unfavorable"
            r = compute_clause_risk(c_type, fav, 0.9, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE")
            scores_base.append(r)
        
        doc_risk_base = compute_document_risk(scores_base, [])

        # Document with identical substantive clauses PLUS lots of non-substantive structural noise
        polluted_doc = (
            "RESIDENTIAL LEASE AGREEMENT\n\n"
            "Dear Tenant,\n\n"
            "LANDLORD: John Smith\nTENANT: Jane Smith\nAddress: 123 Main Street\nDate: October 14, 2026\n\n"
            "1. RENT AND PAYMENT\n"
            "Tenant shall pay monthly rent of $2,000 on the first day of each month.\n\n"
            "2. LANDLORD ACCESS\n"
            "Landlord may enter the premises without notice in emergency situations.\n\n"
            "IN WITNESS WHEREOF the parties have executed this Agreement:\n"
            "SIGNATURE OF LANDLORD: _______________\n"
            "SIGNATURE OF TENANT: _______________"
        )
        blocks_polluted = segment_blocks(polluted_doc, document_type="rental_agreement")
        sub_polluted = [b for b in blocks_polluted if b["is_substantive"]]
        self.assertEqual(len(sub_polluted), 2, "Only the 2 operative clauses must be substantive")

        scores_polluted = []
        for b in sub_polluted:
            c_type = "rent" if "rent" in b["text"] else "access"
            fav = "fair" if "rent" in b["text"] else "unfavorable"
            r = compute_clause_risk(c_type, fav, 0.9, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE")
            scores_polluted.append(r)

        doc_risk_polluted = compute_document_risk(scores_polluted, [])

        # Invariant: Aggregation scores are strictly identical!
        self.assertEqual(doc_risk_base, doc_risk_polluted, "Adding structural metadata must not change aggregate risk score")
        self.assertEqual(compute_risk_band(doc_risk_base), compute_risk_band(doc_risk_polluted))

    def test_j_downstream_and_cached_outputs(self):
        """
        Test J: Downstream and cached outputs
        Input: A previously analyzed document whose stored structural blocks have stale, nonzero risk values.
        Expected: The current eligibility gate in Clause.to_dict() prevents excluded blocks from
        re-entering the risk calculation or being displayed as risky.
        """
        stale_clause = Clause(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            clause_index=0,
            redacted_text="TERMINATION",
            start_offset=0,
            end_offset=11,
            content_type="HEADING",
            is_substantive=False,
            risk_score=75.0,  # STALE legacy value
            clause_type="termination",  # STALE legacy prediction
            favorability_label="unfavorable",  # STALE legacy label
            human_review=True,  # STALE legacy flag
            filter_reason="Section heading without operative legal text",
        )

        d = stale_clause.to_dict()
        self.assertFalse(d["is_substantive"])
        self.assertEqual(d["risk_score"], 0.0, "Stale nonzero risk score must be defensively suppressed to 0.0")
        self.assertEqual(d["clause_type"], "NOT_APPLICABLE")
        self.assertEqual(d["classification"], "NOT_APPLICABLE")
        self.assertEqual(d["favorability_label"], "neutral")
        self.assertFalse(d["human_review"])
        self.assertEqual(d["clause_eligibility_score"], 0.0)

    def test_k_empty_and_whitespace_documents(self):
        """Test completely empty or whitespace documents."""
        for empty_val in ("", "   \n\t  \n"):
            blocks = segment_blocks(empty_val, document_type="rental_agreement")
            self.assertEqual(len(blocks), 0)
            band = compute_risk_band(0.0, has_substantive_content=False)
            self.assertEqual(band, "insufficient_content")

    def test_l_unicode_and_mixed_language_structural_labels(self):
        """Test Unicode headings, em-dashes, and multilingual structural labels."""
        multilingual_titles = [
            "CONDITIONS GÉNÉRALES DE VENTE",
            "TÉRMINOS Y CONDICIONES GENERALES",
            "ALLGEMEINE GESCHÄFTSBEDINGUNGEN",
            "SECTION 1 — CONFIDENTIALITY",
            "§ 4. Governing Law and Jurisdiction",
        ]
        for mt in multilingual_titles:
            meta = classify_content_type(mt, document_type="rental_agreement")
            self.assertFalse(meta["is_substantive"], f"Structural label unfairly marked substantive: {mt}")
            self.assertEqual(meta["risk_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
