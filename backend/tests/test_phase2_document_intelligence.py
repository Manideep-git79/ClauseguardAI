import unittest
import sys
from pathlib import Path
from PIL import Image

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.nlp.block_segmenter import segment_blocks, try_split_mixed_line, get_sequential_spans
from app.services.nlp.clause_eligibility_gate import (
    classify_content_type,
    has_operative_legal_language,
    is_salutation,
    is_party_metadata,
    is_definition_clause,
    CONTENT_TYPE_SUBSTANTIVE_CLAUSE,
    CONTENT_TYPE_DEFINITION,
    CONTENT_TYPE_PARTY_METADATA,
    CONTENT_TYPE_DOCUMENT_TITLE,
    CONTENT_TYPE_PAGE_METADATA,
    CONTENT_TYPE_SIGNATURE_BLOCK,
)
from app.services.risk.risk_engine import evaluate_clause_risk_and_review
from app.services.ingestion.extractor import (
    order_page_blocks_layout_aware,
    clean_ocr_text,
    merge_native_and_ocr_text,
)


class TestPhase2DocumentIntelligence(unittest.TestCase):
    """
    Comprehensive regression suite for Phase 2:
    - Comma-separated salutation splitting
    - Party/entity prefixes with and without operative covenants
    - Employment offer operative patterns
    - Definition handling and KPI consistency
    - Review explanation accuracy (0.45 threshold)
    - Multi-column layout-aware reading order
    - Exact character offset integrity
    - Zero metadata leakage
    - Native and OCR text merge parity
    """

    def test_salutation_splitting_with_operative_body(self):
        """Verifies that 'Dear Marcus, the Company will pay you...' splits into GREETING and SUBSTANTIVE_CLAUSE."""
        line = "Dear Marcus, the Company will pay you a starting annual base salary of $225,000.00."
        split_res = try_split_mixed_line(line)
        self.assertIsNotNone(split_res, "Comma-separated salutation with operative text must be split")
        h, delim, b = split_res
        self.assertEqual(h, "Dear Marcus")
        self.assertEqual(b, "the Company will pay you a starting annual base salary of $225,000.00.")

        # Test full block segmentation
        blocks = segment_blocks(line, document_type="job_offer_letter")
        self.assertEqual(len(blocks), 2, "Must produce exactly 2 blocks: greeting and substantive clause")
        
        # Block 0: Greeting metadata
        self.assertEqual(blocks[0]["text"], "Dear Marcus")
        self.assertEqual(blocks[0]["content_type"], "GREETING")
        self.assertFalse(blocks[0]["is_substantive"])
        self.assertEqual(blocks[0]["risk_score"], 0.0)

        # Block 1: Substantive clause
        self.assertEqual(blocks[1]["text"], "the Company will pay you a starting annual base salary of $225,000.00.")
        self.assertEqual(blocks[1]["content_type"], CONTENT_TYPE_SUBSTANTIVE_CLAUSE)
        self.assertTrue(blocks[1]["is_substantive"])

        # Offset integrity
        self.assertEqual(line[blocks[0]["start_offset"]:blocks[0]["end_offset"]], "Dear Marcus")
        self.assertEqual(line[blocks[1]["start_offset"]:blocks[1]["end_offset"]], blocks[1]["text"])

    def test_salutation_colon_separated_splitting(self):
        """Verifies colon-separated salutations with operative obligations split cleanly."""
        line = "Dear Tenant: You shall pay all utility invoices within thirty (30) days of receipt."
        split_res = try_split_mixed_line(line)
        self.assertIsNotNone(split_res)
        h, delim, b = split_res
        self.assertEqual(h, "Dear Tenant")
        self.assertEqual(b, "You shall pay all utility invoices within thirty (30) days of receipt.")

    def test_greeting_only_document_produces_no_substantive_clauses(self):
        """Verifies that courtesy greetings without operative commitments are excluded from substantive analysis."""
        doc_text = "Dear Marcus,\n\nGreetings.\n\nSincerely,\nHuman Resources"
        blocks = segment_blocks(doc_text, document_type="job_offer_letter")
        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 0, "Greeting-only documents must have 0 substantive clauses")
        for b in blocks:
            self.assertFalse(b["is_substantive"])
            self.assertEqual(b["risk_score"], 0.0)

    def test_party_prefix_with_operative_covenant_splits_cleanly(self):
        """Verifies that 'EMPLOYEE: Candidate shall maintain strict confidentiality...' splits and retains operative clause."""
        line = "EMPLOYEE: Candidate shall maintain strict confidentiality regarding all proprietary systems."
        split_res = try_split_mixed_line(line)
        self.assertIsNotNone(split_res, "Party prefix with operative covenant must be split")
        h, delim, b = split_res
        self.assertEqual(h, "EMPLOYEE")
        self.assertEqual(b, "Candidate shall maintain strict confidentiality regarding all proprietary systems.")

        blocks = segment_blocks(line, document_type="job_offer_letter")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["content_type"], CONTENT_TYPE_PARTY_METADATA)
        self.assertFalse(blocks[0]["is_substantive"])

        self.assertEqual(blocks[1]["content_type"], CONTENT_TYPE_SUBSTANTIVE_CLAUSE)
        self.assertTrue(blocks[1]["is_substantive"])
        self.assertEqual(blocks[1]["text"], "Candidate shall maintain strict confidentiality regarding all proprietary systems.")

    def test_party_prefix_without_operative_covenant_retained_as_metadata(self):
        """Verifies that 'TENANT: Jane Smith, 123 Main St' is preserved as pure metadata without splitting."""
        line = "TENANT: Jane Smith, 123 Main Street, Apt 4B, New York, NY"
        split_res = try_split_mixed_line(line)
        self.assertIsNone(split_res, "Pure party contact lines must not be split into pseudo-clauses")

        blocks = segment_blocks(line, document_type="rental_agreement")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["content_type"], CONTENT_TYPE_PARTY_METADATA)
        self.assertFalse(blocks[0]["is_substantive"])

    def test_employment_offer_declarative_salary_recognized(self):
        """Verifies declarative salary terms are recognized as substantive legal commitments."""
        text = "The Company will pay you a starting annual base salary of $225,000.00, payable in semi-monthly installments."
        self.assertTrue(has_operative_legal_language(text))
        meta = classify_content_type(text, document_type="job_offer_letter")
        self.assertTrue(meta["is_substantive"])
        self.assertEqual(meta["content_type"], CONTENT_TYPE_SUBSTANTIVE_CLAUSE)

    def test_employment_offer_bonus_and_equity_recognized(self):
        """Verifies bonus opportunity and RSU equity grant terms are recognized as substantive."""
        bonus_text = "Your target bonus opportunity will be thirty percent (30%) of your annual base salary."
        self.assertTrue(has_operative_legal_language(bonus_text))
        meta_bonus = classify_content_type(bonus_text, document_type="job_offer_letter")
        self.assertTrue(meta_bonus["is_substantive"])

        equity_text = "You will receive an initial grant of 15,000 Restricted Stock Units (RSUs) vesting over a four-year schedule."
        self.assertTrue(has_operative_legal_language(equity_text))
        meta_equity = classify_content_type(equity_text, document_type="job_offer_letter")
        self.assertTrue(meta_equity["is_substantive"])

    def test_employment_offer_benefits_and_reporting_recognized(self):
        """Verifies reporting structure, hybrid work, and benefit plan terms are recognized as substantive."""
        rep_text = "In this role, you will report directly to the Chief Information Security Officer (CISO)."
        self.assertTrue(has_operative_legal_language(rep_text))
        meta_rep = classify_content_type(rep_text, document_type="job_offer_letter")
        self.assertTrue(meta_rep["is_substantive"])

        benefit_text = "You will be eligible to participate in the Company's 401(k) Retirement Savings Plan with corporate match."
        self.assertTrue(has_operative_legal_language(benefit_text))
        meta_ben = classify_content_type(benefit_text, document_type="job_offer_letter")
        self.assertTrue(meta_ben["is_substantive"])

    def test_employment_offer_contingencies_and_at_will_recognized(self):
        """Verifies employment contingencies (background check) and at-will doctrine are recognized as substantive."""
        contingency_text = "This offer is contingent upon your successful completion of a comprehensive background check."
        self.assertTrue(has_operative_legal_language(contingency_text))
        meta_cont = classify_content_type(contingency_text, document_type="job_offer_letter")
        self.assertTrue(meta_cont["is_substantive"])

        at_will_text = "Your employment with Vantage Financial Technologies LLC is strictly at-will."
        self.assertTrue(has_operative_legal_language(at_will_text))
        meta_aw = classify_content_type(at_will_text, document_type="job_offer_letter")
        self.assertTrue(meta_aw["is_substantive"])

    def test_definition_clause_identified_with_zero_base_risk(self):
        """Verifies definitions are recognized as substantive with content_type == DEFINITION and 0 base risk."""
        def_text = '"Confidential Information" means all non-public technical data, algorithms, and business metrics.'
        self.assertTrue(is_definition_clause(def_text))
        meta = classify_content_type(def_text, document_type="commercial_contract")
        self.assertEqual(meta["content_type"], CONTENT_TYPE_DEFINITION)
        self.assertTrue(meta["is_substantive"])
        self.assertEqual(meta["risk_score"], 0.0)

    def test_uncertain_review_explanation_matches_45_threshold(self):
        """Verifies UNCERTAIN review explanation precisely reflects the < 0.45 threshold."""
        eval_res = evaluate_clause_risk_and_review(
            clause_type="UNCERTAIN",
            favorability_label="needs_review",
            favorability_confidence=0.35,
            content_type=CONTENT_TYPE_SUBSTANTIVE_CLAUSE,
            is_substantive=True
        )
        self.assertTrue(eval_res["human_review"])
        self.assertEqual(eval_res["risk_score"], 0.0)
        self.assertIn("<0.45", eval_res["review_reason"], "Explanation must state <0.45 threshold, not <0.25")

    def test_layout_aware_multi_column_ordering_preserves_column_flow(self):
        """Verifies that 2-column pages read Column 1 top-to-bottom completely before Column 2."""
        raw_blocks = [
            {"type": 0, "bbox": [50, 40, 550, 75], "lines": [{"spans": [{"text": "MASTER SERVICES AGREEMENT"}]}]},
            {"type": 0, "bbox": [50, 100, 275, 160], "lines": [{"spans": [{"text": "Col1: Section 1.1 Scope of Services"}]}]},
            {"type": 0, "bbox": [325, 105, 550, 165], "lines": [{"spans": [{"text": "Col2: Section 2.1 Payment Terms"}]}]},
            {"type": 0, "bbox": [50, 180, 275, 240], "lines": [{"spans": [{"text": "Col1: Section 1.2 Performance Standards"}]}]},
            {"type": 0, "bbox": [325, 185, 550, 245], "lines": [{"spans": [{"text": "Col2: Section 2.2 Late Invoices"}]}]},
            {"type": 0, "bbox": [50, 700, 550, 725], "lines": [{"spans": [{"text": "Page 1 of 5"}]}]},
        ]
        page_width = 600.0
        ordered = order_page_blocks_layout_aware(raw_blocks, page_width)
        ordered_texts = [b["lines"][0]["spans"][0]["text"] for b in ordered]

        expected_order = [
            "MASTER SERVICES AGREEMENT",
            "Col1: Section 1.1 Scope of Services",
            "Col1: Section 1.2 Performance Standards",
            "Col2: Section 2.1 Payment Terms",
            "Col2: Section 2.2 Late Invoices",
            "Page 1 of 5",
        ]
        self.assertEqual(ordered_texts, expected_order, "Multi-column layout must not interleave across columns")

    def test_single_column_layout_order_untouched(self):
        """Verifies single-column layout order is untouched."""
        raw_blocks = [
            {"type": 0, "bbox": [50, 50, 550, 90], "lines": [{"spans": [{"text": "Header"}]}]},
            {"type": 0, "bbox": [50, 120, 550, 180], "lines": [{"spans": [{"text": "Paragraph 1"}]}]},
            {"type": 0, "bbox": [50, 200, 550, 260], "lines": [{"spans": [{"text": "Paragraph 2"}]}]},
            {"type": 0, "bbox": [50, 280, 550, 340], "lines": [{"spans": [{"text": "Paragraph 3"}]}]},
        ]
        ordered = order_page_blocks_layout_aware(raw_blocks, 600.0)
        self.assertEqual(ordered, raw_blocks, "Single column blocks must remain in original order")

    def test_table_and_sidebar_blocks_preserve_reading_order(self):
        """Verifies pages with a sidebar block and main text do not crash layout sorting."""
        raw_blocks = [
            {"type": 0, "bbox": [40, 50, 560, 80], "lines": [{"spans": [{"text": "Title"}]}]},
            {"type": 0, "bbox": [40, 100, 180, 500], "lines": [{"spans": [{"text": "Sidebar Navigation Notes"}]}]},
            {"type": 0, "bbox": [200, 100, 560, 250], "lines": [{"spans": [{"text": "Main Content Clause 1"}]}]},
            {"type": 0, "bbox": [200, 270, 560, 420], "lines": [{"spans": [{"text": "Main Content Clause 2"}]}]},
        ]
        ordered = order_page_blocks_layout_aware(raw_blocks, 600.0)
        self.assertEqual(len(ordered), 4, "All blocks must be preserved")
        self.assertEqual(ordered[0]["lines"][0]["spans"][0]["text"], "Title")

    def test_native_and_ocr_text_merge_parity(self):
        """Verifies that duplicate lines between native digital extraction and OCR are safely deduplicated."""
        native_text = "SECTION 1: TERM\nThe term of this agreement shall commence on November 1, 2026.\n\nPage 1 of 1"
        ocr_text = "SECTION 1: TERM\nThe term of this agreement shall commence on November 1, 2026.\nADDENDUM: Exhibit A rules apply."
        merged = merge_native_and_ocr_text(native_text, ocr_text)
        self.assertIn("ADDENDUM: Exhibit A rules apply.", merged)
        self.assertEqual(merged.count("SECTION 1: TERM"), 1, "Duplicate section header must appear exactly once")
        self.assertEqual(merged.count("November 1, 2026"), 1, "Duplicate body text must appear exactly once")

    def test_complete_synthetic_contract_offsets_and_metadata_separation(self):
        """Tests end-to-end segmentation on a realistic synthetic contract ensuring zero metadata leakage."""
        contract_text = (
            "COMMERCIAL SERVICES AGREEMENT\n\n"
            "Date: November 1, 2026\n\n"
            "CLIENT: Acme Quantitative Trading LLC\n"
            "CONTRACTOR: SecureCloud Systems Inc.\n\n"
            "Dear Associate, we are pleased to offer you the position of Principal Security Architect.\n\n"
            "SECTION 1: COMPENSATION\n"
            "The Company will pay you a starting annual base salary of $250,000.00.\n\n"
            "\"Confidential Data\" means all proprietary algorithms and trading source code.\n\n"
            "IN WITNESS WHEREOF, the parties have executed this Agreement.\n"
            "Authorized Signature: __________________\n"
            "[PAGE 1]"
        )
        blocks = segment_blocks(contract_text, document_type="job_offer_letter")

        # Offset verification: every block matches contract_text exactly
        for b in blocks:
            s_off, e_off = b["start_offset"], b["end_offset"]
            self.assertEqual(contract_text[s_off:e_off], b["text"], f"Offset mismatch for block: {b['text']!r}")

        # Metadata separation
        title_blocks = [b for b in blocks if b["content_type"] == CONTENT_TYPE_DOCUMENT_TITLE]
        self.assertEqual(len(title_blocks), 1)
        self.assertFalse(title_blocks[0]["is_substantive"])

        greeting_blocks = [b for b in blocks if b["content_type"] == "GREETING"]
        self.assertEqual(len(greeting_blocks), 1)
        self.assertEqual(greeting_blocks[0]["text"], "Dear Associate")
        self.assertFalse(greeting_blocks[0]["is_substantive"])

        substantive = [b for b in blocks if b["is_substantive"]]
        sub_texts = [b["text"] for b in substantive]
        self.assertTrue(any("position of Principal Security Architect" in t for t in sub_texts))
        self.assertTrue(any("annual base salary of $250,000.00" in t for t in sub_texts))
        self.assertTrue(any('"Confidential Data" means' in t for t in sub_texts))


if __name__ == "__main__":
    unittest.main()
