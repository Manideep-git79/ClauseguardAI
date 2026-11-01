import unittest
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.nlp.clause_eligibility_gate import (
    classify_content_type,
    is_table_of_contents_entry,
    is_table_of_contents_block,
    has_operative_legal_language,
)
from app.services.nlp.block_segmenter import segment_blocks
from app.services.nlp.header_footer_detector import (
    is_page_metadata,
    is_running_header_footer,
    build_document_repeated_headers_footers,
)
from app.services.risk.risk_engine import (
    evaluate_clause_risk_and_review,
    compute_document_risk,
)

class TestPhase6SegmentationMetadata(unittest.TestCase):
    # 1. Cover-page metadata exclusion
    def test_01_cover_page_metadata_exclusion(self):
        cover_page_text = (
            "RESIDENTIAL LEASE AGREEMENT\n"
            "Dated: October 10, 2026\n"
            "By and Between: Landlord Properties LLC and Tenant\n"
            "Effective Date: November 1, 2026\n"
            "CONFIDENTIAL AND PROPRIETARY"
        )
        blocks = segment_blocks(cover_page_text, document_type="rental_agreement")
        for b in blocks:
            self.assertFalse(b["is_substantive"], f"Expected non-substantive block, got {b['content_type']}")
            self.assertEqual(b["risk_score"], 0.0)
            self.assertEqual(b["classification"], "NOT_APPLICABLE")

    # 2. Single-page and multi-page TOCs
    def test_02_table_of_contents_detection(self):
        toc_text = (
            "TABLE OF CONTENTS\n"
            "Section 1. Definitions .................................... 1\n"
            "Section 2. Term and Rent .................................. 3\n"
            "Section 3. Maintenance and Repairs ........................ 6\n"
            "Section 4. Termination and Default ........................ 10\n"
            "Section 5. Governing Law .................................. 15"
        )
        blocks = segment_blocks(toc_text, document_type="rental_agreement")
        self.assertGreater(len(blocks), 0)
        for b in blocks:
            self.assertFalse(b["is_substantive"])
            self.assertIn(b["content_type"], ("TABLE_OF_CONTENTS", "SECTION_HEADING"))
            self.assertEqual(b["risk_score"], 0.0)

    # 3. TOC entries versus genuine clauses with similar wording
    def test_03_toc_entry_vs_genuine_clause(self):
        toc_entry = "Section 1. Definitions .................................... 4"
        genuine_clause = (
            "Section 1. Definitions\n"
            "As used in this Agreement, 'Premises' shall mean 123 Main Street, Suite 400."
        )
        meta_toc = classify_content_type(toc_entry, document_type="rental_agreement")
        meta_clause = classify_content_type(genuine_clause, document_type="rental_agreement")

        self.assertEqual(meta_toc["content_type"], "TABLE_OF_CONTENTS")
        self.assertFalse(meta_toc["is_substantive"])
        self.assertEqual(meta_toc["risk_score"], 0.0)

        self.assertEqual(meta_clause["content_type"], "SUBSTANTIVE_CLAUSE")
        self.assertTrue(meta_clause["is_substantive"])

    # 4. Repeated headers and footers across pages
    def test_04_repeated_headers_and_footers(self):
        doc_text = (
            "[PAGE 1]\n"
            "CONFIDENTIAL LEASE AGREEMENT\n"
            "Tenant shall pay monthly rent of $2,000 on or before the 1st of each month.\n"
            "Page 1 of 2\n\n"
            "[PAGE 2]\n"
            "CONFIDENTIAL LEASE AGREEMENT\n"
            "Tenant shall maintain general liability insurance in the amount of $500,000.\n"
            "Page 2 of 2"
        )
        repeated = build_document_repeated_headers_footers(doc_text)
        self.assertIn("CONFIDENTIAL LEASE AGREEMENT", repeated)

        blocks = segment_blocks(doc_text, document_type="rental_agreement")
        running_headers = [b for b in blocks if "CONFIDENTIAL LEASE AGREEMENT" in b["text"]]
        for rh in running_headers:
            self.assertFalse(rh["is_substantive"])
            self.assertEqual(rh["risk_score"], 0.0)

    # 5. Page numbers and running titles (including roman numerals)
    def test_05_page_numbers_and_running_titles(self):
        page_examples = [
            "Page 1 of 21",
            "[PAGE 2]",
            "Page iv of xii",
            "- iv -",
            "- 5 -",
            "Page No. 12",
        ]
        for p in page_examples:
            self.assertTrue(is_page_metadata(p) or is_running_header_footer(p) or p.startswith("[PAGE"), f"Failed for {p}")
            meta = classify_content_type(p, document_type="rental_agreement")
            self.assertFalse(meta["is_substantive"])
            self.assertEqual(meta["risk_score"], 0.0)

    # 6. Legitimate short clauses must NOT be filtered
    def test_06_legitimate_short_clauses_preserved(self):
        short_clauses = [
            "The policy is non-assignable.",
            "Grace period: 30 days.",
            "This lease shall terminate on December 31, 2026.",
            "Tenant agrees to pay rent on the first day of each month.",
            "Governing law: State of California.",
        ]
        for sc in short_clauses:
            meta = classify_content_type(sc, document_type="rental_agreement")
            self.assertTrue(meta["is_substantive"], f"Short clause unfairly filtered: {sc}")
            self.assertEqual(meta["content_type"], "SUBSTANTIVE_CLAUSE")

    # 7. Nested numbering and lettered subsections kept together
    def test_07_nested_numbering_subsections_preserved_together(self):
        clause_with_subsections = (
            "4. Maintenance and Repairs:\n"
            "(a) Tenant shall maintain the interior premises in clean condition.\n"
            "(b) Landlord shall be responsible for exterior roof and foundation repairs.\n\n"
            "5. Alterations:\n"
            "Tenant shall not make alterations without Landlord's prior written consent."
        )
        blocks = segment_blocks(clause_with_subsections, document_type="rental_agreement")
        # Should be split into Section 4 and Section 5, NOT 4 separate fragments!
        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 2)
        self.assertIn("(a) Tenant shall maintain", substantive[0]["text"])
        self.assertIn("(b) Landlord shall be responsible", substantive[0]["text"])
        self.assertIn("5. Alterations:", substantive[1]["text"])

    # 8. Repeated identical lines with distinct, unambiguous source offsets
    def test_08_repeated_lines_distinct_sequential_offsets(self):
        text = (
            "Section 1. Notice.\n"
            "All notices shall be given in writing.\n\n"
            "Section 2. Preamble.\n"
            "The parties hereby agree to the terms herein.\n\n"
            "Section 3. Notice Repeat.\n"
            "All notices shall be given in writing."
        )
        blocks = segment_blocks(text, document_type="rental_agreement")
        notice_blocks = [b for b in blocks if "All notices shall be given in writing." in b["text"]]
        self.assertEqual(len(notice_blocks), 2)
        # Ensure start and end offsets are strictly different and non-overlapping
        self.assertNotEqual(notice_blocks[0]["start_offset"], notice_blocks[1]["start_offset"])
        self.assertLess(notice_blocks[0]["end_offset"], notice_blocks[1]["start_offset"])
        # Verify exact character slices match block text
        self.assertEqual(text[notice_blocks[0]["start_offset"]:notice_blocks[0]["end_offset"]], notice_blocks[0]["text"])
        self.assertEqual(text[notice_blocks[1]["start_offset"]:notice_blocks[1]["end_offset"]], notice_blocks[1]["text"])

    # 9. Multi-paragraph clauses with semicolons or continuations
    def test_09_multi_paragraph_clauses(self):
        multi_para_clause = (
            "Tenant shall comply with all building regulations;\n"
            "and Tenant shall not cause any nuisance or disturbance to neighbors."
        )
        blocks = segment_blocks(multi_para_clause, document_type="rental_agreement")
        substantive = [b for b in blocks if b["is_substantive"]]
        self.assertEqual(len(substantive), 1)
        self.assertIn("building regulations;", substantive[0]["text"])
        self.assertIn("nuisance or disturbance", substantive[0]["text"])

    # 10. Duplicate extraction artifacts versus repeated genuine obligations
    def test_10_extraction_duplicates_vs_genuine_repeated_obligations(self):
        # Consecutive identical blocks -> Duplicate extraction artifact
        consecutive_duplicate = (
            "Tenant shall pay monthly rent of $2,000 on or before the first.\n\n"
            "Tenant shall pay monthly rent of $2,000 on or before the first."
        )
        blocks_consec = segment_blocks(consecutive_duplicate, document_type="rental_agreement")
        self.assertEqual(len(blocks_consec), 2)
        self.assertTrue(blocks_consec[0]["is_substantive"])
        self.assertFalse(blocks_consec[1]["is_substantive"])
        self.assertEqual(blocks_consec[1]["content_type"], "EXTRACTION_DUPLICATE")
        self.assertEqual(blocks_consec[1]["risk_score"], 0.0)

        # Non-consecutive repeated obligations in distinct legal contexts -> Both preserved!
        distinct_context_text = (
            "Article 1: Monthly Obligations.\n"
            "Tenant shall maintain property insurance.\n\n"
            "Article 2: Additional Terms.\n"
            "Landlord shall provide trash removal services.\n\n"
            "Article 3: Special Covenants.\n"
            "Tenant shall maintain property insurance."
        )
        blocks_distinct = segment_blocks(distinct_context_text, document_type="rental_agreement")
        insurance_blocks = [b for b in blocks_distinct if "Tenant shall maintain property insurance." in b["text"]]
        self.assertEqual(len(insurance_blocks), 2)
        for ib in insurance_blocks:
            self.assertTrue(ib["is_substantive"])

    # 11. Unknown substantive clauses remain visible for human review
    def test_11_unknown_substantive_clauses_routed_to_review(self):
        ambiguous_text = "In furtherance of the general understanding of both sides"
        meta = classify_content_type(ambiguous_text, document_type="rental_agreement")
        self.assertTrue(meta["is_substantive"])
        self.assertEqual(meta["classification"], "UNCERTAIN")
        self.assertTrue(meta["human_review"])
        self.assertEqual(meta["risk_score"], 0.0)

        eval_res = evaluate_clause_risk_and_review(
            clause_type="UNCERTAIN",
            favorability_label="neutral",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
        )
        self.assertTrue(eval_res["human_review"])
        self.assertEqual(eval_res["favorability_label"], "needs_review")
        self.assertIsNotNone(eval_res["review_reason"])

    # 12. Excluded metadata contributes zero substantive risk
    def test_12_excluded_metadata_contributes_zero_risk(self):
        excluded_blocks = [
            "TABLE OF CONTENTS\n1. Rent ..... 4\n2. Term ..... 6",
            "Page 3 of 10",
            "CONFIDENTIAL OFFER LETTER",
            "Signature: ___________________ Date: ___________",
            "LIFE INSURANCE CORPORATION OF INDIA",
        ]
        for eb in excluded_blocks:
            meta = classify_content_type(eb, document_type="rental_agreement")
            self.assertFalse(meta["is_substantive"])
            self.assertEqual(meta["risk_score"], 0.0)
            eval_res = evaluate_clause_risk_and_review(
                clause_type=meta["classification"],
                favorability_label="neutral",
                content_type=meta["content_type"],
                is_substantive=meta["is_substantive"],
            )
            self.assertEqual(eval_res["risk_score"], 0.0)
            self.assertFalse(eval_res["human_review"])

        # When document has only excluded content, document risk is strictly 0.0
        doc_risk = compute_document_risk([], missing_clauses=[])
        self.assertEqual(doc_risk, 0.0)

    # 13. Clause counts remain mutually exclusive and sum correctly
    def test_13_clause_counts_mutually_exclusive_and_sum_correctly(self):
        # Synthetic list of substantive clauses
        from app.models.clause import Clause
        c1 = Clause(clause_index=0, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="unfavorable", risk_score=80.0)
        c2 = Clause(clause_index=1, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="needs_review", risk_score=0.0, human_review=True)
        c3 = Clause(clause_index=2, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", clause_type="UNCERTAIN", favorability_label="neutral", risk_score=0.0)
        c4 = Clause(clause_index=3, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="fair", risk_score=15.0)
        c5 = Clause(clause_index=4, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="neutral", risk_score=10.0)
        # Excluded metadata block
        c6 = Clause(clause_index=5, is_substantive=False, content_type="TABLE_OF_CONTENTS", favorability_label="neutral", risk_score=0.0)

        clauses = [c1, c2, c3, c4, c5, c6]
        substantive = [c for c in clauses if c.is_substantive and c.content_type == "SUBSTANTIVE_CLAUSE"]
        self.assertEqual(len(substantive), 5)

        unfavorable_count = sum(1 for c in substantive if (c.favorability_label or "").lower() == "unfavorable")
        review_count = sum(
            1 for c in substantive
            if (c.favorability_label or "").lower() != "unfavorable"
            and ((c.favorability_label or "").lower() == "needs_review"
                 or c.clause_type == "UNCERTAIN"
                 or c.human_review)
        )
        fair_count = max(0, len(substantive) - unfavorable_count - review_count)

        self.assertEqual(unfavorable_count, 1)
        self.assertEqual(review_count, 2)  # c2 and c3
        self.assertEqual(fair_count, 2)    # c4 and c5
        self.assertEqual(unfavorable_count + review_count + fair_count, len(substantive))

    # 14. Section heading attached to following clause body
    def test_14_section_heading_association(self):
        text = (
            "SECTION 3: GOVERNING LAW\n"
            "This Agreement shall be construed and governed in accordance with the laws of California."
        )
        blocks = segment_blocks(text, document_type="rental_agreement")
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0]["is_substantive"])
        self.assertIn("SECTION 3: GOVERNING LAW", blocks[0]["text"])
        self.assertIn("laws of California", blocks[0]["text"])
        # Invariant: text[start:end] == block text
        self.assertEqual(text[blocks[0]["start_offset"]:blocks[0]["end_offset"]], blocks[0]["text"])

    # 15. Exact offset invariant holds for every segmented block
    def test_15_exact_offset_invariants_across_document(self):
        doc = (
            "[PAGE 1]\n"
            "RESIDENTIAL LEASE AGREEMENT\n\n"
            "TABLE OF CONTENTS\n"
            "1. Rent ................................. 1\n"
            "2. Utilities ............................ 2\n\n"
            "1. Rent:\n"
            "Tenant shall pay monthly rent of $1,800.\n\n"
            "[PAGE 2]\n"
            "2. Utilities:\n"
            "Landlord shall pay for water and trash collection services."
        )
        blocks = segment_blocks(doc, document_type="rental_agreement")
        self.assertGreater(len(blocks), 3)
        for b in blocks:
            s = b["start_offset"]
            e = b["end_offset"]
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(e, len(doc))
            self.assertEqual(doc[s:e], b["text"], f"Offset mismatch for block: {b['text'][:30]}")

if __name__ == "__main__":
    unittest.main()
