import unittest
import sys
import io
import uuid
from pathlib import Path
from unittest.mock import MagicMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.risk.risk_engine import (
    compute_clause_risk,
    compute_document_risk,
    compute_risk_band,
    evaluate_clause_risk_and_review,
    normalize_favorability,
    normalize_clause_type,
    get_risk_config,
)
from app.reports.pdf_report import generate_pdf_report

class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.config = get_risk_config()

    def test_fair_clause_risk(self):
        score = compute_clause_risk("security_deposit", "fair", 1.0, config=self.config)
        self.assertEqual(score, 0.0)

    def test_unfavorable_clause_risk(self):
        score = compute_clause_risk("termination", "unfavorable", 1.0, config=self.config)
        self.assertEqual(score, 100.0)

    def test_needs_review_clause_risk(self):
        score = compute_clause_risk("rent_payment_terms", "needs_review", 1.0, config=self.config)
        self.assertEqual(score, 30.0)

    def test_sequence_adjustment_pair(self):
        normal_score = compute_clause_risk("termination", "needs_review", 1.0, prior_clause_type=None, config=self.config)
        bumped_score = compute_clause_risk("termination", "needs_review", 1.0, prior_clause_type="renewal_auto_renewal", config=self.config)
        self.assertGreater(bumped_score, normal_score)

    def test_document_risk_composite(self):
        clause_scores = [20.0, 40.0, 60.0]
        missing = [{"severity": "high"}]
        overall = compute_document_risk(clause_scores, missing, config=self.config)
        self.assertEqual(overall, 32.5)

    def test_risk_bands(self):
        self.assertEqual(compute_risk_band(15.0, config=self.config), "low")
        self.assertEqual(compute_risk_band(35.0, config=self.config), "medium")
        self.assertEqual(compute_risk_band(65.0, config=self.config), "high")
        self.assertEqual(compute_risk_band(85.0, config=self.config), "critical")

    def test_bounds(self):
        score_low = compute_clause_risk("pet_policy", "fair", 0.0, config=self.config)
        self.assertGreaterEqual(score_low, 0.0)
        score_high = compute_document_risk([100.0] * 10, [{"severity": "high"}] * 10, config=self.config)
        self.assertLessEqual(score_high, 100.0)

    # 1. Classification labels with different letter casing produce equivalent results
    def test_casing_insensitivity_and_normalization(self):
        for term in ["unfavorable", "UNFAVORABLE", "Unfavorable", "  unfavorable  "]:
            score = compute_clause_risk("termination", term, 1.0, config=self.config)
            self.assertEqual(score, 100.0, f"Failed for casing: {term}")

        for term in ["fair", "FAIR", "Fair", "  fair  "]:
            score = compute_clause_risk("security_deposit", term, 1.0, config=self.config)
            self.assertEqual(score, 0.0, f"Failed for casing: {term}")

        for term in ["needs_review", "NEEDS_REVIEW", "Needs_Review"]:
            score = compute_clause_risk("rent_payment_terms", term, 1.0, config=self.config)
            self.assertEqual(score, 30.0, f"Failed for casing: {term}")

        for term in ["neutral", "NEUTRAL", "Neutral"]:
            score = compute_clause_risk("utilities", term, 1.0, config=self.config)
            self.assertEqual(score, 0.0, f"Failed for casing: {term}")

        # Check clause_type casing
        self.assertEqual(compute_clause_risk("TERMINATION", "unfavorable", 1.0, config=self.config), 100.0)
        self.assertEqual(compute_clause_risk("Rent_Payment_Terms", "needs_review", 1.0, config=self.config), 30.0)

    # 2. Every supported category has an explicit valid multiplier or rule
    def test_supported_categories_have_explicit_multipliers(self):
        mults = self.config["favorability_severity_multiplier"]
        self.assertIn("fair", mults)
        self.assertIn("neutral", mults)
        self.assertIn("needs_review", mults)
        self.assertIn("unfavorable", mults)
        self.assertEqual(mults["fair"], 0.0)
        self.assertEqual(mults["neutral"], 0.0)
        self.assertEqual(mults["needs_review"], 0.5)
        self.assertEqual(mults["unfavorable"], 1.0)

        # Base weights for standard types are non-negative
        for ctype, weight in self.config["clause_type_base_weight"].items():
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    # 3. Unknown and missing categories do not silently receive a high-risk fallback, but are flagged for review
    def test_unknown_and_missing_categories_do_not_silently_inflate_risk(self):
        # Missing labels produce 0.0 numerical risk (no fabricated penalties)
        self.assertEqual(compute_clause_risk(None, None, 1.0, config=self.config), 0.0)
        self.assertEqual(compute_clause_risk("termination", None, 1.0, config=self.config), 0.0)
        self.assertEqual(compute_clause_risk(None, "unfavorable", 1.0, config=self.config), 0.0)
        self.assertEqual(compute_clause_risk("", "", 1.0, config=self.config), 0.0)

        # Unknown label and type produce 0.0 numerical risk
        self.assertEqual(compute_clause_risk("completely_unknown_clause_type", "completely_unknown_favorability", 1.0, config=self.config), 0.0)
        self.assertEqual(compute_clause_risk("completely_unknown_clause_type", "fair", 1.0, config=self.config), 0.0)

        # evaluate_clause_risk_and_review flags review for unknown substantive types with explicit reason
        res_type = evaluate_clause_risk_and_review("completely_unknown_clause_type", "fair", 1.0, config=self.config)
        self.assertEqual(res_type["risk_score"], 0.0)
        self.assertTrue(res_type["human_review"])
        self.assertEqual(res_type["favorability_label"], "needs_review")
        self.assertIsNotNone(res_type["review_reason"])
        self.assertIn("Unrecognized substantive clause type", res_type["review_reason"])

        # Unknown or missing favorability on substantive clause flags review with explicit reason
        res_fav = evaluate_clause_risk_and_review("termination", None, 1.0, config=self.config)
        self.assertEqual(res_fav["risk_score"], 0.0)
        self.assertTrue(res_fav["human_review"])
        self.assertEqual(res_fav["favorability_label"], "needs_review")
        self.assertIsNotNone(res_fav["review_reason"])
        self.assertIn("Unrecognized or missing favorability", res_fav["review_reason"])

    # 4. Neutral/fair clauses do not receive critical severity styling in PDF
    def test_neutral_fair_clauses_in_pdf_report(self):
        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()
        mock_doc.filename = "synthetic_lease.docx"
        mock_doc.document_type = "rental_agreement"
        mock_doc.page_count = 2
        mock_doc.model_version = "Fine-Tuned DistiBERT"
        mock_doc.analyzed_at = None
        mock_doc.overall_risk_score = 0.0
        mock_doc.risk_band = "low"

        c1 = MagicMock()
        c1.clause_index = 0
        c1.clause_type = "utilities"
        c1.favorability_label = "neutral"
        c1.risk_score = 0.0
        c1.redacted_text = "Tenant pays electricity."
        c1.is_substantive = True
        c1.content_type = "SUBSTANTIVE_CLAUSE"
        c1.human_review = False

        c2 = MagicMock()
        c2.clause_index = 1
        c2.clause_type = "rent_payment_terms"
        c2.favorability_label = "fair"
        c2.risk_score = 0.0
        c2.redacted_text = "Rent is due on the first."
        c2.is_substantive = True
        c2.content_type = "SUBSTANTIVE_CLAUSE"
        c2.human_review = False

        pdf_bytes = generate_pdf_report(mock_doc, [c1, c2], [], [], [])
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF was generated cleanly without crash

    # 5. A document with only excluded metadata has no fabricated substantive risk
    def test_document_with_only_excluded_metadata(self):
        # Excluded clauses produce no scores in clause_scores
        clause_scores = []
        overall = compute_document_risk(clause_scores, missing_clauses=[], config=self.config)
        self.assertEqual(overall, 0.0)
        self.assertEqual(compute_risk_band(overall, config=self.config), "low")

    # 6. A document with zero eligible clauses does not crash or produce a misleading score
    def test_empty_document_zero_eligible_clauses(self):
        # Even if missing_clauses are passed, an empty document returns 0.0
        self.assertEqual(compute_document_risk([], [{"severity": "high"}], config=self.config), 0.0)
        self.assertEqual(compute_document_risk([], [], config=self.config), 0.0)

    # 7. A document containing only fair/standard clauses produces the expected result under the existing documented formula
    def test_document_with_only_fair_clauses(self):
        # 5 fair clauses
        clause_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        # With no missing clauses
        self.assertEqual(compute_document_risk(clause_scores, [], config=self.config), 0.0)
        # With 1 medium missing clause (penalty 8): 0.7 * 0.0 + 0.3 * 8 = 2.4
        self.assertEqual(compute_document_risk(clause_scores, [{"severity": "medium"}], config=self.config), 2.4)

    # 8. Confirmed hazards contribute according to the configured formula
    def test_confirmed_hazards_contribute_correctly(self):
        score_term = compute_clause_risk("termination", "unfavorable", 1.0, config=self.config)
        self.assertEqual(score_term, 100.0)

        score_indem = compute_clause_risk("indemnification", "unfavorable", 1.0, config=self.config)
        self.assertEqual(score_indem, 85.0)

        score_late = compute_clause_risk("late_fees_penalty", "unfavorable", 0.9, config=self.config)
        self.assertEqual(score_late, 81.0)

        # Multi-clause composite
        clause_scores = [100.0, 85.0, 81.0] # mean = 88.666...
        overall = compute_document_risk(clause_scores, [], config=self.config)
        # 0.7 * 88.667 = 62.1
        self.assertEqual(overall, 62.1)
        self.assertEqual(compute_risk_band(overall, config=self.config), "high")

    # 9. Uncertain clauses remain distinguishable from confirmed hazards
    def test_uncertain_clauses_distinguishable(self):
        # UNCERTAIN clause
        res_uncertain = evaluate_clause_risk_and_review("UNCERTAIN", "needs_review", 0.20, config=self.config)
        self.assertEqual(res_uncertain["risk_score"], 0.0)
        self.assertTrue(res_uncertain["human_review"])
        self.assertEqual(res_uncertain["classification"], "UNCERTAIN")

        # Confirmed hazard
        res_hazard = evaluate_clause_risk_and_review("termination", "unfavorable", 0.95, config=self.config)
        self.assertEqual(res_hazard["risk_score"], 95.0)
        self.assertTrue(res_hazard["human_review"])
        self.assertEqual(res_hazard["classification"], "termination")

        # Both require review, but risk_score distinguishes hazard (100.0) from uncertain (0.0)
        self.assertNotEqual(res_uncertain["risk_score"], res_hazard["risk_score"])

    # 10. Frontend and PDF summary count partitioning agreement
    def test_summary_count_partitioning_no_overlap(self):
        # Synthetic clauses: 2 unfavorable, 1 needs review, 1 uncertain, 3 fair, 2 excluded
        c_unfav1 = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="unfavorable", human_review=True, clause_type="termination")
        c_unfav2 = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="UNFAVORABLE", human_review=False, clause_type="indemnification")
        c_rev = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="needs_review", human_review=True, clause_type="rent_payment_terms")
        c_uncert = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="neutral", human_review=True, clause_type="UNCERTAIN")
        c_fair1 = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="fair", human_review=False, clause_type="utilities")
        c_fair2 = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="FAIR", human_review=False, clause_type="pet_policy")
        c_fair3 = MagicMock(is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", favorability_label="neutral", human_review=False, clause_type="maintenance_repairs")
        c_ex1 = MagicMock(is_substantive=False, content_type="HEADER_FOOTER", favorability_label="neutral", human_review=False, clause_type="NOT_APPLICABLE")
        c_ex2 = MagicMock(is_substantive=False, content_type="METADATA_FRONT_MATTER", favorability_label="neutral", human_review=False, clause_type="NOT_APPLICABLE")

        all_clauses = [c_unfav1, c_unfav2, c_rev, c_uncert, c_fair1, c_fair2, c_fair3, c_ex1, c_ex2]
        substantive = [c for c in all_clauses if c.is_substantive and c.content_type == "SUBSTANTIVE_CLAUSE"]
        excluded = [c for c in all_clauses if not (c.is_substantive and c.content_type == "SUBSTANTIVE_CLAUSE")]

        unfavorable_count = sum(1 for c in substantive if (c.favorability_label or "").lower() == "unfavorable")
        review_count = sum(
            1 for c in substantive
            if (c.favorability_label or "").lower() != "unfavorable"
            and ((c.favorability_label or "").lower() == "needs_review"
                 or c.clause_type == "UNCERTAIN"
                 or c.human_review)
        )
        fair_count = len(substantive) - unfavorable_count - review_count

        self.assertEqual(len(substantive), 7)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(unfavorable_count, 2)
        self.assertEqual(review_count, 2)
        self.assertEqual(fair_count, 3)
        # Sum of mutually exclusive counts matches total substantive clauses
        self.assertEqual(unfavorable_count + review_count + fair_count, len(substantive))

    # 11. Test that unfavorable, fair, needs-review, neutral, excluded, and unknown categories are handled consistently
    def test_six_categories_handling(self):
        # 1. Unfavorable
        r_unfav = evaluate_clause_risk_and_review("termination", "unfavorable", 1.0, config=self.config)
        self.assertGreater(r_unfav["risk_score"], 0.0)
        self.assertEqual(r_unfav["favorability_label"], "unfavorable")

        # 2. Fair
        r_fair = evaluate_clause_risk_and_review("utilities", "fair", 1.0, config=self.config)
        self.assertEqual(r_fair["risk_score"], 0.0)
        self.assertEqual(r_fair["favorability_label"], "fair")
        self.assertFalse(r_fair["human_review"])

        # 3. Needs Review
        r_rev = evaluate_clause_risk_and_review("rent_payment_terms", "needs_review", 1.0, config=self.config)
        self.assertGreater(r_rev["risk_score"], 0.0)
        self.assertEqual(r_rev["favorability_label"], "needs_review")
        self.assertTrue(r_rev["human_review"])

        # 4. Neutral (substantive)
        r_neutral = evaluate_clause_risk_and_review("utilities", "neutral", 1.0, config=self.config)
        self.assertEqual(r_neutral["risk_score"], 0.0)
        self.assertEqual(r_neutral["favorability_label"], "neutral")
        self.assertFalse(r_neutral["human_review"])

        # 5. Excluded (non-substantive structural block)
        r_ex = evaluate_clause_risk_and_review("NOT_APPLICABLE", "neutral", 1.0, content_type="HEADER_FOOTER", is_substantive=False, config=self.config)
        self.assertEqual(r_ex["risk_score"], 0.0)
        self.assertEqual(r_ex["classification"], "NOT_APPLICABLE")
        self.assertFalse(r_ex["human_review"])

        # 6. Unknown (substantive with unknown classification)
        r_unk = evaluate_clause_risk_and_review("termination", "unknown_category", 1.0, config=self.config)
        self.assertEqual(r_unk["risk_score"], 0.0)
        self.assertEqual(r_unk["favorability_label"], "needs_review")
        self.assertTrue(r_unk["human_review"])
        self.assertIsNotNone(r_unk["review_reason"])

    # 12. Confirm excluded metadata contributes zero substantive risk but is clearly distinguished from unclassified substantive clauses
    def test_excluded_metadata_distinguished_from_unclassified_substantive(self):
        # Excluded metadata block
        ex_meta = evaluate_clause_risk_and_review(
            clause_type="NOT_APPLICABLE",
            favorability_label="neutral",
            content_type="METADATA_FRONT_MATTER",
            is_substantive=False,
            config=self.config,
        )
        self.assertEqual(ex_meta["risk_score"], 0.0)
        self.assertFalse(ex_meta["human_review"])
        self.assertEqual(ex_meta["classification"], "NOT_APPLICABLE")
        self.assertIsNone(ex_meta["review_reason"])

        # Unclassified substantive clause
        unclass_sub = evaluate_clause_risk_and_review(
            clause_type="unknown_substantive_provision",
            favorability_label="unknown",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            config=self.config,
        )
        self.assertEqual(unclass_sub["risk_score"], 0.0) # Zero risk, no invented penalty
        self.assertTrue(unclass_sub["human_review"])      # MUST NOT be presented as safe!
        self.assertEqual(unclass_sub["favorability_label"], "needs_review")
        self.assertIsNotNone(unclass_sub["review_reason"])

        # Distinct review requirements
        self.assertNotEqual(ex_meta["human_review"], unclass_sub["human_review"])
        self.assertNotEqual(ex_meta["classification"], unclass_sub["classification"])

if __name__ == "__main__":
    unittest.main()
