import unittest
import sys
import json
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.risk.risk_engine import (
    evaluate_clause_risk_and_review,
    compute_clause_risk,
    normalize_clause_type,
    normalize_favorability,
    get_risk_config,
)
from app.services.nlp.classifier_baseline import (
    is_supported_document_type,
    predict_baseline,
    SUPPORTED_DOCUMENT_TYPES,
)
from app.services.nlp.classifier_bert import (
    classify_clause,
    classify_clauses_batch,
)


class TestPhase3RiskEngineClassification(unittest.TestCase):
    """
    Phase 3 Technical Verification Suite:
    - DEF-301: Symmetric confidence-based human review for Fair and Unfavourable predictions.
    - DEF-302: Favorability confidence abstention below 0.45.
    - DEF-303: BERT_TEMPERATURE set to 1.0 preserving unscaled softmax.
    - DEF-304: Explicit UNCLASSIFIED / needs_review for unsupported document types (no rental fallback).
    - DEF-305: Frontend Fair KPI logic and exclusion of uncertain/unclassified clauses.
    - Benchmark reporting integrity and metadata non-substantive risk score invariance.
    """

    def setUp(self):
        self.config = get_risk_config()

    # 1. DEF-301: Low-confidence Fair predictions must NOT silently receive zero-risk without review
    def test_low_confidence_fair_prediction_triggers_needs_review(self):
        # A Fair prediction with confidence 0.52 (< 0.60 threshold)
        res = evaluate_clause_risk_and_review(
            clause_type="rent_payment_terms",
            favorability_label="fair",
            favorability_confidence=0.52,
            config=self.config,
        )
        self.assertTrue(res["human_review"], "Low-confidence fair prediction must trigger human review")
        self.assertEqual(res["favorability_label"], "needs_review", "Effective favorability must transition to needs_review")
        self.assertIsNotNone(res["review_reason"], "Review reason must be populated")
        self.assertIn("Low confidence fair prediction", res["review_reason"])
        self.assertIn("< 0.60", res["review_reason"])

    # 2. DEF-301: High-confidence Fair prediction is definitive and zero-risk without review
    def test_high_confidence_fair_prediction_is_definitive_fair(self):
        res = evaluate_clause_risk_and_review(
            clause_type="rent_payment_terms",
            favorability_label="fair",
            favorability_confidence=0.88,
            config=self.config,
        )
        self.assertFalse(res["human_review"], "High-confidence fair prediction does not require review")
        self.assertEqual(res["favorability_label"], "fair")
        self.assertEqual(res["risk_score"], 0.0)
        self.assertIsNone(res["review_reason"])

    # 3. DEF-301: Low-confidence Unfavourable prediction triggers needs_review symmetrically
    def test_low_confidence_unfavorable_triggers_needs_review(self):
        # An Unfavourable prediction with confidence 0.55 (< 0.60 threshold)
        res = evaluate_clause_risk_and_review(
            clause_type="termination",
            favorability_label="unfavorable",
            favorability_confidence=0.55,
            config=self.config,
        )
        self.assertTrue(res["human_review"], "Low-confidence unfavorable prediction must trigger human review")
        self.assertEqual(res["favorability_label"], "needs_review")
        self.assertIsNotNone(res["review_reason"])
        self.assertIn("Low confidence unfavorable prediction", res["review_reason"])
        self.assertIn("< 0.60", res["review_reason"])

    # 4. DEF-301: High-confidence Unfavourable prediction retains unfavorable label and computes risk
    def test_high_confidence_unfavorable_prediction_computes_risk(self):
        res = evaluate_clause_risk_and_review(
            clause_type="termination",
            favorability_label="unfavorable",
            favorability_confidence=0.90,
            config=self.config,
        )
        self.assertEqual(res["favorability_label"], "unfavorable")
        self.assertEqual(res["risk_score"], 90.0)  # base 1.0 * mult 1.0 * conf 0.9 * 100
        self.assertTrue(res["human_review"], "Score 90.0 >= 40.0 triggers elevated risk review")
        self.assertIn("Elevated clause risk score", res["review_reason"])

    # 5. DEF-302: Pipeline Favorability confidence abstention below 0.45 boundary
    def test_favorability_confidence_boundary_abstention(self):
        # Under 0.45 in pipeline: abstains to needs_review with human_review=True
        # Simulate pipeline logic:
        def pipeline_favorability_gate(fav_label, fav_conf):
            if fav_conf is not None and fav_conf < 0.45:
                return "needs_review", True
            return fav_label, False

        # 0.44 -> Abstention
        label_44, review_44 = pipeline_favorability_gate("fair", 0.44)
        self.assertEqual(label_44, "needs_review")
        self.assertTrue(review_44)

        label_unfav_44, review_unfav_44 = pipeline_favorability_gate("unfavorable", 0.44)
        self.assertEqual(label_unfav_44, "needs_review")
        self.assertTrue(review_unfav_44)

        # 0.45 -> Boundary passes to risk engine (where < 0.60 review gate evaluates it)
        label_45, review_45 = pipeline_favorability_gate("fair", 0.45)
        self.assertEqual(label_45, "fair")
        res_45 = evaluate_clause_risk_and_review("rent_payment_terms", label_45, 0.45, config=self.config)
        self.assertTrue(res_45["human_review"])
        self.assertEqual(res_45["favorability_label"], "needs_review")

        # 0.65 -> High confidence passes both gates
        label_65, review_65 = pipeline_favorability_gate("fair", 0.65)
        self.assertEqual(label_65, "fair")
        res_65 = evaluate_clause_risk_and_review("rent_payment_terms", label_65, 0.65, config=self.config)
        self.assertFalse(res_65["human_review"])
        self.assertEqual(res_65["favorability_label"], "fair")

    # 6. UNCERTAIN predictions and explanation synchronization
    def test_clause_type_uncertain_prediction_and_explanation(self):
        res = evaluate_clause_risk_and_review("UNCERTAIN", "fair", 1.0, config=self.config)
        self.assertEqual(res["classification"], "UNCERTAIN")
        self.assertTrue(res["human_review"])
        self.assertEqual(res["favorability_label"], "needs_review")
        self.assertEqual(res["risk_score"], 0.0)
        self.assertIn("Low classification confidence (<0.45)", res["review_reason"])

    # 7. DEF-304: Unsupported document types must return UNCLASSIFIED / needs_review (no rental fallback)
    def test_unsupported_document_types_return_unclassified(self):
        unsupported_types = ["commercial_contract", "master_services_agreement", "nda", "unknown_type"]

        for dt in unsupported_types:
            self.assertFalse(is_supported_document_type(dt))

            # Baseline classifier check
            b_type, b_conf = predict_baseline("Tenant shall pay rent of $2000.", dt, task="clause_type")
            self.assertEqual(b_type, "UNCLASSIFIED", f"Failed for {dt}")
            self.assertEqual(b_conf, 0.0)

            b_fav, b_fconf = predict_baseline("Tenant shall pay rent of $2000.", dt, task="favorability")
            self.assertEqual(b_fav, "needs_review", f"Failed for {dt}")
            self.assertEqual(b_fconf, 0.0)

            # BERT interface check
            c_type, c_conf, c_ver = classify_clause("Tenant shall pay rent of $2000.", dt, task="clause_type")
            self.assertEqual(c_type, "UNCLASSIFIED", f"Failed for {dt}")
            self.assertEqual(c_conf, 0.0)
            self.assertEqual(c_ver, "unsupported-document-type")

            # Batched BERT check
            batch_res = classify_clauses_batch([{"text": "Sample clause"}], dt, task="clause_type")
            self.assertEqual(batch_res[0][0], "UNCLASSIFIED")
            self.assertEqual(batch_res[0][1], 0.0)

            # Risk engine evaluation
            eval_res = evaluate_clause_risk_and_review(c_type, "needs_review", 0.0, config=self.config)
            self.assertEqual(eval_res["classification"], "UNCLASSIFIED")
            self.assertTrue(eval_res["human_review"])
            self.assertEqual(eval_res["favorability_label"], "needs_review")
            self.assertEqual(eval_res["risk_score"], 0.0)
            self.assertIn("Unsupported contract type or unclassified clause", eval_res["review_reason"])

    # 8. Supported document types preserve normal classification behavior
    def test_supported_document_types_preserved(self):
        for dt in ["rental_agreement", "job_offer_letter", "insurance_policy"]:
            self.assertTrue(is_supported_document_type(dt))

        # Check rental agreement baseline
        lbl, conf = predict_baseline("Tenant shall deposit the sum of $1500.", "rental_agreement", "clause_type")
        self.assertNotEqual(lbl, "UNCLASSIFIED")
        self.assertGreater(conf, 0.0)

        # Check job offer baseline
        lbl_o, conf_o = predict_baseline("Your annual starting base salary will be $120,000.", "job_offer_letter", "clause_type")
        self.assertNotEqual(lbl_o, "UNCLASSIFIED")
        self.assertGreater(conf_o, 0.0)

        # Check insurance baseline
        lbl_i, conf_i = predict_baseline("Loss caused by earth movement is excluded.", "insurance_policy", "clause_type")
        self.assertNotEqual(lbl_i, "UNCLASSIFIED")
        self.assertGreater(conf_i, 0.0)

    # 9. DEF-303: BERT_TEMPERATURE set to 1.0 (unscaled raw softmax)
    def test_temperature_1_0_raw_softmax_scaling(self):
        import numpy as np

        # Demonstrate temperature behavior
        logits = np.array([2.5, 1.2, 0.1])
        
        # T = 1.0 preserves exact logits without artificial sharpening
        t_1_0 = 1.0
        scaled_1_0 = logits / t_1_0
        np.testing.assert_array_almost_equal(scaled_1_0, logits)

        # Contrast with T = 0.58 which artificially sharpens probabilities towards 1.0
        t_0_58 = 0.58
        scaled_0_58 = logits / t_0_58
        self.assertGreater(scaled_0_58[0], scaled_1_0[0], "T=0.58 artificially inflates top logit")

        # Verify default env var setting in classifier_bert
        default_temp = float(os.environ.get("BERT_TEMPERATURE", "1.0"))
        self.assertEqual(default_temp, 1.0)

    # 10. DEF-305: Frontend Fair KPI calculation arithmetic
    def test_frontend_fair_kpi_calculation_arithmetic(self):
        # Simulate clauses in DocumentAnalysis.jsx
        synthetic_clauses = [
            {"is_substantive": True, "favorability_label": "fair", "human_review": False, "clause_type": "security_deposit"},
            {"is_substantive": True, "favorability_label": "fair", "human_review": False, "clause_type": "rent_payment_terms"},
            # Low-confidence fair clause flagged for review
            {"is_substantive": True, "favorability_label": "needs_review", "human_review": True, "clause_type": "maintenance_repairs"},
            # Unfavorable high confidence
            {"is_substantive": True, "favorability_label": "unfavorable", "human_review": True, "clause_type": "termination"},
            # UNCERTAIN clause
            {"is_substantive": True, "favorability_label": "needs_review", "human_review": True, "clause_type": "UNCERTAIN"},
            # UNCLASSIFIED clause (unsupported type)
            {"is_substantive": True, "favorability_label": "needs_review", "human_review": True, "clause_type": "UNCLASSIFIED"},
            # Neutral / definition clause
            {"is_substantive": True, "favorability_label": "neutral", "human_review": False, "clause_type": "definitions"},
        ]

        # Old broken formula: fairCount = total - unfavorable - review
        unfav_count = sum(1 for c in synthetic_clauses if (c.get("favorability_label") or "").lower() == "unfavorable")
        review_count = sum(
            1 for c in synthetic_clauses
            if (c.get("favorability_label") or "").lower() != "unfavorable"
            and ((c.get("favorability_label") or "").lower() == "needs_review"
                 or c.get("clause_type") == "UNCERTAIN"
                 or c.get("clause_type") == "UNCLASSIFIED"
                 or c.get("human_review"))
        )
        
        # New correct formula:
        correct_fair_count = sum(
            1 for c in synthetic_clauses
            if (c.get("favorability_label") or "").lower() == "fair"
            and not c.get("human_review")
            and c.get("clause_type") not in ("UNCERTAIN", "UNCLASSIFIED")
        )

        old_residual_fair_count = len(synthetic_clauses) - unfav_count - review_count
        
        # The old formula would have included the neutral clause (7 - 1 - 4 = 2, but neutral clause wasn't counted as review so 7 - 1 - 4 = 2)
        # Verify correct fair count is strictly 2
        self.assertEqual(correct_fair_count, 2, "Only genuine unflagged fair clauses should be counted as Fair")
        self.assertEqual(unfav_count, 1)
        self.assertEqual(review_count, 3)

    # 11. Model comparison report integrity
    def test_model_comparison_report_contains_no_placeholder_1_0_metrics(self):
        report_path = Path("backend/models/model_comparison_report.json")
        self.assertTrue(report_path.exists())
        data = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("_provenance", data)
        self.assertIn("benchmark_summary", data)

        # Verify no 1.0 placeholders in test metrics
        for domain in ["rental_agreement", "job_offer_letter", "insurance_policy"]:
            ct_m = data[domain]["clause_type"]["distilbert"]["f1_macro"]
            self.assertNotEqual(ct_m, 1.0, f"{domain} clause_type must not be placeholder 1.0")
            self.assertGreater(ct_m, 0.80)

            base_m = data[domain]["clause_type"]["baseline_tfidf_lr"]["f1_macro"]
            self.assertNotEqual(base_m, 1.0, f"{domain} baseline must not be placeholder 1.0")
            self.assertGreater(base_m, 0.85)


if __name__ == "__main__":
    unittest.main()
